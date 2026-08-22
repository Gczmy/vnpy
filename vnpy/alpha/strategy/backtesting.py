from collections import defaultdict
from datetime import date, datetime
from copy import copy
from typing import cast
import json
import traceback

import numpy as np
import polars as pl
import plotly.graph_objects as go               # type: ignore
from plotly.subplots import make_subplots       # type: ignore
from tqdm import tqdm

from vnpy.trader.constant import Direction, Offset, Interval, Status, Exchange
from vnpy.trader.object import OrderData, TradeData, BarData
from vnpy.trader.utility import round_to, extract_vt_symbol

from ..logger import logger
from ..lab import AlphaLab
from .template import AlphaStrategy


# ================= A股交易现实约束 (与券商/交易所规则一致) =================
# 印花税: 2023-08-28 起由千1 减半为万5 (单边卖出)
STAMP_TAX_HALVING_DATE = date(2023, 8, 28)
STAMP_TAX_AFTER_HALVING = 0.0005
# 创业板注册制: 2020-08-24 起涨跌幅 10% -> 20%
CYB_20PCT_DATE = datetime(2020, 8, 24)
# 有涨跌停的交易所 (A股); 加密货币 24h 连续交易无涨跌停
A_SHARE_EXCHANGES = {Exchange.SSE, Exchange.SZSE, Exchange.BSE}


def _effective_stamp_rate(
    long_rate: float,
    short_rate: float,
    dt: datetime | date | None,
    stamp_halving: bool = True,
) -> float:
    """有效印花税率: 2023-08-28 起为万5, 此前为 contract 里 short_rate - long_rate (千1)。"""
    rate: float = short_rate - long_rate
    if rate > 0 and stamp_halving and dt is not None:
        d: date = dt.date() if isinstance(dt, datetime) else dt
        if d >= STAMP_TAX_HALVING_DATE:
            rate = min(rate, STAMP_TAX_AFTER_HALVING)
    return rate


class BacktestingEngine:
    """Alpha strategy backtesting engine"""

    gateway_name: str = "BACKTESTING"

    def __init__(self, lab: AlphaLab) -> None:
        """Constructor"""
        self.lab: AlphaLab = lab

        self.vt_symbols: list[str] = []
        self.start: datetime
        self.end: datetime

        self.long_rates: dict[str, float] = {}
        self.short_rates: dict[str, float] = {}
        self.sizes: dict[str, float] = {}
        self.priceticks: dict[str, float] = {}

        self.capital: float = 0
        self.risk_free: float = 0
        self.annual_days: int = 0

        self.strategy_class: type[AlphaStrategy]
        self.strategy: AlphaStrategy
        self.bars: dict[str, BarData] = {}
        self.datetime: datetime | None = None

        self.interval: Interval
        self.history_data: dict[tuple, BarData] = {}
        self.dts: set[datetime] = set()

        self.limit_order_count: int = 0
        self.limit_orders: dict[str, OrderData] = {}
        self.active_limit_orders: dict[str, OrderData] = {}

        self.trade_count: int = 0
        self.trades: dict[str, TradeData] = {}

        self.logs: list[str] = []

        self.daily_results: dict[date, PortfolioDailyResult] = {}
        self.daily_df: pl.DataFrame

        self.pre_closes: defaultdict = defaultdict(float)

        self.cash: float = 0
        self.signal_df: pl.DataFrame

        self.slippage: float = 0.0005    # 滑点比例 (默认万5 = 现实默认; 0=无)
        self.min_commission: float = 5.0  # 单笔最低佣金 (默认 5 元 = 现实券商; 0=无下限)

        # 现实约束开关 (默认全开 = A股现实; 供 A/B 实验诊断用)
        self.use_price_limits: bool = True   # 涨跌停限制 (板块/日期感知)
        self.block_zero_volume: bool = True  # 停牌/无量日不成交
        self.stamp_tax_halving: bool = True  # 2023-08-28 起印花税减半为万5

        self.stamp_tax: float = 0         # 印花税累计 (卖出额 × 有效印花税率)
        self.slippage_cost: float = 0     # 滑点成本累计 (滑点前后价差 × 量 × 乘数)

    def set_parameters(
        self,
        vt_symbols: list[str],
        interval: Interval,
        start: datetime,
        end: datetime,
        capital: int = 1_000_000,
        risk_free: float = 0,
        annual_days: int = 240,
        slippage: float = 0.0005,
        min_commission: float = 5.0,
        use_price_limits: bool = True,
        block_zero_volume: bool = True,
        stamp_tax_halving: bool = True
    ) -> None:
        """Set parameters

        ⚠️ 默认按 A股现实成本与约束: 滑点万5 + 最低佣金 5 元/笔 + 涨跌停 + 停牌不成交
        + 印花税 2023-08-28 起减半。
        零成本 (滑点=0, 最低佣金=0) 或关闭约束需显式传参, 零成本会在统计时输出警告。
        """
        self.vt_symbols = vt_symbols
        self.interval = interval

        self.start = start
        self.end = end
        self.capital = capital
        self.risk_free = risk_free
        self.annual_days = annual_days
        self.slippage = slippage
        self.min_commission = min_commission
        self.use_price_limits = use_price_limits
        self.block_zero_volume = block_zero_volume
        self.stamp_tax_halving = stamp_tax_halving

        self.cash = capital

        contract_settings: dict = self.lab.load_contract_setttings()
        for vt_symbol in vt_symbols:
            setting: dict | None = contract_settings.get(vt_symbol, None)
            if not setting:
                logger.warning(f"找不到合约{vt_symbol}的交易配置，请检查！")
                continue

            self.long_rates[vt_symbol] = setting["long_rate"]
            self.short_rates[vt_symbol] = setting["short_rate"]
            self.sizes[vt_symbol] = setting["size"]
            self.priceticks[vt_symbol] = setting["pricetick"]

    def add_strategy(self, strategy_class: type, setting: dict, signal_df: pl.DataFrame) -> None:
        """Add strategy"""
        self.strategy_class = strategy_class
        self.strategy = strategy_class(
            self, strategy_class.__name__, copy(self.vt_symbols), setting
        )
        self.signal_df = signal_df

    def load_data(self) -> None:
        """Load historical data"""
        logger.info("开始加载历史数据")

        if not self.end:
            self.end = datetime.now()

        if self.start >= self.end:
            logger.info("起始日期必须小于结束日期")
            return

        # Clear previously loaded historical data
        self.history_data.clear()
        self.dts.clear()

        # Load historical data for each symbol
        empty_symbols: list[str] = []
        for vt_symbol in tqdm(self.vt_symbols, total=len(self.vt_symbols)):
            data: list[BarData] = self.lab.load_bar_data(
                vt_symbol,
                self.interval,
                self.start,
                self.end
            )

            for bar in data:
                self.dts.add(bar.datetime)
                self.history_data[(bar.datetime, vt_symbol)] = bar

            data_count = len(data)
            if not data_count:
                empty_symbols.append(vt_symbol)

        if empty_symbols:
            logger.info(f"部分合约历史数据为空：{empty_symbols}")

        logger.info("所有历史数据加载完成")

    def run_backtesting(self) -> None:
        """Start backtesting"""
        self.strategy.on_init()
        logger.info("策略初始化完成")

        # Use remaining historical data for strategy backtesting
        dts: list = list(self.dts)
        dts.sort()

        logger.info("开始回放历史数据")
        for dt in dts:
            try:
                self.new_bars(dt)
            except Exception:
                logger.info("触发异常，回测终止")
                logger.info(traceback.format_exc())
                return

        logger.info("历史数据回放结束")

    def calculate_result(self, with_positions: bool = False) -> pl.DataFrame | None:
        """Calculate daily mark-to-market profit and loss

        with_positions=True 时额外输出每日持仓明细列 (funding 叠加层用):
          - pos_detail: 该日收盘持仓 dict {vt_symbol: 数量}
          - close_detail: 该日收盘价 dict {vt_symbol: 价格}
        不改动任何 pnl 计算, 仅在返回 DataFrame 增加两列 (默认关闭, 零回归)。
        """
        logger.info("开始计算逐日盯市盈亏")

        if not self.trades:
            logger.info("成交记录为空，仅计算持仓盈亏")
        else:
            for trade in self.trades.values():
                if not trade.datetime:
                    continue

                d: date = trade.datetime.date()
                daily_result: PortfolioDailyResult = self.daily_results[d]
                daily_result.add_trade(trade)

        pre_closes: dict[str, float] = {}
        start_poses: dict[str, float] = {}

        for daily_result in self.daily_results.values():
            daily_result.calculate_pnl(
                pre_closes,
                start_poses,
                self.sizes,
                self.long_rates,
                self.short_rates,
                self.min_commission,
                self.stamp_tax_halving
            )

            pre_closes = daily_result.close_prices
            start_poses = daily_result.end_poses

        results: dict = defaultdict(list)

        for daily_result in self.daily_results.values():
            fields: list = [
                "date", "trade_count", "turnover",
                "commission", "trading_pnl",
                "holding_pnl", "total_pnl", "net_pnl"
            ]
            for key in fields:
                value = getattr(daily_result, key)
                results[key].append(value)
            if with_positions:
                # 持仓明细存 JSON 字符串 (polars 对 dict 列表首项为空时推断 Struct({}) 全空,
                # 见 funding_overlay 调试; JSON 字符串最可靠)
                results["pos_detail"].append(
                    json.dumps({k: v for k, v in daily_result.end_poses.items()
                                if abs(v) > 1e-12}, ensure_ascii=False))
                results["close_detail"].append(
                    json.dumps(dict(daily_result.close_prices), ensure_ascii=False))

        # 初始化 daily_df (防止无交易时未定义)
        self.daily_df = pl.DataFrame()
        
        if results:
            self.daily_df = pl.DataFrame([
                pl.Series("date", results["date"], dtype=pl.Date),
                pl.Series("trade_count", results["trade_count"], dtype=pl.Int64),
                pl.Series("turnover", results["turnover"], dtype=pl.Float64),
                pl.Series("commission", results["commission"], dtype=pl.Float64),
                pl.Series("trading_pnl", results["trading_pnl"], dtype=pl.Float64),
                pl.Series("holding_pnl", results["holding_pnl"], dtype=pl.Float64),
                pl.Series("total_pnl", results["total_pnl"], dtype=pl.Float64),
                pl.Series("net_pnl", results["net_pnl"], dtype=pl.Float64),
            ])
            if with_positions:
                self.daily_df = self.daily_df.with_columns([
                    pl.Series("pos_detail", results["pos_detail"]),
                    pl.Series("close_detail", results["close_detail"]),
                ])

        logger.info("逐日盯市盈亏计算完成")
        return self.daily_df

    def calculate_statistics(self) -> dict:
        """Calculate strategy statistics"""
        logger.info("开始计算策略统计指标")

        # 零成本警示: 与现实不符, 不允许作为默认口径
        if self.slippage <= 0 and self.min_commission <= 0:
            logger.warning("⚠️ 零成本回测 (滑点=0 且最低佣金=0), 结果与现实不符, 仅用于对照实验!")

        # Initialize statistics
        start_date: str = ""
        end_date: str = ""
        total_days: int = 0
        profit_days: int = 0
        loss_days: int = 0
        end_balance: float = 0
        max_drawdown: float = 0
        max_ddpercent: float = 0
        max_drawdown_duration: int = 0
        total_net_pnl: float = 0
        daily_net_pnl: float = 0
        total_commission: float = 0
        daily_commission: float = 0
        total_turnover: float = 0
        daily_turnover: float = 0
        total_trade_count: int = 0
        daily_trade_count: float = 0
        daily_turnover_rate: float = 0
        total_return: float = 0
        annual_return: float = 0
        daily_return: float = 0
        return_std: float = 0
        sharpe_ratio: float = 0
        return_drawdown_ratio: float = 0

        # Check if bankruptcy occurred
        positive_balance: bool = False

        # Calculate capital-related metrics
        df: pl.DataFrame = self.daily_df

        # C3: load_data 一无所获时 daily_df 为空 (Null-schema) DataFrame,
        # with_columns/取列会抛 ColumnNotFoundError, 直接返回空统计
        if df is not None and not df.is_empty():
            df = df.with_columns(
                # Strategy capital
                balance=pl.col("net_pnl").cum_sum() + self.capital
            ).with_columns(
                # Strategy return
                pl.col("balance").pct_change().fill_null(0).alias("return"),
                # Capital high watermark
                highlevel=pl.col("balance").cum_max()
            ).with_columns(
                # Capital drawdown
                drawdown=pl.col("balance") - pl.col("highlevel"),
                # Percentage drawdown
                ddpercent=(pl.col("balance") / pl.col("highlevel") - 1) * 100
            )

            # Check if bankruptcy occurred
            positive_balance = (df["balance"] > 0).all()
            if not positive_balance:
                logger.info("回测中出现爆仓（资金小于等于0），无法计算策略统计指标")

            # Save data object
            self.daily_df = df

        # Calculate statistics
        if positive_balance:
            start_date = df["date"][0]
            end_date = df["date"][-1]

            total_days = len(df)
            profit_days = df.filter(pl.col("net_pnl") > 0).height
            loss_days = df.filter(pl.col("net_pnl") < 0).height

            end_balance = df["balance"][-1]
            max_drawdown = cast(float, df["drawdown"].min())
            max_ddpercent = cast(float, df["ddpercent"].min())

            max_drawdown_end_idx = cast(int, df["drawdown"].arg_min())
            max_drawdown_end = df["date"][max_drawdown_end_idx]

            if isinstance(max_drawdown_end, date):
                max_drawdown_start_idx = cast(int, df.slice(0, max_drawdown_end_idx + 1)["balance"].arg_max())
                max_drawdown_start = df["date"][max_drawdown_start_idx]
                max_drawdown_duration = (max_drawdown_end - max_drawdown_start).days
            else:
                max_drawdown_duration = 0

            total_net_pnl = cast(float, df["net_pnl"].sum())
            daily_net_pnl = total_net_pnl / total_days

            total_commission = cast(float, df["commission"].sum())
            daily_commission = total_commission / total_days

            total_turnover = cast(float, df["turnover"].sum())
            daily_turnover = total_turnover / total_days

            total_trade_count = cast(int, df["trade_count"].sum())
            daily_trade_count = total_trade_count / total_days

            # 日均换手率 = 日均成交金额 / 平均持仓市值 (近似用平均净值)
            avg_balance = (self.capital + end_balance) / 2
            daily_turnover_rate = (daily_turnover / avg_balance * 100) if avg_balance else 0

            total_return = (end_balance / self.capital - 1) * 100
            annual_return = total_return / total_days * self.annual_days
            daily_return = cast(float, df["return"].mean()) * 100
            return_std = cast(float, df["return"].std()) * 100

            if return_std and self.annual_days > 0:
                # C2: 年化→日化应除以 annual_days (非平方根); daily_return 已是
                # 百分数而 risk_free 惯例是小数 (如 0.03 = 3%), 统一 ×100。
                daily_risk_free = (self.risk_free * 100) / self.annual_days
                sharpe_ratio = (daily_return - daily_risk_free) / return_std * np.sqrt(self.annual_days)
            else:
                sharpe_ratio = 0

            return_drawdown_ratio = (-total_net_pnl / max_drawdown) if max_drawdown else 0

        # Output results
        logger.info("-" * 30)
        logger.info(f"首个交易日：  {start_date}")
        logger.info(f"最后交易日：  {end_date}")

        logger.info(f"总交易日：  {total_days}")
        logger.info(f"盈利交易日：  {profit_days}")
        logger.info(f"亏损交易日：  {loss_days}")

        logger.info(f"起始资金：  {self.capital:,.2f}")
        logger.info(f"结束资金：  {end_balance:,.2f}")

        logger.info(f"总收益率：  {total_return:,.2f}%")
        logger.info(f"年化收益：  {annual_return:,.2f}%")
        logger.info(f"最大回撤:   {max_drawdown:,.2f}")
        logger.info(f"百分比最大回撤: {max_ddpercent:,.2f}%")
        logger.info(f"最长回撤天数:   {max_drawdown_duration}")

        logger.info(f"总盈亏：  {total_net_pnl:,.2f}")
        logger.info(f"总手续费：  {total_commission:,.2f}")
        logger.info(f"  其中佣金：  {total_commission - self.stamp_tax:,.2f}")
        logger.info(f"  其中印花税：  {self.stamp_tax:,.2f}")
        logger.info(f"滑点成本：  {self.slippage_cost:,.2f}")
        logger.info(f"总成交金额：  {total_turnover:,.2f}")
        logger.info(f"总成交笔数：  {total_trade_count}")

        logger.info(f"日均盈亏：  {daily_net_pnl:,.2f}")
        logger.info(f"日均手续费：  {daily_commission:,.2f}")
        logger.info(f"日均成交金额：  {daily_turnover:,.2f}")
        logger.info(f"日均成交笔数：  {daily_trade_count}")
        logger.info(f"日均换手率：  {daily_turnover_rate:,.2f}%")
        logger.info(f"滑点假设：  {self.slippage:.6f}")
        logger.info(f"最低佣金：  {self.min_commission:.2f} 元/笔")

        logger.info(f"日均收益率：  {daily_return:,.2f}%")
        logger.info(f"收益标准差：  {return_std:,.2f}%")
        logger.info(f"Sharpe Ratio：  {sharpe_ratio:,.2f}")
        logger.info(f"收益回撤比：  {return_drawdown_ratio:,.2f}")

        statistics: dict = {
            "start_date": start_date,
            "end_date": end_date,
            "total_days": total_days,
            "profit_days": profit_days,
            "loss_days": loss_days,
            "capital": self.capital,
            "end_balance": end_balance,
            "max_drawdown": max_drawdown,
            "max_ddpercent": max_ddpercent,
            "max_drawdown_duration": max_drawdown_duration,
            "total_net_pnl": total_net_pnl,
            "daily_net_pnl": daily_net_pnl,
            "total_commission": total_commission,
            "daily_commission": daily_commission,
            "total_turnover": total_turnover,
            "daily_turnover": daily_turnover,
            "total_trade_count": total_trade_count,
            "daily_trade_count": daily_trade_count,
            "daily_turnover_rate": daily_turnover_rate,
            "slippage": self.slippage,
            "min_commission": self.min_commission,
            "total_commission_fee": total_commission - self.stamp_tax,
            "total_stamp_tax": self.stamp_tax,
            "total_slippage_cost": self.slippage_cost,
            "total_return": total_return,
            "annual_return": annual_return,
            "daily_return": daily_return,
            "return_std": return_std,
            "sharpe_ratio": sharpe_ratio,
            "return_drawdown_ratio": return_drawdown_ratio,
        }

        # Filter extreme values
        for key, value in statistics.items():
            if value in (np.inf, -np.inf):
                value = 0
            statistics[key] = np.nan_to_num(value)

        logger.info("策略统计指标计算完成")
        return statistics

    def show_chart(self) -> None:
        """Display chart"""
        df: pl.DataFrame = self.daily_df

        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=["Balance", "Drawdown", "Daily Pnl", "Pnl Distribution"],
            vertical_spacing=0.06
        )

        balance_line = go.Scatter(
            x=df["date"],
            y=df["balance"],
            mode="lines",
            name="Balance"
        )
        drawdown_scatter = go.Scatter(
            x=df["date"],
            y=df["drawdown"],
            fillcolor="red",
            fill='tozeroy',
            mode="lines",
            name="Drawdown"
        )
        pnl_bar = go.Bar(y=df["net_pnl"], name="Daily Pnl")
        pnl_histogram = go.Histogram(x=df["net_pnl"], nbinsx=100, name="Days")

        fig.add_trace(balance_line, row=1, col=1)
        fig.add_trace(drawdown_scatter, row=2, col=1)
        fig.add_trace(pnl_bar, row=3, col=1)
        fig.add_trace(pnl_histogram, row=4, col=1)

        fig.update_layout(height=1000, width=1000)
        fig.show()

    def show_performance(self, benchmark_symbol: str) -> None:
        """Display performance metrics"""
        # Load benchmark prices
        benchmark_bars: list[BarData] = self.lab.load_bar_data(benchmark_symbol, self.interval, self.start, self.end)

        benchmark_prices: list[float] = []
        for bar in benchmark_bars:
            benchmark_prices.append(bar.close_price)

        # Calculate strategy performance
        performance_df: pl.DataFrame = (
            self.daily_df.with_columns(
                # Cumulative return
                cumulative_return=pl.col("balance").pct_change().cum_sum(),
                # Cumulative cost
                cumulative_cost=(pl.col("commission") / pl.col("balance").shift(1)).cum_sum()
            ).with_columns(
                # Benchmark price
                benchmark_price=pl.Series(values=benchmark_prices, dtype=pl.Float64)
            ).with_columns(
                # Benchmark return
                benchmark_return=pl.col("benchmark_price").pct_change().cum_sum()
            ).with_columns(
                # Excess return
                excess_return=(pl.col("cumulative_return") - pl.col("benchmark_return"))
            ).with_columns(
                # Net excess return
                net_excess_return=(pl.col("excess_return") - pl.col("cumulative_cost")),
            ).with_columns(
                # Excess return drawdown
                excess_return_drawdown=(pl.col("excess_return") - pl.col("excess_return").cum_max()),
                # Net excess return drawdown
                net_excess_return_drawdown=(pl.col("net_excess_return") - pl.col("net_excess_return").cum_max())
            )
        )

        # Draw chart
        fig: go.Figure = make_subplots(
            rows=5,
            cols=1,
            subplot_titles=["Return", "Alpha", "Turnover", "Alpha Drawdown", "Alpha Drawdown with Cost"],
            vertical_spacing=0.06
        )

        strategy_curve: go.Scatter = go.Scatter(
            x=performance_df["date"],
            y=performance_df["cumulative_return"],
            mode="lines",
            name="Strategy"
        )
        net_strategy_curve: go.Scatter = go.Scatter(
            x=performance_df["date"],
            y=performance_df["cumulative_return"] - performance_df["cumulative_cost"],
            mode="lines",
            name="Strategy with Cost"
        )
        benchmark_curve: go.Scatter = go.Scatter(
            x=performance_df["date"],
            y=performance_df["benchmark_return"],
            mode="lines",
            name="Benchmark"
        )
        excess_curve: go.Scatter = go.Scatter(
            x=performance_df["date"],
            y=performance_df["excess_return"],
            mode="lines",
            name="Alpha"
        )
        net_excess_curve: go.Scatter = go.Scatter(
            x=performance_df["date"],
            y=performance_df["net_excess_return"],
            mode="lines",
            name="Alpha with Cost"
        )
        turnover_curve: go.Scatter = go.Scatter(
            x=self.daily_df["date"],
            y=self.daily_df["turnover"] / self.daily_df["balance"].shift(1),
            name="Turnover",
        )
        excess_drawdown_curve: go.Scatter = go.Scatter(
            x=performance_df["date"],
            y=performance_df["excess_return_drawdown"],
            fill='tozeroy',
            mode="lines",
            name="Alpha Drawdown"
        )
        net_excess_drawdown_curve: go.Scatter = go.Scatter(
            x=performance_df["date"],
            y=performance_df["net_excess_return_drawdown"],
            fill='tozeroy',
            mode="lines",
            name="Alpha Drawdown with Cost"
        )

        fig.add_trace(strategy_curve, row=1, col=1)
        fig.add_trace(net_strategy_curve, row=1, col=1)
        fig.add_trace(benchmark_curve, row=1, col=1)
        fig.add_trace(excess_curve, row=2, col=1)
        fig.add_trace(net_excess_curve, row=2, col=1)
        fig.add_trace(turnover_curve, row=3, col=1)
        fig.add_trace(excess_drawdown_curve, row=4, col=1)
        fig.add_trace(net_excess_drawdown_curve, row=5, col=1)

        fig.update_layout(
            height=1500,
            width=1200,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
            xaxis2=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
            xaxis3=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
            xaxis4=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
            xaxis5=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
            yaxis2=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
            yaxis3=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
            yaxis4=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
            yaxis5=dict(showgrid=True, gridwidth=1, gridcolor='LightGray')
        )
        fig.show()

    def update_daily_close(self, bars: dict[str, BarData], dt: datetime) -> None:
        """Update daily closing price"""
        d: date = dt.date()

        close_prices: dict[str, float] = {}
        for bar in bars.values():
            if not bar.close_price:
                close_prices[bar.vt_symbol] = self.pre_closes[bar.vt_symbol]
            else:
                close_prices[bar.vt_symbol] = bar.close_price

        daily_result: PortfolioDailyResult | None = self.daily_results.get(d, None)

        if daily_result:
            daily_result.update_close_prices(close_prices)
        else:
            self.daily_results[d] = PortfolioDailyResult(d, close_prices)

    def new_bars(self, dt: datetime) -> None:
        """Push historical data"""
        self.datetime = dt

        bars: dict[str, BarData] = {}
        for vt_symbol in self.vt_symbols:
            last_bar = self.bars.get(vt_symbol, None)
            if last_bar:
                if last_bar.close_price:
                    self.pre_closes[vt_symbol] = last_bar.close_price

            bar: BarData | None = self.history_data.get((dt, vt_symbol), None)

            # Check if historical data for the specified time of the contract is obtained
            if bar:
                # Update K-line for order matching
                self.bars[vt_symbol] = bar
                # Cache K-line data for strategy.on_bars update
                bars[vt_symbol] = bar
            # If not available, but there is contract data cached in the self.bars dictionary, use previous data to fill
            elif vt_symbol in self.bars:
                old_bar: BarData = self.bars[vt_symbol]

                fill_bar: BarData = BarData(
                    symbol=old_bar.symbol,
                    exchange=old_bar.exchange,
                    datetime=dt,
                    open_price=old_bar.close_price,
                    high_price=old_bar.close_price,
                    low_price=old_bar.close_price,
                    close_price=old_bar.close_price,
                    gateway_name=old_bar.gateway_name
                )
                self.bars[vt_symbol] = fill_bar

        self.cross_order()
        self.strategy.on_bars(bars)

        self.update_daily_close(self.bars, dt)

    def _uses_price_limits(self, vt_symbol: str) -> bool:
        """是否有涨跌停: A股 (SSE/SZSE) 有; 加密货币 24h 连续交易无"""
        _, exchange = extract_vt_symbol(vt_symbol)
        return exchange in A_SHARE_EXCHANGES

    def _limit_ratio(self, vt_symbol: str) -> float:
        """涨跌幅限制: 科创板 688xx 上市起 20%; 创业板 300/301xx 2020-08-24 起 20%
        (此前 10%); 主板 10%。ST (±5%) 无法从代码识别, 沪深300 不含 ST, 忽略。"""
        symbol: str = vt_symbol.split(".")[0]
        if symbol.startswith(("43", "83", "87", "920")):
            return 0.30  # 北交所 BSE
        if symbol.startswith("688"):
            return 0.20  # 科创板
        if symbol.startswith(("300", "301")):
            if self.datetime and self.datetime >= CYB_20PCT_DATE:
                return 0.20  # 创业板 (2020-08-24 起)
            return 0.10
        return 0.10

    def cross_order(self) -> None:
        """Match limit orders"""
        for order in list(self.active_limit_orders.values()):
            bar: BarData = self.bars[order.vt_symbol]

            # 停牌/无量日: 当日无成交不可成交 (fill_bar volume=0), 订单继续挂单
            if self.block_zero_volume and bar.volume <= 0:
                continue

            long_cross_price: float = bar.low_price
            short_cross_price: float = bar.high_price
            long_best_price: float = bar.open_price
            short_best_price: float = bar.open_price

            # Push order status update for unfilled orders
            if order.status == Status.SUBMITTING:
                order.status = Status.NOTTRADED
                self.strategy.update_order(order)

            # Calculate price limits (板块/日期感知, A股才启用)
            pricetick: float = self.priceticks[order.vt_symbol]
            pre_close: float = self.pre_closes.get(order.vt_symbol, 0)

            limit_up: float = 0
            limit_down: float = 0
            if self.use_price_limits and self._uses_price_limits(order.vt_symbol) and pre_close > 0:
                ratio: float = self._limit_ratio(order.vt_symbol)
                limit_up = round_to(pre_close * (1 + ratio), pricetick)
                limit_down = round_to(pre_close * (1 - ratio), pricetick)
                # Check limit orders that can be matched
                long_cross: bool = (
                    order.direction == Direction.LONG
                    and order.price >= long_cross_price
                    and long_cross_price > 0
                    and bar.low_price < limit_up        # Not a full-day limit-up market
                )
                short_cross: bool = (
                    order.direction == Direction.SHORT
                    and order.price <= short_cross_price
                    and short_cross_price > 0
                    and bar.high_price > limit_down     # Not a full-day limit-down market
                )
            else:
                # 无涨跌停市场 (加密货币等) 或首日无昨收: 常规撮合
                long_cross = (
                    order.direction == Direction.LONG
                    and order.price >= long_cross_price
                    and long_cross_price > 0
                )
                short_cross = (
                    order.direction == Direction.SHORT
                    and order.price <= short_cross_price
                    and short_cross_price > 0
                )

            if not long_cross and not short_cross:
                continue

            # Push order status update for filled orders
            order.traded = order.volume
            order.status = Status.ALLTRADED
            self.strategy.update_order(order)

            if order.vt_orderid in self.active_limit_orders:
                self.active_limit_orders.pop(order.vt_orderid)

            # Generate trade information
            self.trade_count += 1

            if long_cross:
                trade_price = min(order.price, long_best_price)
            else:
                trade_price = max(order.price, short_best_price)

            # 滑点: 买入价上浮 / 卖出价下浮 (模拟冲击成本), 归整到最小变动价位,
            # 且不越过当日涨跌停价 (现实约束)
            if self.slippage > 0:
                order_size: float = self.sizes[order.vt_symbol]
                if order.direction == Direction.LONG:
                    slipped_price: float = round_to(trade_price * (1 + self.slippage), pricetick)
                    if limit_up > 0:
                        slipped_price = min(slipped_price, limit_up)
                    self.slippage_cost += (slipped_price - trade_price) * order.volume * order_size
                else:
                    slipped_price = round_to(trade_price * (1 - self.slippage), pricetick)
                    if limit_down > 0:
                        slipped_price = max(slipped_price, limit_down)
                    self.slippage_cost += (trade_price - slipped_price) * order.volume * order_size
                trade_price = slipped_price

            trade: TradeData = TradeData(
                symbol=order.symbol,
                exchange=order.exchange,
                orderid=order.orderid,
                tradeid=str(self.trade_count),
                direction=order.direction,
                offset=order.offset,
                price=trade_price,
                volume=order.volume,
                datetime=self.datetime,
                gateway_name=self.gateway_name,
            )

            # Update available funds
            size: float = self.sizes[trade.vt_symbol]

            trade_turnover: float = trade.price * trade.volume * size

            # B3: 现金校验 — gap-down/超买防护 (方向性偏乐观修正):
            # 买入现金不足则缩量至可负担 (扣除保底佣金), 连最小单位都买不起则拒单 + 日志。
            # 策略侧 cash_ratio 缓冲大概率兜住, 此处兜底极端日的小额隐性杠杆。
            if trade.direction == Direction.LONG and self.cash < trade_turnover:
                unit_cost: float = trade.price * size
                affordable: int = int((self.cash - self.min_commission) / unit_cost) if unit_cost > 0 else 0
                if affordable < 1:
                    self.write_log(
                        f"现金不足拒单: {trade.vt_symbol} 需 {trade_turnover + self.min_commission:.2f} "
                        f"> 现金 {self.cash:.2f}"
                    )
                    order.status = Status.REJECTED
                    self.strategy.update_order(order)
                    continue
                if affordable < trade.volume:
                    self.write_log(
                        f"现金不足缩量: {trade.vt_symbol} {trade.volume}->{affordable} "
                        f"(现金 {self.cash:.2f}, 需 {trade_turnover:.2f})"
                    )
                    trade.volume = float(affordable)
                    order.traded = float(affordable)
                    trade_turnover = trade.price * trade.volume * size

            # 有效印花税: 2023-08-28 起减半为万5 (此前千1)
            stamp: float = _effective_stamp_rate(
                self.long_rates[trade.vt_symbol],
                self.short_rates[trade.vt_symbol],
                self.datetime,
                self.stamp_tax_halving,
            )

            if trade.direction == Direction.LONG:
                trade_commission: float = max(
                    trade_turnover * self.long_rates[trade.vt_symbol],
                    self.min_commission,
                )
            else:
                # 卖出 = 佣金(保底) + 印花税(不保底, 单边卖出征收), 与券商实际收费一致
                trade_commission = max(
                    trade_turnover * self.long_rates[trade.vt_symbol],
                    self.min_commission,
                ) + trade_turnover * stamp
                self.stamp_tax += trade_turnover * stamp

            if trade.direction == Direction.LONG:
                self.cash -= trade_turnover
            else:
                self.cash += trade_turnover

            self.cash -= trade_commission

            # Push trade information
            self.strategy.update_trade(trade)
            self.trades[trade.vt_tradeid] = trade

    def get_signal(self) -> pl.DataFrame:
        """Get model prediction signal for current time"""
        if not self.datetime:
            self.write_log("尚未开始数据回放，无法加载模型预测值")
            return pl.DataFrame()

        dt: datetime = self.datetime.replace(tzinfo=None)
        signal: pl.DataFrame = self.signal_df.filter(pl.col("datetime") == dt)

        if signal.is_empty():
            self.write_log(f"找不到{dt}对应的信号模型预测值")

        return signal

    def send_order(
        self,
        strategy: AlphaStrategy,
        vt_symbol: str,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
    ) -> list[str]:
        """Send order"""
        price = round_to(price, self.priceticks[vt_symbol])
        symbol, exchange = extract_vt_symbol(vt_symbol)

        self.limit_order_count += 1

        order: OrderData = OrderData(
            symbol=symbol,
            exchange=exchange,
            orderid=str(self.limit_order_count),
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            status=Status.SUBMITTING,
            datetime=self.datetime,
            gateway_name=self.gateway_name,
        )

        self.active_limit_orders[order.vt_orderid] = order
        self.limit_orders[order.vt_orderid] = order

        return [order.vt_orderid]

    def cancel_order(self, strategy: AlphaStrategy, vt_orderid: str) -> None:
        """Cancel order"""
        if vt_orderid not in self.active_limit_orders:
            return
        order: OrderData = self.active_limit_orders.pop(vt_orderid)

        order.status = Status.CANCELLED
        self.strategy.update_order(order)

    def write_log(self, msg: str, strategy: AlphaStrategy | None = None) -> None:
        """Output log message"""
        msg = f"{self.datetime}  {msg}"
        self.logs.append(msg)

    def get_all_trades(self) -> list[TradeData]:
        """Get all trade information"""
        return list(self.trades.values())

    def get_all_orders(self) -> list[OrderData]:
        """Get all order information"""
        return list(self.limit_orders.values())

    def get_all_daily_results(self) -> list["PortfolioDailyResult"]:
        """Get all daily profit and loss information"""
        return list(self.daily_results.values())

    def get_cash_available(self) -> float:
        """Get current available cash"""
        return self.cash

    def get_holding_value(self) -> float:
        """Get current holding market value"""
        holding_value: float = 0

        for vt_symbol, pos in self.strategy.pos_data.items():
            bar: BarData = self.bars[vt_symbol]
            size: float = self.sizes[vt_symbol]

            holding_value += bar.close_price * pos * size

        return holding_value


class ContractDailyResult:
    """Contract daily profit and loss result"""

    def __init__(self, result_date: date, close_price: float) -> None:
        """Constructor"""
        self.date: date = result_date
        self.close_price: float = close_price
        self.pre_close: float = 0

        self.trades: list[TradeData] = []
        self.trade_count: int = 0

        self.start_pos: float = 0
        self.end_pos: float = 0

        self.turnover: float = 0
        self.commission: float = 0

        self.trading_pnl: float = 0
        self.holding_pnl: float = 0
        self.total_pnl: float = 0
        self.net_pnl: float = 0

    def add_trade(self, trade: TradeData) -> None:
        """Add trade information"""
        self.trades.append(trade)

    def calculate_pnl(
        self,
        pre_close: float,
        start_pos: float,
        size: float,
        long_rate: float,
        short_rate: float,
        min_commission: float = 0,
        stamp_tax_halving: bool = True
    ) -> None:
        """Calculate profit and loss"""
        self.stamp_tax_halving = stamp_tax_halving

        # If there is no previous close price, use 1 instead to avoid division error
        if pre_close:
            self.pre_close = pre_close
        # else:
        #     self.pre_close = 1

        # Calculate holding profit and loss
        self.start_pos = start_pos
        self.end_pos = start_pos

        self.holding_pnl = self.start_pos * (self.close_price - self.pre_close) * size

        # Calculate trading profit and loss
        self.trade_count = len(self.trades)

        for trade in self.trades:
            if trade.direction == Direction.LONG:
                pos_change: float = trade.volume
                rate: float = long_rate
            else:
                pos_change = -trade.volume
                rate = long_rate

            self.end_pos += pos_change

            turnover: float = trade.volume * size * trade.price

            self.trading_pnl += pos_change * (self.close_price - trade.price) * size
            self.turnover += turnover
            # 佣金保底 (与 cross_order 现金口径一致); 卖出另加印花税 (2023-08-28 起万5)
            self.commission += max(turnover * rate, min_commission)
            if trade.direction == Direction.SHORT:
                self.commission += turnover * _effective_stamp_rate(
                    long_rate, short_rate, self.date, self.stamp_tax_halving)

        # Calculate daily profit and loss
        self.total_pnl = self.trading_pnl + self.holding_pnl
        self.net_pnl = self.total_pnl - self.commission

    def update_close_price(self, close_price: float) -> None:
        """Update daily close price"""
        self.close_price = close_price


class PortfolioDailyResult:
    """Portfolio daily profit and loss result"""

    def __init__(self, result_date: date, close_prices: dict[str, float]) -> None:
        """Constructor"""
        self.date: date = result_date
        self.close_prices: dict[str, float] = close_prices
        self.pre_closes: dict[str, float] = {}
        self.start_poses: dict[str, float] = {}
        self.end_poses: dict[str, float] = {}

        self.contract_results: dict[str, ContractDailyResult] = {}

        for vt_symbol, close_price in close_prices.items():
            self.contract_results[vt_symbol] = ContractDailyResult(result_date, close_price)

        self.trade_count: int = 0
        self.turnover: float = 0
        self.commission: float = 0
        self.trading_pnl: float = 0
        self.holding_pnl: float = 0
        self.total_pnl: float = 0
        self.net_pnl: float = 0

    def add_trade(self, trade: TradeData) -> None:
        """Add trade information"""
        contract_result: ContractDailyResult = self.contract_results[trade.vt_symbol]
        contract_result.add_trade(trade)

    def calculate_pnl(
        self,
        pre_closes: dict[str, float],
        start_poses: dict[str, float],
        sizes: dict[str, float],
        long_rates: dict[str, float],
        short_rates: dict[str, float],
        min_commission: float = 0,
        stamp_tax_halving: bool = True
    ) -> None:
        """Calculate profit and loss"""
        self.pre_closes = pre_closes
        self.start_poses = start_poses

        for vt_symbol, contract_result in self.contract_results.items():
            contract_result.calculate_pnl(
                pre_closes.get(vt_symbol, 0),
                start_poses.get(vt_symbol, 0),
                sizes[vt_symbol],
                long_rates[vt_symbol],
                short_rates[vt_symbol],
                min_commission,
                stamp_tax_halving
            )

            self.trade_count += contract_result.trade_count
            self.turnover += contract_result.turnover
            self.commission += contract_result.commission
            self.trading_pnl += contract_result.trading_pnl
            self.holding_pnl += contract_result.holding_pnl
            self.total_pnl += contract_result.total_pnl
            self.net_pnl += contract_result.net_pnl

            self.end_poses[vt_symbol] = contract_result.end_pos

    def update_close_prices(self, close_prices: dict[str, float]) -> None:
        """Update daily close prices"""
        self.close_prices.update(close_prices)

        for vt_symbol, close_price in close_prices.items():
            contract_result: ContractDailyResult | None = self.contract_results.get(vt_symbol, None)
            if contract_result:
                contract_result.update_close_price(close_price)
            else:
                self.contract_results[vt_symbol] = ContractDailyResult(self.date, close_price)
