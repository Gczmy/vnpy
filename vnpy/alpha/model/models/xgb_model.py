from typing import cast

import numpy as np
import polars as pl
import xgboost as xgb

from vnpy.alpha.dataset import AlphaDataset, Segment
from vnpy.alpha.model import AlphaModel
from vnpy.alpha.model.rank_utils import rank_groups, rank_labels


class XgbModel(AlphaModel):
    """XGBoost ensemble learning algorithm"""

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_depth: int = 7,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
        log_evaluation_period: int = 1,
        seed: int | None = None,
        colsample_bytree: float = 1.0,
        booster: str = "gbtree",
        objective: str = "reg:squarederror"
    ):
        """
        Parameters
        ----------
        learning_rate : float
            Learning rate (eta)
        max_depth : int
            Maximum tree depth (与 LGB num_leaves=31 容量大致相当)
        num_boost_round : int
            Maximum number of training rounds
        early_stopping_rounds : int
            Number of rounds for early stopping
        log_evaluation_period : int
            Interval rounds for printing training logs
        seed : int | None
            Random seed. ⚠️ 仅当同时引入随机源 (如 colsample_bytree<1) 时才产生 seed 方差:
            XGBoost 默认 gbtree + 全特征确定性地贪心分裂, 仅设 seed 不改变结果
            (2026-08-12 实测: 三个 "seed" 信号 md5 完全一致, 见 extended_oos_2026.md §24⑤)。
            注意: 不能用 subsample 做随机源——按行采样会破坏 ranking 的 query group 结构。
        colsample_bytree : float
            (0, 1] 每棵树随机采样的特征比例。默认 1.0 (全特征, 确定性); 3-seed 协议
            建议 0.9: 特征采样不影响 query group, 是 ranking 任务唯一安全的随机源。
        booster : str
            "gbtree" (默认) 或 "dart" (Dropout GBDT, 继承 gbtree 全部参数, 每轮随机
            丢弃部分树缓解过拟合)。dart 下训练有随机性、早停不稳定 -> fit() 传
            early_stopping_rounds=None, 用固定轮数 + 事后 iteration_range 选优
            (见 docs/gbdt_dart_experiment_plan.md)。
        objective : str
            "reg:squarederror" (默认, 点式回归) 或 "rank:ndcg" (LambdaMART 排序学习,
            按交易日构造 query group, label 自动 5 档整数分箱, ndcg@10 早停)
        """
        self.params: dict = {
            "objective": objective,
            "eta": learning_rate,
            "max_depth": max_depth,
            "booster": booster,
            "seed": seed,
            "colsample_bytree": colsample_bytree,
        }
        if booster == "dart":
            # DART 推荐超参 (调研: XGBoost DART 教程参数建议, 见排期文档 §2.4)
            self.params.update({
                "rate_drop": 0.1,
                "skip_drop": 0.5,
                "sample_type": "uniform",
                "normalize_type": "tree",
            })
        if objective == "rank:ndcg":
            self.params["eval_metric"] = "ndcg@10"

        self.num_boost_round: int = num_boost_round
        self.early_stopping_rounds: int = early_stopping_rounds
        self.log_evaluation_period: int = log_evaluation_period

        self.model: xgb.Booster | None = None

    def _prepare_data(self, dataset: AlphaDataset) -> list[xgb.DMatrix]:
        """准备训练/验证数据 (与 LgbModel._prepare_data 同构)"""
        ds: list[xgb.DMatrix] = []

        for segment in [Segment.TRAIN, Segment.VALID]:
            df: pl.DataFrame = dataset.fetch_learn(segment)
            df = df.sort(["datetime", "vt_symbol"])

            data = df.select(df.columns[2: -1]).to_pandas()
            label = np.array(df["label"])

            if self.params["objective"] == "rank:ndcg":
                # 排序学习: 每日 = 一个 query group, label 转 5 档整数
                label = rank_labels(df)
                group = rank_groups(df)
                ds.append(xgb.DMatrix(data, label=label, group=group))
            else:
                ds.append(xgb.DMatrix(data, label=label))

        return ds

    def fit(self, dataset: AlphaDataset) -> None:
        """训练模型 (early stopping on valid set)"""
        ds: list[xgb.DMatrix] = self._prepare_data(dataset)

        self.model = xgb.train(
            self.params,
            ds[0],
            num_boost_round=self.num_boost_round,
            evals=[(ds[0], "train"), (ds[1], "valid")],
            # dart 下早停不稳定 (训练有随机性, 验证曲线震荡) -> 固定轮数训练
            early_stopping_rounds=None if self.params.get("booster") == "dart"
            else self.early_stopping_rounds,
            verbose_eval=self.log_evaluation_period,
        )

    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        """在给定 segment 上预测"""
        if self.model is None:
            raise ValueError("model is not fitted yet!")

        df: pl.DataFrame = dataset.fetch_infer(segment)
        df = df.sort(["datetime", "vt_symbol"])

        # 必须用 pandas DataFrame 构建 DMatrix, 保留特征名
        # (XGBoost 训练时从 DataFrame 记住了 feature_names, 预测时 numpy 会报错)
        data = df.select(df.columns[2: -1]).to_pandas()
        dtest = xgb.DMatrix(data)

        # 用早停最优迭代数 (best_iteration), 避免尾部过拟合轮次。
        # ⚠️ XGBoost 的 best_iteration 是 property: 未用早停时访问抛 AttributeError
        # (dart 模式下 early_stopping_rounds=None) -> getattr 捕获后走全模型预测。
        best_iter = getattr(self.model, "best_iteration", None)
        if best_iter:
            result = cast(
                np.ndarray,
                self.model.predict(dtest, iteration_range=(0, best_iter + 1)),
            )
        else:
            result = cast(np.ndarray, self.model.predict(dtest))
        return result
