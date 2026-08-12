"""GBDT 排序学习 (LambdaMART / rank:ndcg) 数据准备工具。

选股任务中每个交易日 = 一个 query group:
    - label: 将横截面 z-score 的未来收益按当日分 5 档 (0-4) 整数等级,
      LightGBM lambdarank / XGBoost rank:ndcg 都要求非负 label
    - group: 每个交易日的股票数

前置条件: 传入的 df 必须已按 [datetime, vt_symbol] 升序排序
(rank_labels 与 rank_groups 的组顺序一致性依赖此排序, 模型类已满足)
"""
from __future__ import annotations

import numpy as np
import polars as pl

N_BINS = 5


def rank_labels(df: pl.DataFrame) -> np.ndarray:
    """每日横截面 5 档分箱 (0-4), 与 group 语义一致, 返回 int32 数组。"""
    cnt = pl.col("label").count().over("datetime")
    # 组内 ordinal rank (1..n) -> 均匀映射到 0..4; n=1 时保守给 0
    binned = (
        pl.when(cnt > 1)
        .then(((pl.col("label").rank("ordinal").over("datetime") - 1)
               / (cnt - 1) * (N_BINS - 1)).floor())
        .otherwise(0)
    )
    return np.asarray(
        df.with_columns(binned.alias("_rank_bin")).select("_rank_bin").to_series(),
        dtype=np.int32,
    )


def rank_groups(df: pl.DataFrame) -> np.ndarray:
    """每个交易日的股票数 (query group sizes), 按 datetime 升序。"""
    return np.asarray(
        df.group_by("datetime").len().sort("datetime")["len"],
        dtype=np.int32,
    )
