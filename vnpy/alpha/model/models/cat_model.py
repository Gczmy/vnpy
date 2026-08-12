from typing import cast

import numpy as np
import polars as pl
from catboost import CatBoostRegressor, Pool

from vnpy.alpha.dataset import AlphaDataset, Segment
from vnpy.alpha.model import AlphaModel


class CatModel(AlphaModel):
    """CatBoost ensemble learning algorithm"""

    def __init__(
        self,
        learning_rate: float = 0.1,
        depth: int = 7,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
        seed: int | None = None
    ):
        """
        Parameters
        ----------
        learning_rate : float
            Learning rate
        depth : int
            Tree depth (与 LGB num_leaves / XGB max_depth 容量大致相当)
        num_boost_round : int
            Maximum number of training iterations
        early_stopping_rounds : int
            Number of rounds for early stopping
        seed : int | None
            Random seed
        """
        self.params: dict = {
            "loss_function": "RMSE",
            "learning_rate": learning_rate,
            "depth": depth,
            "random_seed": seed,
            "iterations": num_boost_round,
            "verbose": False,
        }

        self.num_boost_round: int = num_boost_round
        self.early_stopping_rounds: int = early_stopping_rounds

        self.model: CatBoostRegressor | None = None

    def _prepare_data(self, dataset: AlphaDataset) -> list[Pool]:
        """准备训练/验证数据 (与 LgbModel._prepare_data 同构)"""
        ds: list[Pool] = []

        for segment in [Segment.TRAIN, Segment.VALID]:
            df: pl.DataFrame = dataset.fetch_learn(segment)
            df = df.sort(["datetime", "vt_symbol"])

            data = df.select(df.columns[2: -1]).to_pandas()
            label = np.array(df["label"])

            ds.append(Pool(data, label))

        return ds

    def fit(self, dataset: AlphaDataset) -> None:
        """训练模型 (early stopping + use_best_model on valid set)"""
        ds: list[Pool] = self._prepare_data(dataset)

        self.model = CatBoostRegressor(**self.params)
        self.model.fit(
            ds[0],
            eval_set=ds[1],
            early_stopping_rounds=self.early_stopping_rounds,
            use_best_model=True,
        )

    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        """在给定 segment 上预测"""
        if self.model is None:
            raise ValueError("model is not fitted yet!")

        df: pl.DataFrame = dataset.fetch_infer(segment)
        df = df.sort(["datetime", "vt_symbol"])

        data = df.select(df.columns[2: -1]).to_pandas()
        result = cast(np.ndarray, self.model.predict(Pool(data)))
        return np.asarray(result).ravel()
