import os
from pathlib import Path

import numpy as np

import tabdpt_datasets.external.tabred.cooking_time as cooking_time
import tabdpt_datasets.external.tabred.delivery_eta as delivery_eta
import tabdpt_datasets.external.tabred.maps_routing as maps_routing
import tabdpt_datasets.external.tabred.weather as weather
from tabdpt_datasets.dataset import Dataset


class TabredDataset(Dataset):
    @staticmethod
    def all_names():
        return ["cooking_time", "delivery_eta", "maps_routing", "weather_forecasting"]

    @staticmethod
    def suite_name():
        return "tabred"

    def __init__(self, name, task_id=None):
        """
        task_id options correspond to the splits provided by the dataset:
        "split-default", "split-random-0", "split-random-1", "split-random-2",
        "split-sliding-window-0", "split-sliding-window-1", "split-sliding-window-2"
        """

        if task_id is None:
            task_id = "split-default"
        assert task_id in (
            "split-default",
            "split-random-0",
            "split-random-1",
            "split-random-2",
            "split-sliding-window-0",
            "split-sliding-window-1",
            "split-sliding-window-2",
        )
        super().__init__(name, task_id)
        self.metadata["target_type"] = "regression"

    def prepare_data(self, download_dir):
        download_dir = Path(download_dir)

        if self.name == "cooking_time":
            download_fn = cooking_time.main
            sub_dir = "cooking-time"
        elif self.name == "delivery_eta":
            download_fn = delivery_eta.main
            sub_dir = "delivery-eta"
        elif self.name == "maps_routing":
            download_fn = maps_routing.main
            sub_dir = "maps-routing"
        elif self.name == "weather_forecasting":
            download_fn = weather.main
            sub_dir = "weather"

        if not all(os.path.exists(download_dir / sub_dir / f) for f in ("X_num.npy", "Y.npy")):
            download_fn(download_dir)
        X_mats = []
        X_mats.append(np.load(download_dir / sub_dir / "X_num.npy"))
        if os.path.exists(download_dir / sub_dir / "X_bin.npy"):
            X_mats.append(np.load(download_dir / sub_dir / "X_bin.npy"))
        if os.path.exists(download_dir / sub_dir / "X_cat.npy"):
            X_cat = np.load(download_dir / sub_dir / "X_cat.npy")
            n_num = sum(X.shape[1] for X in X_mats)
            self.metadata["categorical_feature_inds"] = [n_num + i for i in range(X_cat.shape[1])]
            X_mats.append(X_cat)
        self.X = np.concatenate(X_mats, axis=1)
        self.y = np.load(download_dir / sub_dir / "Y.npy")
        self._train_inds = np.load(download_dir / sub_dir / self._task_id / "train_idx.npy")
        self._val_inds = np.load(download_dir / sub_dir / self._task_id / "val_idx.npy")
        self._test_inds = np.load(download_dir / sub_dir / self._task_id / "test_idx.npy")

    def all_instances(self):
        return self.X, self.y

    def train_inds(self):
        return self._train_inds

    def val_inds(self):
        return self._val_inds

    def test_inds(self):
        return self._test_inds
