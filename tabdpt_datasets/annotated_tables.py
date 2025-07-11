import hashlib
import os
import time
from pathlib import Path

import kaggle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from tabdpt_datasets.dataset import Dataset
from tabdpt_datasets.external.annotated_tables.ids_deduped import ANNOTATED_TABLE_IDS


class AnnotatedTablesDataset(Dataset):
    """
    Dataset of the tables used to evaluate TabPFN in https://arxiv.org/pdf/2406.16349, with
    duplicate datasets left out. Since Kaggle doesn't have a standard file layout, we just take the
    largest CSV file. Because of that, there's no target or test set, and this dataset should only
    be used for training.
    """

    @staticmethod
    def all_names():
        return [f"annotated_tables_{id}" for id in ANNOTATED_TABLE_IDS]

    @staticmethod
    def suite_name():
        return "annotated_tables"

    def __init__(self, name, task_id=None, split_seed=0):
        """
        Note that split_seed is only used for train/val split, since there's no test set.
        """
        if task_id is None:
            task_id = f"random-seed{split_seed}"
        else:
            assert task_id.startswith("random-seed")
            split_seed = int(task_id.removeprefix("random-seed"))
        super().__init__(name, task_id)
        self.kaggle_id = name[len("annotated_tables_") :]
        self.rng = np.random.default_rng(split_seed)
        self.metadata["kaggle_dataset_name"] = self.kaggle_id.split("/")[-1]

    def prepare_data(self, download_dir):
        dataset_dir = os.path.join(
            download_dir, "annotated-tables", self.kaggle_id.replace("/", "-")
        )
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            try:
                kaggle.api.dataset_download_files(self.kaggle_id, dataset_dir, unzip=True)
            except kaggle.rest.ApiException as e:
                if e.status == 429:  # Rate limit
                    time.sleep(0.5)
                    kaggle.api.dataset_download_files(self.kaggle_id, dataset_dir, unzip=True)
                elif e.status == 403:  # Rate limit
                    raise ValueError(
                        f"HTTP 403 when downloading Kaggle dataset {self.kaggle_id}. It was "
                        "probably deleted from Kaggle - try removing it from "
                        "datasets/external/annotated_tables/ids.py"
                    )
                else:
                    raise e
        csv_paths = sorted(Path(dataset_dir).rglob("*.csv"), key=lambda p: p.stat().st_size)
        if len(csv_paths) == 0:
            raise ValueError("Couldn not load data for " + self.name)

        hasher = hashlib.new("sha1")
        with open(csv_paths[-1], "rb") as f:
            hasher.update(f.read())
        self.metadata["file_sha1"] = hasher.hexdigest()
        X = pd.read_csv(csv_paths[-1], low_memory=False)

        categorical_inds = []
        to_drop = []
        n_dropped = 0
        for i, col in enumerate(X.columns):
            if X[col].dtype == "object" or isinstance(X[col], pd.CategoricalDtype):
                enc = OrdinalEncoder()
                X[[col]] = enc.fit_transform(X[[col]])
                # Drop high cardinality
                if len(enc.categories_[0]) > 100:
                    to_drop.append(col)
                    n_dropped += 1
                else:
                    categorical_inds.append(i - n_dropped)
        X.drop(columns=to_drop, inplace=True)
        self.metadata["categorical_feature_inds"] = categorical_inds
        self.X = X.to_numpy().astype(np.float32)

        n = len(X)
        perm = self.rng.permutation(n)
        self._train_inds = perm[: int(n * 0.85)]
        self._val_inds = perm[int(n * 0.85) :]

    def all_instances(self):
        return self.X, None

    def train_inds(self):
        return self._train_inds

    def val_inds(self):
        return self._val_inds

    def test_inds(self):
        return []
