import os
import warnings
from typing import Optional

import numpy as np
import openml
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from tabdpt_datasets.dataset import Dataset


class OpenMLDataset(Dataset):
    """
    Generic class for loading any OpenML dataset
    """

    @staticmethod
    def all_names():
        return None

    @staticmethod
    def suite_name():
        return "openml"

    def __init__(
        self,
        name: str,
        task_id: Optional[str] = None,
        openml_dataset_id: Optional[str | int] = None,
        openml_task_id: Optional[str | int] = None,
    ):
        """Initializes the OpenMLDataset.

        name (str): The name of the dataset.
        task_id (Optional[str], optional): Specifies the type of data split to use. Supported formats:
            - "default": Uses the default split provided by the OpenML dataset if available, otherwise
              defaults to a random split with seed 0.
            - "fold<FOLD>": Uses the fold with the given index from the OpenML task definition. Only
              supported with OpenML task IDs.
            - "random-seed<SEED>": Creates a random 70/15/15 split using the specified integer seed.
        openml_dataset_id (Optional[str | int], optional): The OpenML dataset ID. Must be specified if
            `openml_task_id` is not provided. Defaults to None.
        openml_task_id (Optional[str | int], optional): The OpenML task ID. Must be specified if
            `openml_dataset_id` is not provided. Defaults to None.

        ValueError: Raised if neither or both `openml_dataset_id` and `openml_task_id` are provided.
        ValueError: Raised if `task_id` starts with "fold" but `openml_task_id` is not provided.
        ValueError: Raised if `task_id` has an invalid format.
        """
        super().__init__(name, task_id)

        # Initialize split indices to None
        self._train_inds = None
        self._val_inds = None
        self._test_inds = None

        if (openml_dataset_id is None) == (openml_task_id is None):
            raise ValueError("Must specify exactly one of openml_dataset_id or openml_task_id")
        self.did = openml_dataset_id
        self.tid = openml_task_id

        if task_id is None or task_id == "default":
            self.rng = np.random.default_rng(0)
            self.fold = 0
        elif task_id.startswith("fold"):
            if openml_task_id is None:
                raise ValueError("Can only use fold tasks with openml_task_id")
            self.fold = int(task_id.removeprefix("fold"))
        elif task_id.startswith("random-seed"):
            split_seed = int(task_id.removeprefix("random-seed"))
            self.rng = np.random.default_rng(split_seed)
            self.fold = None
        else:
            raise ValueError(f"Invalid task_id {task_id}")

    @property
    def openml_dataset(self):
        if not hasattr(self, "_openml_dataset"):
            raise ValueError("Data not loaded yet")
        return self._openml_dataset

    def prepare_data(self, download_dir: str):
        """
        Downloads the OpenML dataset and prepares the data for use.
        Args:
            download_dir (str): Directory to download the OpenML dataset to.
        """
        openml.config.set_root_cache_directory(os.path.join(download_dir, "openml_cache"))

        if self.tid:
            # retreive task and dataset information
            task = openml.tasks.get_task(
                self.tid,
                download_splits=True,
                download_data=True,
                download_qualities=True,
                download_features_meta_data=True,
            )
            dataset = openml.datasets.get_dataset(
                task.dataset_id,
                download_data=True,
                download_qualities=True,
                download_features_meta_data=True,
            )

            # retrieve data and metadata
            X, y, _, self.column_names = dataset.get_data(target=dataset.default_target_attribute)
            n = len(X)

            self.metadata["openml_task_id"] = self.tid
            self.metadata["openml_dataset_id"] = dataset.dataset_id

            #
            if self.fold is not None:
                split = task.get_train_test_split_indices(fold=self.fold)
                self._train_inds = split.train
                self._val_inds = []
                self._test_inds = split.test

        else:
            dataset = openml.datasets.get_dataset(
                self.did,
                download_data=True,
                download_qualities=True,
                download_features_meta_data=True,
            )
            X, y, _, self.column_names = dataset.get_data(target=dataset.default_target_attribute)
            self.metadata["openml_dataset_id"] = self.did

        self.metadata["openml_dataset_name"] = dataset.name
        self.metadata["openml_dataset_description"] = dataset.description

        if not self.tid or self.fold is None:
            n = len(X)
            perm = self.rng.permutation(n)
            self._train_inds = perm[: int(n * 0.7)]
            self._val_inds = perm[int(n * 0.7) : int(n * 0.85)]
            self._test_inds = perm[int(n * 0.85) :]

        if dataset.default_target_attribute and "," in dataset.default_target_attribute:
            y = None
            warnings.warn(
                f"Dataset {self.metadata['openml_dataset_id']} has multiple targets, which is "
                "not supported. Omitting targets."
            )

        self._openml_dataset = dataset

        categorical_inds = []
        for i, col in enumerate(X.columns):
            # Convert categorical features to ordinal integers
            if X[col].dtype == "object" or pd.api.types.is_categorical_dtype(X[col]):
                enc = OrdinalEncoder()
                X[[col]] = enc.fit_transform(X[[col]])
                categorical_inds.append(i)

        self.metadata["categorical_feature_inds"] = categorical_inds

        self.X = X.to_numpy().astype(np.float32)

        if y is None:
            self.y = None
            self.metadata["target_type"] = "none"
            return

        target_feature = [
            f for f in dataset.features.values() if f.name == dataset.default_target_attribute
        ][0]

        # encode target variable if it is categorical
        if (
            target_feature.data_type == "nominal"
            or y.dtype == "object"
            or pd.api.types.is_categorical_dtype(y)
        ):
            enc = LabelEncoder()
            self.y = enc.fit_transform(y)
            self.metadata["target_type"] = "classification"
        else:
            self.y = y.to_numpy().astype(np.float32)
            self.metadata["target_type"] = "regression"

    def all_instances(self):
        return self.X, self.y

    def train_inds(self):
        return self._train_inds

    def val_inds(self):
        return self._val_inds

    def test_inds(self):
        return self._test_inds


class OpenMLTaskDataset(Dataset):
    """
    Abstract class for dataset classes that are fully defined by a list of OpenML task IDs, which
    are the names in all_names()
    """

    def __init__(self, name, task_id=None):
        self.tid = int(name)
        self.openml_dataset = OpenMLDataset(name, task_id, openml_task_id=self.tid)
        task_id = self.openml_dataset.task_id()
        super().__init__(name, task_id)
        assert name in self.all_names()

    def prepare_data(self, download_dir):
        self.openml_dataset.prepare_data(download_dir)
        self.metadata = {**self.openml_dataset.metadata, **self.metadata}

    def all_instances(self):
        return self.openml_dataset.all_instances()

    def train_inds(self):
        return self.openml_dataset.train_inds()

    def val_inds(self):
        return self.openml_dataset.val_inds()

    def test_inds(self):
        return self.openml_dataset.test_inds()


# Note: openml task ids, not dataset ids
# See https://github.com/LeoGrin/tabular-benchmark?tab=readme-ov-file#downloading-the-datasets
# for source suite ids.
GRIN_NUM_CLS_IDS = [
    361055,
    361060,
    361061,
    361062,
    361063,
    361065,
    361066,
    361068,
    361069,
    361070,
    361273,
    361274,
    361275,
    361276,
    361277,
    361278,
]
GRIN_CAT_CLS_IDS = [361110, 361111, 361113, 361282, 361283, 361285, 361286]
GRIN_NUM_REG_IDS = [
    361072,
    361073,
    361074,
    361076,
    361077,
    361078,
    361079,
    361080,
    361081,
    361082,
    361083,
    361084,
    361085,
    361086,
    361087,
    361088,
    361279,
    361280,
    361281,
]
GRIN_CAT_REG_IDS = [
    361093,
    361094,
    361096,
    361097,
    361098,
    361099,
    361101,
    361102,
    361103,
    361104,
    361287,
    361288,
    361289,
    361291,
    361292,
    361293,
    361294,
]

GRIN_ALL_IDS = GRIN_NUM_CLS_IDS + GRIN_CAT_CLS_IDS + GRIN_NUM_REG_IDS + GRIN_CAT_REG_IDS


class GrinsztajnDataset(OpenMLTaskDataset):
    @staticmethod
    def all_names():
        return [str(tid) for tid in GRIN_ALL_IDS]

    @staticmethod
    def suite_name():
        return "grinsztajn"


CC18_IDS = [
    3,
    6,
    11,
    12,
    14,
    15,
    16,
    18,
    22,
    23,
    28,
    29,
    31,
    32,
    37,
    43,
    45,
    49,
    53,
    219,
    2074,
    2079,
    3021,
    3022,
    3481,
    3549,
    3560,
    3573,
    3902,
    3903,
    3904,
    3913,
    3917,
    3918,
    7592,
    9910,
    9946,
    9952,
    9957,
    9960,
    9964,
    9971,
    9976,
    9977,
    9978,
    9981,
    9985,
    10093,
    10101,
    14952,
    14954,
    14965,
    14969,
    14970,
    125920,
    125922,
    146195,
    146800,
    146817,
    146819,
    146820,
    146821,
    146822,
    146824,
    146825,
    167119,
    167120,
    167121,
    167124,
    167125,
    167140,
    167141,
]


class CC18Dataset(OpenMLTaskDataset):
    @staticmethod
    def all_names():
        return [str(tid) for tid in CC18_IDS]

    @staticmethod
    def suite_name():
        return "cc18"


CTR23_IDS = [
    361234,
    361235,
    361236,
    361237,
    361241,
    361242,
    361243,
    361244,
    361247,
    361249,
    361250,
    361251,
    361252,
    361253,
    361254,
    361255,
    361256,
    361257,
    361258,
    361259,
    361260,
    361261,
    361264,
    361266,
    361267,
    361268,
    361269,
    361272,
    361616,
    361617,
    361618,
    361619,
    361621,
    361622,
    361623,
]


class CTR23Dataset(OpenMLTaskDataset):
    @staticmethod
    def all_names():
        return [str(tid) for tid in CTR23_IDS]

    @staticmethod
    def suite_name():
        return "ctr23"


AMLB_IDS = [
    3,
    12,
    31,
    53,
    3917,
    3945,
    7592,
    7593,
    9952,
    9977,
    9981,
    10101,
    14965,
    34539,
    146195,
    146212,
    146606,
    146818,
    146821,
    146822,
    146825,
    167119,
    167120,
    168329,
    168330,
    168331,
    168332,
    168335,
    168337,
    168338,
    168868,
    168908,
    168909,
    168910,
    168911,
    168912,
    189354,
    189355,
    189356,
]


class AMLBDataset(OpenMLTaskDataset):
    @staticmethod
    def all_names():
        return [str(tid) for tid in AMLB_IDS]

    @staticmethod
    def suite_name():
        return "amlb"


ICLR_TRAINING_IDS = [
    41138,
    4135,
    4535,
    41434,
    375,
    1120,
    41150,
    40900,
    40536,
    1043,
    1119,
    1169,
    41147,
    1459,
    1466,
    1118,
    41142,
    23380,
    1596,
    41163,
    1471,
    846,
    1044,
    41164,
    1477,
    1476,
    1038,
    41159,
    23512,
    1479,
    821,
    41168,
    41143,
    184,
    1483,
    40679,
    24,
    1116,
    1568,
    1493,
    30,
    41145,
    1567,
    871,
    41161,
    41165,
    312,
    40685,
    1036,
    41146,
    41166,
    1509,
    40733,
    44089,
    44122,
    45022,
    45020,
    45028,
    45026,
    45038,
    45039,
    1111,
    1457,
    41167,
    41158,
    41144,
    41156,
    40498,
    41169,
    41162,
    42734,
    42732,
    42746,
    42742,
    43072,
    137,
    273,
    382,
    389,
    396,
    802,
    816,
    843,
    930,
    966,
    981,
    1002,
    1018,
    1037,
    1042,
    1112,
    1130,
    1142,
    1444,
    1453,
    1481,
    1503,
    1507,
    40646,
    40680,
    40706,
    44055,
    44056,
    44061,
    44063,
    44065,
    44068,
    44069,
    45041,
    45043,
    45045,
    45046,
    45047,
    44136,
    44137,
    44145,
    45032,
    4549,
    42572,
    42705,
    42728,
    41540,
    42724,
    42727,
    42730,
    41980,
    42563,
    3050,
    3277,
    43071,
]


class ICLRTrainingDataset(Dataset):
    @staticmethod
    def all_names():
        return [str(did) for did in ICLR_TRAINING_IDS]

    @staticmethod
    def suite_name():
        return "iclr-training"

    def __init__(self, name):
        super().__init__(name)
        assert name in self.all_names()
        self.did = int(name)
        self.openml_dataset = OpenMLDataset(name, openml_dataset_id=self.did)

    def prepare_data(self, download_dir):
        self.openml_dataset.prepare_data(download_dir)
        self.metadata = {**self.openml_dataset.metadata, **self.metadata}

    def all_instances(self):
        return self.openml_dataset.all_instances()

    def train_inds(self):
        return self.openml_dataset.train_inds()

    def val_inds(self):
        return self.openml_dataset.val_inds()

    def test_inds(self):
        return self.openml_dataset.test_inds()
