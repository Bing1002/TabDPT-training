from __future__ import annotations

from abc import ABC, abstractmethod, abstractstaticmethod

import numpy as np
import scipy
import torch.utils.data


class Dataset(ABC):
    """
    Base class for dataset classes.

    A dataset class represents a suite of datasets and an instance of the class represents a single
    loaded dataset from the suite.

    To support evaluation over a range of methods, the current code assumes that the entire dataset
    can be loaded into memory and accessed as numpy arrays using all_instances, which also supports
    fast indexing. Synthetic datasets with a fixed set of instances can either generate their data at
    initialization or lazily.

    We will probably want to add separate methods for synthetic datasets that generate an indefinite
    number of instances on the fly, since these are less appropriate for evaluation and might not be
    usable for all baseline models anyway (e.g., decision trees that expect all instances to be
    available at once).

    To enable reproducible splits, datasets take an optional parameter task_id. Subclasses should
    handle splitting in prepare_data in a reproducible way given a task_id. They can also implement
    functions to generate splits that return a task_id, ensuring future reproducibility.
    """

    def __init__(self, name: str, task_id: str = None):
        """
        Params:
        name - the name of the dataset to load
        task_id - an optional ID that subclasses should use to ensure reproducibility
        """
        self.name = name
        self._task_id = task_id
        self.metadata = {"suite_name": self.suite_name(), "dataset_name": name}
        self.column_names = None

    @abstractstaticmethod
    def suite_name() -> str:
        """
        Name of the suite of datasets provided by this class, for cataloguing. Typically using all
        lowercase for consistency.
        """
        pass

    @abstractstaticmethod
    def all_names() -> list[str] | None:
        """
        Return the names of all datasets provided by this class, or None if it does not provide a
        fixed set of datasets
        """
        pass

    @abstractmethod
    def prepare_data(self, download_dir: str):
        """
        Download data if needed and do any CPU-side preprocessing, splitting, etc as needed.
        """
        pass

    def task_id(self) -> str | None:
        """
        Returns the task_id that can be used to reproduce the current task, or None if not possible.

        If task_id is specified as a constructor argument, this should always return the same value.
        Otherwise, subclasses can either choose to modify _task_id or override this method directly
        to return the task_id for the settings that the dataset was set up with.
        """
        return self._task_id

    def __len__(self) -> int:
        xs, _ = self.all_instances()
        return xs.shape[0]

    def __getitem__(self, i) -> tuple[np.ndarray, np.ndarray | int | float | None]:
        xs, ys = self.all_instances()
        if ys is not None:
            return xs[i], ys[i]
        return xs[i], None

    @abstractmethod
    def all_instances(self) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Return all instances as a feature matrix and target vector.
        """
        pass

    @abstractmethod
    def train_inds(self) -> list[int] | np.ndarray | range:
        pass

    @abstractmethod
    def val_inds(self) -> list[int] | np.ndarray | range:
        pass

    @abstractmethod
    def test_inds(self) -> list[int] | np.ndarray | range:
        pass

    def train_instances(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = self.all_instances()
        return X[self.train_inds()], y[self.train_inds()]

    def val_instances(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = self.all_instances()
        return X[self.val_inds()], y[self.val_inds()]

    def test_instances(self) -> tuple[np.ndarray, np.ndarray]:
        X, y = self.all_instances()
        return X[self.test_inds()], y[self.test_inds()]

    def auto_populate_metadata(self):
        X, y = self.all_instances()
        self.metadata["n_rows"] = X.shape[0]
        self.metadata["n_features"] = X.shape[1]
        self.metadata["n_cells"] = X.shape[0] * X.shape[1]
        if y is None:
            self.metadata["target_type"] = "none"
        else:
            self.metadata["y_mean"] = np.mean(y)
            self.metadata["y_var"] = np.var(y)
            if "target_type" not in self.metadata:
                self.metadata["target_type"] = "unknown"
            self.metadata["n_cells"] += y.shape[0]

        self.metadata["frac_missing"] = np.isnan(X).mean()
        self.metadata["frac_rows_with_missing"] = np.isnan(X).any(axis=1).mean()
        self.metadata["frac_features_with_missing"] = np.isnan(X).any(axis=0).mean()

        lin_coeffs = []
        for i in range(X.shape[1]):
            col = np.nan_to_num(X[:, i])
            if np.all(np.isclose(col, col[0])):
                lin_coeffs.append(None)
                continue
            try:
                res = np.linalg.lstsq(np.stack((col, np.ones_like(col)), axis=1), y, rcond=None)
                lin_coeffs.append(res[0].tolist())
            except np.linalg.LinAlgError:
                lin_coeffs.append(None)
                continue
        self.metadata["column_lin_coeffs"] = lin_coeffs

        self.metadata["column_means"] = X.mean(axis=0).tolist()
        self.metadata["column_vars"] = X.var(axis=0).tolist()
        self.metadata["column_skews"] = scipy.stats.skew(X, axis=0).tolist()
        self.metadata["column_kurtoses"] = scipy.stats.kurtosis(X, axis=0).tolist()


class TorchDataset(torch.utils.data.Dataset):
    """
    Utility class for accessing a Dataset split as a torch Dataset
    """

    def __init__(self, dataset: Dataset, split):
        assert split in ("train", "val", "test", "all")
        self.dataset = dataset
        if split == "all":
            self.inds = range(len(dataset))
        elif split == "train":
            self.inds = dataset.train_inds()
        elif split == "val":
            self.inds = dataset.val_inds()
        elif split == "test":
            self.inds = dataset.test_inds()

    def __len__(self):
        return len(self.inds)

    def __getitem__(self, i):
        return self.dataset[self.inds[i]]
