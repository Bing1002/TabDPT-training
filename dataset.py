import random
from collections import defaultdict

import numpy as np
import openml
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from tqdm import tqdm

from model import pad_x
from utils import FAISS, DataPreprocessor


def collate_fn(batch):
    """Collate function for DataLoader.

    Args:
        batch (list): A list of samples from the dataset.

    Returns:
        tuple: A tuple containing the concatenated input tensors and the corresponding target tensors.
    """
    ret = [torch.cat(el, dim=1) for el in zip(*batch)]
    return ret


class FullDataset(Dataset):
    """FullDataset class for loading and processing datasets.
    This class inherits from `torch.utils.data.Dataset` and is used to load datasets from OpenML,
    preprocess them, and provide samples for training a model.
    It supports both regression and classification tasks, with the ability to use k-NN for context sampling.
    """

    def __init__(self, device: str, config: DictConfig):
        """FullDataset initialization.

        Args:
            device (str): The device to use for tensor operations.
            config (DictConfig): The configuration for the dataset.
        """
        self.steps_per_epoch = (
            config.training.num_agg * config.training.num_model_updates * config.training.num_epochs
        )
        self.batch_size = config.training.batch_size
        self.device = device
        self.context_length = getattr(config.training, "seq_len", 1024)
        self.retrieval = getattr(config.data, "retrieval", True)
        self.max_feat = config.model.max_num_features
        self.y_reg_augment = config.data.y_reg_augment

        # Load dataset IDs from CSV file
        # TODO: make dataset_ids configurable
        self.dataset_ids = (
            pd.read_csv("data_splits/noleak_training_datasets.csv")["did"].values.ravel().tolist()
        )
        self.datasets = defaultdict(dict)

        self.dataset_ids = self.dataset_ids[0:4]
        # Load datasets and preprocess them
        for did in tqdm(self.dataset_ids):
            # download the dataset from OpenML
            dataset = openml.datasets.get_dataset(
                did,
                download_data=False,
                download_qualities=False,
                download_features_meta_data=False,
            )

            # Get the data as a pandas DataFrame
            X, y, _, _ = dataset.get_data(
                dataset_format="dataframe", target=dataset.default_target_attribute
            )

            # TODO: preprocessing is a bit different from the usual .fit_transform() since concatenation happens in the middle
            preprocessor = DataPreprocessor()
            # Preprocess the data
            X, y, cat_vals = preprocessor.convert_cat2num(X, y)
            X = preprocessor.imputer.fit_transform(X)

            # Concatenate the target as the last column
            X = np.concatenate([X, y[:, None]], axis=1)
            X = preprocessor.scaler.fit_transform(X)

            if config.data.retrieval:
                faiss_knn = FAISS(X, metric="L2", use_hnsw=False)
            else:
                faiss_knn = None

            self.datasets[did] = {"X": X, "index": faiss_knn, "cat": cat_vals}

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.steps_per_epoch * self.batch_size

    def transform_target(
        self, y: np.ndarray, cls_threshold: float = 10, cls_prob: float = 0.3
    ) -> tuple[np.ndarray, str]:
        """Transform the target variable into classification or regression format.
        Args:
            y (np.ndarray): The target variable.
            cls_threshold (int): The threshold for classification.
            cls_prob (float): The probability of returning a classification task.
        Returns:
            tuple: A tuple containing the transformed target variable and the task type ("cls" or "reg").
        """
        unique_y = np.unique(y)
        if len(unique_y) > cls_threshold:
            if np.random.rand() > cls_prob:
                return y, "reg"
            else:
                num_class = np.random.randint(2, cls_threshold)
                cls_boundary = np.random.choice(
                    sorted(np.unique(y))[1:-1], num_class - 1, replace=False
                )
                y = (y[:, None] > cls_boundary[None, :]).sum(1)
                return y, "cls"
        elif len(unique_y) >= 2:
            le = LabelEncoder()
            y = le.fit_transform(y)
            y = y.astype(np.float32)
            return y, "cls"
        else:
            return y, "reg"

    def check_cls_sample(self, X: torch.Tensor, y: torch.Tensor) -> bool:
        """Check if the classification sample is valid.
        Args:
            X (torch.Tensor): The input features.
            y (torch.Tensor): The target labels.
        Returns:
            bool: True if the sample is valid, False otherwise.
        """
        assert y.dim() == 3 and y.shape[1] == 1 and y.shape[2] == 1
        assert X.dim() == 3 and X.shape[1] == 1
        return True

    def check_reg_sample(self, X: torch.Tensor, y: torch.Tensor) -> bool:
        """Check if the regression sample is valid.

        Args:
            X (torch.Tensor): The input features.
            y (torch.Tensor): The target labels.
        Returns:
            bool: True if the sample is valid, False otherwise.
        """
        assert y.dim() == 3 and y.shape[1] == 1 and y.shape[2] == 1
        assert X.dim() == 3 and X.shape[1] == 1
        if torch.max(y.abs().ravel()) > 10:
            return False
        if np.max(np.unique(y.ravel(), return_counts=True)[1]) > 0.95 * y.numel():
            return False
        return True

    @torch.no_grad()
    def __getitem__(self, _) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """get a sample from the dataset.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The input features, target values, and task type.
        """
        i = 0
        while i < 100:
            X, y, task = self.generate_sample()
            if task == 0:
                if self.check_cls_sample(X, y):
                    break
            else:
                if self.check_reg_sample(X, y):
                    break
            i += 1
        return X, y, task

    @torch.no_grad()
    def context_sampler(self, num_features: int, num_samples: int, X_sample: torch.Tensor):
        """Sample a context from the input tensor.

        Args:
            num_features (int): The number of features in the input tensor.
            num_samples (int): The number of samples in the input tensor.
            X_sample (torch.Tensor): The input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, str]: The context features, target values, and task type.
        """
        target_idx = random.randint(0, num_features - 1)

        # If context length > the number of samples, we need to sample with replacement
        if self.context_length > num_samples:
            random_selection_indices = np.random.choice(num_samples, num_samples, replace=False)
            random_selection_indices = np.concatenate(
                [
                    np.random.choice(num_samples, self.context_length - num_samples, replace=True),
                    random_selection_indices,
                ]
            )
        else:
            # If context length <= the number of samples, we can sample without replacement
            random_selection_indices = np.random.choice(
                num_samples, self.context_length, replace=False
            )
        # Select the samples and extract the target feature
        X_nni = X_sample[random_selection_indices]
        y_nni = X_nni[:, target_idx]

        # Remove the target feature from the feature set
        X_nni = np.delete(X_nni, target_idx, axis=1)

        # Transform the target feature into classification or regression format
        y_nni, task = self.transform_target(y_nni.ravel(), cls_threshold=10, cls_prob=0.5)

        # Reshape the tensors to match the expected dimensions
        X_nni = X_nni.reshape(-1, 1, X_nni.shape[-1])
        y_nni = y_nni.reshape(-1, 1, 1)

        # Select a random subset of features
        # Ensure that the number of features sampled is within the limits
        num_features_sampled = random.randint(
            min(self.max_feat // 2, X_nni.shape[-1] // 2), min(self.max_feat, X_nni.shape[-1])
        )
        random_feature_indices = np.random.choice(
            X_nni.shape[-1], num_features_sampled, replace=False
        )
        X_nni = X_nni[..., random_feature_indices]

        # Convert to PyTorch tensors
        X_nni = torch.Tensor(X_nni)
        y_nni = torch.Tensor(y_nni)

        return X_nni, y_nni, task

    @torch.no_grad()
    def use_knn(self, X_sample, index_sample, num_samples, num_features):
        """Use k-NN to find similar samples.

        Args:
            X_sample (np.ndarray): The input features.
            index_sample (FAISS): The FAISS index for k-NN search.
            num_samples (int): The number of samples.
            num_features (int): The number of features.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, str]: The context features, target values, and task type.
        """
        # Randomly select a query point
        x_q_idx = random.randint(0, num_samples - 1)
        x_q = X_sample[x_q_idx].reshape(1, -1).copy()
        x_q = x_q.astype(np.float32)

        # Randomly select a target feature index
        target_idx = random.randint(0, num_features - 1)

        # Get the indices of the k-nearest neighbors of the query point
        indices_X_nni = index_sample.get_knn_indices(x_q, self.context_length)
        X_nni = X_sample[torch.tensor(indices_X_nni)]
        X_nni = np.swapaxes(X_nni, 0, 1)

        # Extract the target feature and remove it from the feature set
        y_nni = X_nni[:, :, target_idx]
        X_nni = np.delete(X_nni, target_idx, axis=2)

        # Transform the target feature
        y_nni, task = self.transform_target(y_nni.ravel(), cls_threshold=10, cls_prob=0.5)

        # select a random subset of features
        num_features_sampled = random.randint(
            min(self.max_feat // 2, X_nni.shape[-1] // 2), min(self.max_feat, X_nni.shape[-1])
        )
        random_feature_indices = np.random.choice(
            X_nni.shape[-1], num_features_sampled, replace=False
        )
        X_nni = X_nni[..., random_feature_indices]

        # Convert to PyTorch tensors
        X_nni = torch.Tensor(X_nni)
        y_nni = torch.Tensor(y_nni)
        return X_nni, y_nni, task

    @torch.no_grad()
    def generate_sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a sample from the dataset.

        Raises:
            ValueError: If the generated sample is invalid.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The input features, target values, and task type.
        """
        # Randomly select a dataset ID
        did_sample = random.choices(self.dataset_ids, k=1)[0]

        # Get the dataset
        X_sample, index_sample, _ = self.datasets[did_sample].values()
        num_samples, num_features = X_sample.shape

        if self.retrieval:  # Randomly select a query point and its k-nearest neighbors as context
            X_nni, y_nni, task = self.use_knn(X_sample, index_sample, num_samples, num_features)
        else:  # Randomly select a sample and its context
            X_nni, y_nni, task = self.context_sampler(num_features, num_samples, X_sample)

        # Pad the features to the maximum number of features
        X_nni, y_nni = (
            pad_x(X_nni, self.max_feat),
            y_nni,
        )

        # Shuffle the rows
        shuffle_indices = np.random.choice(self.context_length, self.context_length, replace=False)
        X = X_nni[shuffle_indices]
        y = y_nni[shuffle_indices]

        if task == "cls":  # If classification task, shuffle the labels
            le = LabelEncoder()
            y_ = le.fit_transform(y.ravel())
            classes = list(le.classes_)
            random.shuffle(classes)
            mapping = {original: shuffled for original, shuffled in zip(le.classes_, classes)}
            y_random = np.vectorize(mapping.get)(y_)
            y = y_random
        elif task == "reg":  # If regression task, apply normalization and augmentation
            y_means = y.mean(dim=0)
            y_stds = y.std(dim=0) + 1e-6
            y = (y - y_means) / y_stds
            if self.y_reg_augment:
                # Apply random nonlinear transformations to the target for augmentation
                # TODO: shouldn't this non-linear transformation be implemented
                # only for a certain calls, not always, e.g. with:
                # if random.random() < 0.5:
                y = random_nonlinear_transform(y)
                y_means = y.mean(dim=0)
                y_stds = y.std(dim=0) + 1e-6
                y = (y - y_means) / y_stds
        else:
            raise ValueError("Task must be 'cls' or 'reg'")

        return (
            torch.Tensor(X),
            torch.Tensor(y).view(-1, 1, 1),
            torch.Tensor([0 if task == "cls" else 1]).unsqueeze(-1).unsqueeze(-1).long(),
        )


# A small bank of nonlinear functions (feel free to add/remove as appropriate)
FUNCTION_BANK = {
    "sin": torch.sin,
    "tanh": torch.tanh,
    "square": lambda x: x**2,
    "identity": lambda x: x,
    "step": lambda x: torch.where(x > 0, 1, 0),
    "relu": torch.nn.functional.relu,
    "sqrt": lambda x: torch.sign(x) * torch.sqrt(torch.abs(x)),
    "log": lambda x: torch.log(1 + torch.abs(x)) * torch.sign(x),
}


def random_nonlinear_transform(
    y: torch.Tensor,
    n_transforms: int = 2,
    scale_range: tuple = (0.5, 2.0),
    bias_range: tuple = (-1.0, 1.0),
) -> torch.Tensor:
    """Applies a series of random nonlinear transformations to the input tensor.

    Args:
        y (torch.Tensor): target tensor to be transformed.
        n_transforms (int, optional): number of transformations to apply. Defaults to 2.
        scale_range (tuple, optional): range of scaling factors to apply. Defaults to (0.5, 2.0).
        bias_range (tuple, optional): range of bias values to apply. Defaults to (-1.0, 1.0).

    Returns:
        torch.Tensor: Transformed tensor after applying the random nonlinear transformations.
    """
    device = y.device
    y_out = y.clone()
    func_keys = list(FUNCTION_BANK.keys())
    for _ in range(n_transforms):
        func_idx = torch.randint(low=0, high=len(func_keys), size=(1,)).item()
        func = FUNCTION_BANK[func_keys[func_idx]]
        scale = torch.empty(1, device=device).uniform_(*scale_range).item()
        bias = torch.empty(1, device=device).uniform_(*bias_range).item()
        if torch.rand(1, device=device) < 0.5:
            y_out = scale * func(y_out + bias)
        else:
            y_out = func(scale * y_out + bias)
    return y_out.float()
