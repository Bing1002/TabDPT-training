import datetime
import os
import random
import sys
from typing import Optional

import faiss
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.tensorboard import SummaryWriter


def compute_losses(output, task, y, config):
    """
    Compute the averaged classification and regression losses.

    Parameters:
    - output: Tensor of shape [L, B, C+1], the outputs of the network.
    - task: Tensor of shape [B, 1, 1], containing 0 for classification or 1 for regression.
    - y: Tensor of shape [L, B, 1], the target values.

    Returns:
    - class_loss: Averaged classification loss (CrossEntropyLoss) or None if no classification task.
    - reg_loss: Averaged regression loss (MSELoss) or None if no regression task.
    """

    # reshape the output and y tensors to [B, L, C+1] and [B, L, 1] respectively
    output = output.transpose(0, 1)
    y = y.transpose(0, 1)

    # Flatten the task tensor to [B]
    task = task.view(-1)
    classification_mask = task == 0
    regression_mask = task == 1

    class_loss = torch.zeros(1, device=output.device)
    reg_loss = torch.zeros(1, device=output.device)

    # Classification Loss
    if classification_mask.any():
        # Select classification batches
        outputs_class = output[classification_mask]  # Shape: [num_class_batches, L, C+1]
        y_class = y[classification_mask]  # Shape: [num_class_batches, L, 1]

        # Reshape for CrossEntropyLoss: [N, C+1], targets [N]
        outputs_class = outputs_class.view(-1, outputs_class.size(-1))  # [N, C+1]
        y_class = y_class.view(-1).long().squeeze()  # [N]

        # Compute CrossEntropyLoss
        ce_loss_fn = nn.CrossEntropyLoss(
            reduction="mean", label_smoothing=config.training.label_smoothing
        )
        class_loss = ce_loss_fn(outputs_class, y_class)

    # Regression Loss
    if regression_mask.any():
        # Select regression batches
        outputs_reg = output[regression_mask, :, -1]  # Shape: [num_reg_batches, L]
        y_reg = y[regression_mask].squeeze(-1)  # Shape: [num_reg_batches, L]

        # Compute MSELoss
        mse_loss_fn = nn.MSELoss(reduction="mean")
        reg_loss = mse_loss_fn(outputs_reg, y_reg)

    return class_loss, reg_loss


def get_combined_loss(loss_cls, loss_reg, task, config):
    """
    Combine classification and regression losses based on the task ratio.
    Parameters:
    - loss_cls: Tensor, classification loss.
    - loss_reg: Tensor, regression loss.
    - task: Tensor, indicating the task type (0 for classification, 1 for regression).
    - config: Configuration object containing training parameters.
    Returns:
    - loss: Tensor, combined loss.
    """
    reg_ratio = task.float().mean()
    loss_cls = loss_cls / config.training.num_agg
    loss_reg = loss_reg / config.training.num_agg
    loss = loss_cls * (1 - reg_ratio) + loss_reg * reg_ratio
    return loss


class FAISS:
    """
    This class initializes a FAISS index with the provided data and allows for efficient nearest neighbor search.
    """

    def __init__(
        self, X: np.ndarray, use_hnsw: bool = False, hnsw_m: int = 32, metric: str = "L2"
    ) -> None:
        """
        Initializes the FAISS index with the provided data.
        Args:
            X (np.ndarray): The data to index, shape should be (n_samples, n_features).
            use_hnsw (bool): Whether to use HNSW index or not.
            hnsw_m (int): The number of bi-directional links created for each element in the HNSW index.
            metric (str): The distance metric to use, either "L2" for Euclidean distance or "IP" for inner product.
        """
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        X = np.ascontiguousarray(X)
        X = X.astype(np.float32)
        if use_hnsw:
            self.index = faiss.IndexHNSWFlat(X.shape[1], hnsw_m)
            if metric == "L2":
                self.index.metric_type = faiss.METRIC_L2
            elif metric == "IP":
                self.index.metric_type = faiss.METRIC_INNER_PRODUCT
            else:
                raise NotImplementedError
        else:
            if metric == "L2":
                self.index = faiss.IndexFlatL2(X.shape[1])
            elif metric == "IP":
                self.index = faiss.IndexFlatIP(X.shape[1])
            else:
                raise NotImplementedError
        self.index.add(X)

    def get_knn_indices(self, queries: np.ndarray | torch.Tensor, k: int) -> np.ndarray:
        """retreive the k-nearest neighbors indices for the given queries.

        Args:
            queries (np.ndarray|torch.Tensor): query points for which to find nearest neighbors.
            k (int): number of nearest neighbors to retrieve.

        Returns:
            np.ndarray: k nearest neighbors indices for each query point.
        """
        if isinstance(queries, torch.Tensor):
            queries = queries.cpu().numpy()
        queries = np.ascontiguousarray(queries)
        assert isinstance(k, int)

        knns = self.index.search(queries, k)
        indices_Xs = knns[1]
        return indices_Xs


def seed_everything(seed: int):
    """Set the random seed for reproducibility.

    Args:
        seed (int): seed value to set for random number generators.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def cleanup():
    """Cleanup the distributed training environment."""
    dist.destroy_process_group()


def signal_handler(*_):
    """Handle Ctrl+C signal to gracefully exit the program."""
    print("Received Ctrl+C, exiting...")
    cleanup()
    sys.exit(0)


def get_module(model: nn.Module) -> nn.Module:
    """Get the underlying module from a DistributedDataParallel model.
    Args:
        model (torch.nn.Module): The model, possibly wrapped in DistributedDataParallel.
    Returns:
        torch.nn.Module: The underlying module if model is DistributedDataParallel, else the model itself.
    """
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def log_param_norms(
    model: nn.Module, writer: SummaryWriter, step: int, task: str, global_step: int
):
    """Log the norms of model parameters to TensorBoard.

    Args:
        model (nn.Module): model to log parameter norms for.
        writer (SummaryWriter): TensorBoard writer to log the norms.
        step (int): current training step, used for logging.
        task (str): regression or classification task, used for logging.
        global_step (int): current global step, used for logging.
    """
    model = get_module(model)
    # Log encoder weight and bias norms (if bias exists)
    writer.add_scalar("norms/encoder_weight", torch.norm(model.encoder.weight).item(), step)
    if model.encoder.bias is not None:
        writer.add_scalar("norms/encoder_bias", torch.norm(model.encoder.bias).item(), step)

    # Log y_encoder weight and bias norms (if bias exists)
    writer.add_scalar("norms/y_encoder_weight", torch.norm(model.y_encoder.weight).item(), step)
    if model.y_encoder.bias is not None:
        writer.add_scalar("norms/y_encoder_bias", torch.norm(model.y_encoder.bias).item(), step)

    # Log transformer encoder total weights and biases (if biases exist)
    transformer_weights_norm = sum(
        torch.norm(layer.kv_proj.weight).item() for layer in model.transformer_encoder
    )

    writer.add_scalar("norms/transformer_weights", transformer_weights_norm, step)

    # Log classifier head weight and bias norms
    head_weights_norm = sum(torch.norm(param).item() for param in model.head.parameters())

    writer.add_scalar("norms/head_weights_biases", head_weights_norm, step)

    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]
        ),
        2,
    )
    total_norm1 = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), 1) for p in model.parameters() if p.grad is not None]
        ),
        1,
    )
    param_norm = torch.norm(torch.stack([torch.norm(p.detach(), 2) for p in model.parameters()]), 2)
    writer.add_scalar("Gradient/Norm/", total_norm, global_step=global_step)
    writer.add_scalar("Gradient/Norm_L1/", total_norm1, global_step=global_step)
    writer.add_scalar("Parameter/Norm/", param_norm, global_step=global_step)
    writer.add_scalar("Prior/reg_ratio", task.float().mean().item(), global_step=global_step)


def print_on_master_only(is_master: bool):
    """Override the built-in print function to only print on the master process.
    Args:
        is_master (bool): Whether the current process is the master process.
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_dist(device: str):
    """Initialize distributed training.

    Args:
        device (str): The device to use for training (e.g., "cuda:0" or "cpu").

    Returns:
        bool: Whether distributed training is being used.
        int: The rank of the current process.
        str: The device to use for training.
    """
    if "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])
        print("torch.distributed.launch and my rank is", rank)
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(seconds=20),
            world_size=torch.cuda.device_count(),
            rank=rank,
        )
        torch.distributed.barrier()
        print_on_master_only(rank == 0)
        return True, rank, f"cuda:{rank}"
    else:
        return False, 0, device


class DataPreprocessor:
    def __init__(self, impute_method: str = "mean", encode_y: bool = False):
        """Initialize the DataPreprocessor.
        Parameters:
        - impute_method: str, method for imputing missing values (default is "mean").
        - encode_y: bool, whether to encode the target variable y (default is False).
        """
        self.impute_method = impute_method
        self.encode_y = encode_y
        # Initialize the imputer and scaler
        self.imputer = SimpleImputer(strategy=impute_method)
        self.scaler = StandardScaler()

    def convert_cat2num(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[bool]]:
        """Convert categorical columns to numerical values.

        Parameters:
        - X: DataFrame, input features.
        - y: Series or None, target variable.

        Returns:
        - X: numpy array, features with categorical columns converted to numerical.
        - y: numpy array, target variable with categorical values converted to numerical.
        - cat_vals: list of bool, indicating which columns were categorical.
        """
        cat_vals = []
        # Convert categorical columns to numerical values using dataframe information
        for col in X.columns:
            if X[col].dtype == "object" or pd.api.types.is_categorical_dtype(X[col]):
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                cat_vals.append(True)
            else:
                cat_vals.append(False)
        X = X.to_numpy().astype(np.float32)

        # Convert categorical target to numerical values
        if y is not None:
            if y.dtype == "object" or pd.api.types.is_categorical_dtype(y):
                le = LabelEncoder()
                y = le.fit_transform(y)
                cat_vals.append(True)
            else:
                y = y.to_numpy().astype(np.float32)
                cat_vals.append(False)
        else:
            y = X[:, -1]
            X = np.delete(X, -1, axis=1)
        return X, y, cat_vals

    def fit_transform(self, X: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """
        Fit the imputer and scaler on the training data and transform it.

        Parameters:
            - X: numpy array, training features.

        Returns:
            - X: numpy array, transformed training features.
        """
        # Impute missing values
        X = self.imputer.fit_transform(X)

        # Scale the features
        X = self.scaler.fit_transform(X)

        return X

    def transform(self, X: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """
        Transform the test data using the fitted imputer and scaler.

        Parameters:
            - X: numpy array, testing features.

        Returns:
            - X: numpy array, transformed testing features.
        """
        # Impute missing values
        X = self.imputer.transform(X)

        # Scale the features
        X = self.scaler.transform(X)

        return X


def standardize(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardize the input tensor by subtracting the mean and dividing by the standard deviation.
    Args:
        tensor (torch.Tensor): Input tensor to standardize.
    Returns:
        torch.Tensor: Standardized tensor.
        torch.Tensor: Mean of the input tensor.
        torch.Tensor: Standard deviation of the input tensor.
    """
    y_means = tensor.mean(dim=0)
    y_stds = tensor.std(dim=0) + 1e-6
    return (tensor - y_means) / y_stds, y_means, y_stds
