from enum import Enum

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from transformer_layer import TransformerEncoderLayer


class Task(Enum):
    """Enum representing the type of task for the model.
    REG: Regression task.
    CLS: Classification task.
    """

    REG = 1
    CLS = 2


def pad_x(X: torch.Tensor, num_features: int) -> torch.Tensor:
    """Pad the input tensor X with zeros to match the specified number of features.
    Args:
        X (torch.Tensor): Input tensor of shape (seq_len, batch_size, n_features).
        num_features (int): Desired number of features after padding.
    Returns:
        torch.Tensor: Padded tensor of shape (seq_len, batch_size, num_features).
    """
    seq_len, batch_size, n_features = X.shape
    zero_feature_padding = torch.zeros(
        (seq_len, batch_size, num_features - n_features), device=X.device
    )
    return torch.cat([X, zero_feature_padding], -1)


def maskmean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute the mean of x along the specified dimension, ignoring masked values.
    Args:
        x (torch.Tensor): Input tensor of shape (time, batch, hidden dimension).
        mask (torch.Tensor): Mask tensor of the same shape as x, where True indicates valid values.
        dim (int): Dimension along which to compute the mean.
    Returns:
        torch.Tensor: Mean of x along the specified dimension, with masked values ignored.
    """
    x = torch.where(mask, x, 0)
    return x.sum(dim=dim, keepdim=True) / mask.sum(dim=dim, keepdim=True)


def maskstd(x: torch.Tensor, mask: torch.Tensor, dim: int = 0):
    """Compute the standard deviation of x along the specified dimension, ignoring masked values.
    Args:
        x (torch.Tensor): Input tensor of shape (time, batch, hidden dimension).
        mask (torch.Tensor): Mask tensor of the same shape as x, where True indicates valid values.
        dim (int): Dimension along which to compute the standard deviation.
    Returns:
        torch.Tensor: Standard deviation of x along the specified dimension, with masked values ignored.
    """
    num = mask.sum(dim=dim, keepdim=True)
    mean = maskmean(x, mask, dim=0)
    diffs = torch.where(mask, mean - x, 0)
    return ((diffs**2).sum(dim=0, keepdim=True) / (num - 1)) ** 0.5


def normalize_data(data: torch.Tensor, eval_pos: int) -> torch.Tensor:
    """Normalize the input data by subtracting the mean and dividing by the standard deviation.

    Args:
        data (torch.Tensor): input data of shape (time, batch, hidden dimension).
        eval_pos (int): Evaluation position used for slicing attention keys and values.

    Returns:
        torch.Tensor: normalized data of shape (time, batch, hidden dimension).
    """
    X = data[:eval_pos] if eval_pos > 0 else data
    mask = ~torch.isnan(X)
    mean = maskmean(X, mask, dim=0)
    std = maskstd(X, mask, dim=0) + 1e-6
    data = (data - mean) / std
    return data


def clip_outliers(data: torch.Tensor, eval_pos: int, n_sigma: int = 4):
    """
    clip outliers in data based on a given number of standard deviations.

    Args:
        data (torch.Tensor): Input data of shape (time, batch, hidden dimension).
        eval_pos (int): Evaluation position used for slicing attention keys and values.
        n_sigma (int): Number of standard deviations to use for clipping outliers.
    Returns:
        torch.Tensor: Data with outliers clipped, of shape (time, batch, hidden dimension)."""
    assert len(data.shape) == 3, "X must be T,B,H"
    X = data[:eval_pos] if eval_pos > 0 else data
    mask = ~torch.isnan(X)
    mean = maskmean(X, mask, dim=0)
    cutoff = n_sigma * maskstd(X, mask, dim=0)
    mask &= cutoff >= torch.abs(X - mean)
    cutoff = n_sigma * maskstd(X, mask, dim=0)
    return torch.clip(data, mean - cutoff, mean + cutoff)


def convert_to_torch_tensor(input: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert a NumPy array or a PyTorch tensor to a PyTorch tensor.
    Args:
        input (np.ndarray | torch.Tensor): Input data to be converted.
    Returns:
        torch.Tensor: Converted PyTorch tensor.
    Raises:
        TypeError: If the input is neither a NumPy array nor a PyTorch tensor.
    """
    if isinstance(input, np.ndarray):
        return torch.from_numpy(input)
    elif torch.is_tensor(input):
        return input
    else:
        raise TypeError("Input must be a NumPy array or a PyTorch tensor.")


class TabDPTModel(nn.Module):
    def __init__(
        self,
        dropout: float,
        n_out: int,
        nhead: int,
        nhid: int,
        ninp: int,
        nlayers: int,
        num_features: int,
    ):
        """TabDPTModel initialization.

        Args:
            dropout (float): Dropout rate.
            n_out (int): Number of output classes.
            nhead (int): Number of attention heads.
            nhid (int): Hidden dimension.
            ninp (int): Input dimension.
            nlayers (int): Number of transformer layers.
            num_features (int): Number of input features.
        """
        super().__init__()
        self.n_out = n_out  # number of output classes
        self.ninp = ninp  # embedding dimension
        self.transformer_encoder = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embed_dim=ninp,
                    num_heads=nhead,
                    ff_dim=nhid,
                )
                for _ in range(nlayers)
            ]
        )
        self.num_features = num_features
        self.encoder = nn.Linear(num_features, ninp)
        self.dropout = nn.Dropout(p=dropout)
        self.y_encoder = nn.Linear(1, ninp)
        self.head = nn.Sequential(nn.Linear(ninp, nhid), nn.GELU(), nn.Linear(nhid, n_out + 1))

    def forward(
        self,
        x_src: torch.Tensor,
        y_src: torch.Tensor,
        return_log_act_norms: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the TabDPTModel.
        Args:
            x_src (torch.Tensor): Input features of shape (time, batch, hidden dimension).
            y_src (torch.Tensor): Target values of shape (T, B).
            return_log_act_norms (bool): Whether to return activation norms for logging.
        Returns:
            torch.Tensor: Predicted values of shape (T, B, n_out + 1).
        """
        eval_pos = y_src.shape[0]

        # preproces features by normalizing and clipping outliers
        x_src = clip_outliers(x_src, -1 if self.training else eval_pos, n_sigma=4)
        x_src = normalize_data(x_src, -1 if self.training else eval_pos)
        x_src = clip_outliers(x_src, -1 if self.training else eval_pos, n_sigma=4)
        x_src = torch.nan_to_num(x_src, nan=0)

        # feature encoding
        x_src = self.encoder(x_src)
        mean = (x_src**2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean)
        x_src = x_src / rms

        # target encoding
        y_src = self.y_encoder(y_src.unsqueeze(-1))
        train_x = x_src[:eval_pos] + y_src
        src = torch.cat([train_x, x_src[eval_pos:]], 0)

        log_act_norms = {}
        log_act_norms["y"] = torch.norm(y_src, dim=-1).mean()

        # transformer layers
        for l, layer in enumerate(self.transformer_encoder):
            if l in [0, 1, 3, 6, 9]:
                log_act_norms[f"layer_{l}"] = torch.norm(src, dim=-1).mean()
            src = layer(src, eval_pos)

        # final head
        pred = self.head(src)

        if return_log_act_norms:
            return pred[eval_pos:], log_act_norms
        else:
            return pred[eval_pos:]

    @classmethod
    def load(cls, model_state: dict, config: DictConfig) -> nn.Module:
        """Load a pre-trained TabDPTModel from a state dictionary.

        Args:
            cls: TODO
            model_state (dict): state dictionary containing the model parameters.
            config (DictConfig): configuration object containing model parameters.

        Returns:
            nn.Module: model instance with loaded parameters.
        """
        # TODO loading model inside its own class without self?
        assert config.model.max_num_classes > 2
        model = TabDPTModel(
            dropout=config.training.dropout,
            n_out=config.model.max_num_classes,
            nhead=config.model.nhead,
            nhid=config.model.emsize * config.model.nhid_factor,
            ninp=config.model.emsize,
            nlayers=config.model.nlayers,
            num_features=config.model.max_num_features,
        )

        module_prefix = "_orig_mod."
        model_state = {k.replace(module_prefix, ""): v for k, v in model_state.items()}
        model.load_state_dict(model_state)
        model.to(config.env.device)
        model.eval()
        return model
