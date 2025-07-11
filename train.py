import os
import random
import shutil
import signal

import hydra
import openml
import torch.distributed as dist
import schedulefree
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import FullDataset, collate_fn
from eval_full import FullEval
from model import TabDPTModel
from utils import (
    cleanup,
    compute_losses,
    get_combined_loss,
    init_dist,
    log_param_norms,
    seed_everything,
    signal_handler,
)


def save(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: DictConfig,
    stats: dict,
    path: str,
    name: str,
) -> None:
    """Save the model and optimizer state.

    Args:
        model (nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        config (DictConfig): The configuration for the experiment.
        stats (dict): The training statistics.
        path (str): The path to save the checkpoint.
        name (str): The name of the checkpoint file.

    Returns:
        None
    """
    ckpt = {
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "cfg": config,
        "stats": stats,
    }
    torch.save(ckpt, f"{path}/{name}.ckpt")


def save_eval_callback(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    vals: list,
    epoch: int,
    config: DictConfig,
    writer: SummaryWriter,
    rank: int,
    stats: dict,
) -> None:
    """Save evaluation checkpoints and log metrics.

    Args:
        model (nn.Module): The model being trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        vals (list): A list of validation datasets.
        epoch (int): The current epoch number.
        config (DictConfig): The configuration for the training process.
        writer (SummaryWriter): The TensorBoard writer for logging.
        rank (int): The rank of the current process.
        stats (dict): A dictionary to store training statistics.

    Returns:
        None
    """
    # save latest model
    stats["epoch_in_training"] = epoch
    if rank == 0:
        save(model, optimizer, config, stats, config.exp_path, "latest")

    # eval and save best
    model.eval()
    if hasattr(optimizer, "eval"):
        optimizer.eval()

    for val in vals:
        for name, metric in val.eval(model, context_length=config.training.eval_seq_len).items():
            if rank == 0:
                print(f"Epoch {epoch} | {name}: {metric}")
                # only save checkpoints using the validation metric
                if metric > stats.get(f"best_{name}", -float("inf")):
                    stats[f"best_{name}"] = metric
                    if name in config.logging.save_metrics:
                        save(model, optimizer, config, stats, config.exp_path, f"best_{name}")
                writer.add_scalar(f"val/{name}/", metric, epoch)

    # reset the model and optimizer to training mode
    model.train()
    if hasattr(optimizer, "train"):
        optimizer.train()


def set_experiment_from_config(config: DictConfig) -> tuple[nn.Module, torch.optim.Optimizer, dict]:
    """Set up the experiment based on the provided configuration.

    Args:
        config (DictConfig): The configuration for the experiment.

    Raises:
        Exception: If the reset policy is not recognized.

    Returns:
        Tuple[nn.Module, torch.optim.Optimizer, dict]: A tuple containing the model,
        optimizer, and a dictionary with training statistics.
    """

    load_state_from_saved = False
    stats = {"epoch_in_training": 0}

    seed_everything(config.seed)

    if hasattr(config.data, "single_dataset_id") and config.data.single_dataset_id is not None:
        # Load the dataset metadata (we assume that download_data is not necessary)
        dataset = openml.datasets.get_dataset(config.data.single_dataset_id, download_data=False)
        # Use the dataset name for the experiment (replace spaces with underscores for file system safety)
        dataset_name = dataset.name.replace(" ", "_")
        config.exp_name = dataset_name
        print("Using single dataset for training:", dataset_name)
    config.exp_path = f"runs/{config.folder}/{config.exp_name}"

    # Only rank 0 should modify the directory
    is_master = (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0)

    # directory setup and checkpoint handling
    if is_master:
        if os.path.exists(config.exp_path):
            print("Directory already exists:", config.exp_path)
            # restart training
            if config.training.reset_policy == "rm" or not os.path.exists(
                f"{config.exp_path}/latest.ckpt"
            ):
                print("Remove the existing directory.")
                shutil.rmtree(config.exp_path, ignore_errors=True)
                os.makedirs(config.exp_path, exist_ok=True)
            # continue training from a saved checkpoint
            elif config.training.reset_policy == "cnt":
                load_state_from_saved = True
                checkpoint = torch.load(f"{config.exp_path}/latest.ckpt")
                model_state = checkpoint["model"]
                opt_state = checkpoint["opt"]
                stats = checkpoint["stats"]
                non_saved_num_epochs = config.training.num_epochs
                config.training.num_epochs = non_saved_num_epochs
                print("Continue training. Using saved config.")
            else:
                raise ValueError(
                    "Invalid reset_policy: must be either 'cnt' (resume) or 'rm' (delete)."
                )
        else:
            os.makedirs(config.exp_path, exist_ok=True)
            print("Created directory: ", config.exp_path)

    # Synchronize all processes so that the directory is created before continuing
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Check if the number of GPUs matches the configuration
    # assert torch.cuda.device_count() == len(
    #     config.env.gpus
    # ), f"Number of GPUs does not match the number of GPUs in the config, expected {len(config.env.gpus)}, found {torch.cuda.device_count()}"

    assert (
        config.training.batch_size % len(config.env.gpus) == 0
    ), "Batch size should be divisible by the number of GPUs"

    # adapt batch size for distributed training
    config.training.batch_size //= len(config.env.gpus)

    # save config
    OmegaConf.save(config=config, f=f"{config.exp_path}/config.yaml")

    print(f"Using {config.env.gpus} GPUs")

    # instantiate the model
    model = TabDPTModel(
        dropout=config.training.dropout,
        n_out=config.model.max_num_classes,
        nhead=config.model.nhead,
        nhid=config.model.nhid_factor * config.model.emsize,
        ninp=config.model.emsize,
        nlayers=config.model.nlayers,
        num_features=config.model.max_num_features,
    )
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.{2}f} M parameters")

    # load the model state if resuming from a saved checkpoint
    # TODO : handle model_state definition in a more robust way
    if load_state_from_saved:
        model = TabDPTModel.load(model_state, config)
        del model_state

    # instantiate the optimizer
    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        warmup_steps=1000,
        betas=(0.98, 0.999),
    )

    # load the optimizer state if resuming from a saved checkpoint
    # TODO : handle model_state definition in a more robust way
    if load_state_from_saved:
        optimizer.load_state_dict(opt_state)
        del opt_state

    return model, optimizer, stats


@hydra.main(version_base=None, config_path="configs", config_name="default_config")
def main(config: DictConfig):
    """Main function to run the training process.

    Args:
        config (DictConfig): The configuration for the training process.
    """

    # signal handling
    signal.signal(signal.SIGINT, signal_handler)

    # print config
    print("Config:", OmegaConf.to_yaml(config))

    # distributed training
    using_dist, rank, device = init_dist(config.env.device)

    model, optimizer, stats = set_experiment_from_config(config)
    model.to(device)

    # compile the model if specified in the config
    if config.training.compile:
        model = torch.compile(model)

    # using distributed data parallel if specified in the config
    if using_dist:
        print("Distributed training enabled.")
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
        )

    # Initialize the SummaryWriter for TensorBoard logging
    writer = SummaryWriter(config.exp_path)

    # set validation datasets
    # TODO : make this configurable
    vals = [
        FullEval(
            device=device,
            max_feat=config.model.max_num_features,
            use_retrieval=config.data.eval_retrieval,
        )
    ]

    # select the device type
    device_type = "cpu" if config.env.device == "cpu" else "cuda"

    # Use autocast for mixed precision training
    selected_autocast = torch.autocast(device_type=device_type, dtype=torch.bfloat16)

    # Initialize the FullDataset for training
    print("Initializing Dataset...")
    dataset = FullDataset(device, config)

    # initialize the DataLoader for the dataset
    data_loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        num_workers=config.env.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False,
        persistent_workers=True,
    )

    # Create an iterator for the DataLoader
    iter_data_loader = iter(data_loader)

    # Initialize the outer training loop
    for epoch in range(stats["epoch_in_training"] + 1, config.training.num_epochs + 1):
        print("Epoch ", epoch)

        # initialize the loss accumulators
        epoch_loss_cls = 0.0
        epoch_loss_reg = 0.0

        # set the model and optimizer to training mode
        model.train()
        if hasattr(optimizer, "train"):
            optimizer.train()

        # initialize the inner training loop
        for batch in tqdm(range(config.training.num_model_updates * config.training.num_agg)):
            # randomly set the evaluation position, i.e. the context length
                # ---- 1. choose a common eval_pos ---------------------------------------
            if dist.get_rank() == 0:
                eval_pos_t = torch.randint(
                    config.training.min_eval_pos,
                    config.training.max_eval_pos + 1,
                    (1,),
                    device=device,
                    dtype=torch.long,
                )
            else:
                eval_pos_t = torch.empty(1, dtype=torch.long, device=device)

            dist.broadcast(eval_pos_t, src=0)
            eval_pos = int(eval_pos_t.item())
            # eval_pos = random.randint(config.training.min_eval_pos, config.training.max_eval_pos)

            # get the next batch from the DataLoader
            x, y, task = [a.to(device) for a in next(iter_data_loader)]

            # efficient forward pass and loss computation
            with selected_autocast, sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                output, log_act_norms = model(
                    x, y.squeeze(-1)[:eval_pos], return_log_act_norms=True
                )

                # compute the losses
                loss_cls, loss_reg = compute_losses(output, task, y.squeeze(-1)[eval_pos:], config)

                # reweight the loss for combined classification and regression tasks
                loss = get_combined_loss(loss_cls, loss_reg, task, config)

            # detach the log_act_norms to avoid memory leak
            log_act_norms = {k: v.detach().item() for k, v in log_act_norms.items()}

            epoch_loss_cls += loss_cls.cpu().detach().item()
            epoch_loss_reg += loss_reg.cpu().detach().item()

            # backpropagation
            loss.backward()

            # gradient accumulation
            if (batch + 1) % config.training.num_agg == 0:
                # compute the global step
                global_step = (
                    batch + config.training.num_model_updates * config.training.num_agg * epoch
                )

                # log the losses and norms
                log_param_norms(model, writer, global_step, task, global_step)

                # clip gradients to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_grad_norm)

                # step the optimizer
                optimizer.step()
                optimizer.zero_grad()

        # log the average losses for the epoch
        writer.add_scalar(
            "loss/cls-average_train/", epoch_loss_cls / config.training.num_model_updates, epoch
        )
        writer.add_scalar(
            "loss/reg-average_train/", epoch_loss_reg / config.training.num_model_updates, epoch
        )

        # evaluate the model every eval_every epochs
        if epoch % config.logging.eval_every == 0:
            # synchronize all processes before evaluation
            if using_dist:
                torch.distributed.barrier()

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model_eval: nn.Module = model.module
            else:
                model_eval: nn.Module = model

            # save evaluation checkpoints and log metrics
            save_eval_callback(model_eval, optimizer, vals, epoch, config, writer, rank, stats)

            if using_dist:
                torch.distributed.barrier()

    # cleanup the model and optimizer
    if using_dist:
        cleanup()

    writer.close()


if __name__ == "__main__":
    main()
