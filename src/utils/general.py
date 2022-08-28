import os
from glob import glob
from logging import Logger
from math import e, log, ceil
from pathlib import Path
from typing import Tuple

import ignite
import ignite.distributed as idist
import pandas as pd
import monai
import torch
from ignite.utils import setup_logger
from monai.utils import set_determinism

from src.utils.vqvae import VQVAEModes
from src.utils.transformer import TransformerModes


def get_gamma(
    config: dict,
    logger: Logger = None,
    epoch_level: bool = False,
    minimum_lr: float = 1e-5,
) -> float:
    """
    Rule of thumb gamma calculator of torch.optim.lr_scheduler.ExponentialLR. It aims to have the learning rate reach
    the 1e-5 value over the whole training period.

    Args:
        config (dict): Dictionary that holds the whole configuration
        logger (Logger): Logger for printing the logic
        epoch_level (bool): Whether or not we decay over epochs or iterations
        minimum_lr (float): Target minimum learning rate
    Returns:
        float: Correct gamma
    """
    steps = config["epochs"] * (1 if epoch_level else config["epoch_length"])
    gamma = e ** (log(minimum_lr / config["learning_rate"]) / steps)

    if logger:
        logger.info("Learning rate decay calculation:")
        logger.info(f"\tInitial learning rate: {config['learning_rate']}")
        logger.info(f"\tDecaying over {steps} steps")
        logger.info(f"\tFinal learning rate: {minimum_lr}")
        logger.info(f"\tGamma: {gamma}")

    return gamma


def get_max_decay_epochs(config: dict, logger: Logger = None) -> int:
    """
    Rule of thumb max_decay_epochs calculator. It aims to have the Exponential Moving Average see at least 200 epochs of
    437 iterations with 32 samples worth of samples before it reaches the max value of 0.99.

    Args:
        config (Dict): Dictionary that holds the whole configuration
        logger (Logger): Logger for printing the logic

    Returns:
        int: Correct max_decay_epochs
    """
    rule_of_thumb_samples = 200 * 437 * 32
    max_decay_epochs = ceil(
        rule_of_thumb_samples / (config["epoch_length"] * config["batch_size"])
    )
    if logger:
        logger.info("Max decay epochs calculation:")
        logger.info(f"\tRule of thumb number of iterations: {rule_of_thumb_samples}")
        logger.info(f"\tProjected max_decay_epochs: {max_decay_epochs}")

    return max_decay_epochs


def check_for_checkpoints(config: dict, logger: Logger = None) -> Path:
    """
    It checks for existing checkpoints in config['checkpoint_directory'] and if found it returns a Path object
    constructed from its absolute path. In training mode, if config['starting_epoch'] is -1 it will return the latest
    checkpoint otherwise it will search for the checkpoint found at that epoch. In evaluation mode,
    if config['starting_epoch'] is -1 it will either select the most recent epoch or the epoch with best
    performance as specified by config["evaluation_checkpoint"].

    The search for checkpoints is based on MONAI's saving logic where the checkpoints are saved as following:
    /path/to/checkpoints/checkpoint_epoch=2.pt

    Args:
        config (Dict): Dictionary that holds the whole configuration
        logger (Logger): Logger for printing the logic
    Returns:
        Paths: If a checkpoint is fount it returns its Path object, otherwise None.
    """
    checkpoint_fp = None
    if config["mode"] in [VQVAEModes.TRAINING.value, TransformerModes.TRAINING.value]:
        if config["starting_epoch"] == -1:
            # Typical checkpoint paths:
            # /path/to/checkpoints/checkpoint_epoch=2.pt
            # Getting the file name ->  checkpoint_epoch=2.pt
            # Getting everything left to -> 2.pt
            # Getting epoch number -> 2
            checkpoints = [
                int(e.split("/")[-1].split("_")[-1].split("=")[-1].split(".")[0])
                for e in glob(config["checkpoint_directory"] + "*checkpoint_epoch*.pt")
            ]

            checkpoints.sort()

            config["starting_epoch"] = checkpoints[-1]

        if config["starting_epoch"] > 0:
            checkpoint_fp = Path(
                f"{config['checkpoint_directory']}checkpoint_epoch={config['starting_epoch']}.pt"
            )

            assert (
                checkpoint_fp.exists()
            ), f"Checkpoint '{checkpoint_fp.as_posix()}' is not found."

            if logger:
                logger.info("Resume from a checkpoint:")
                logger.info(f"\tStarting epoch: {config['starting_epoch']}.")
                logger.info(f"\tCheckpoint found at: {checkpoint_fp}.")

    if config["mode"] not in [
        VQVAEModes.TRAINING.value,
        TransformerModes.TRAINING.value,
    ]:
        if config["starting_epoch"] > 0:
            checkpoint_fp = Path(
                f"{config['checkpoint_directory']}checkpoint_epoch={config['starting_epoch']}.pt"
            )

            assert (
                checkpoint_fp.exists()
            ), f"Checkpoint '{checkpoint_fp.as_posix()}' is not found."

            if logger:
                logger.info("Evaluating from a chosen checkpoint:")
                logger.info(f"\tEpoch: {config['starting_epoch']}.")
                logger.info(f"\tCheckpoint found at: {checkpoint_fp}.")

        elif config["evaluation_checkpoint"] == "recent":
            checkpoints = [
                int(e.split("/")[-1].split("_")[-1].split("=")[-1].split(".")[0])
                for e in glob(config["checkpoint_directory"] + "*checkpoint_epoch*.pt")
            ]

            checkpoints.sort()

            checkpoint_fp = Path(
                f"{config['checkpoint_directory']}checkpoint_epoch={checkpoints[-1]}.pt"
            )
            if logger:
                logger.info("Evaluating using most recent checkpoint:")
                logger.info(f"\tCheckpoint found at: {checkpoint_fp}.")

        elif config["evaluation_checkpoint"] == "best":
            checkpoints = glob(
                config["checkpoint_directory"] + "checkpoint_key_metric*.pt"
            )
            assert (
                len(checkpoints) == 1
            ), f"Should only be one best metric checkpoint, found {checkpoints}"
            checkpoint_fp = Path(checkpoints[0])
            if logger:
                logger.info("Evaluating using best performing checkpoint:")
                logger.info(f"\tCheckpoint found at: {checkpoint_fp}.")

    return checkpoint_fp


def log_basic_info(config: dict, logger: Logger):
    """
    Prints the version of PyTorch, Ignite and MONAI as long with configuration found in config.

    Args:
        config (Dict): Dictionary that holds the whole configuration
        logger (Logger): Logger for printing the logic
    """
    logger.info(f"Train VQ-VAE")
    logger.info(f"\tPyTorch version: {torch.__version__}")
    logger.info(f"\tIgnite version: {ignite.__version__}")
    logger.info(f"\tMONAI version: {monai.__version__}")

    if idist.get_world_size() > 1 and False:
        logger.info("Distributed setting:")
        logger.info(f"\tBackend: {idist.backend()}")
        logger.info(f"\tWorld Size: {idist.get_world_size()}")
        logger.info(f"\tDistributed Port: {config['distributed_port']}")
        for key, value in os.environ.items():
            if "NCCL" in key:
                logger.info(f"\t{key}: {value}")

    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"\t{key}: {value}")


def get_device(config: dict, logger: Logger) -> torch.device:
    """
    Given the GPU index via config['device'] it will set the GPU device to the GPU at that index given the
    CUDA_DEVICE_ORDER of PCI_BUS_ID.

    Args:
        config (Dict): Dictionary that holds the whole configuration
        logger (Logger): Logger for printing the logic
    Returns:
        torch.device: Returns a "cuda" torch.device
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    gpu_idx = config["local_rank"] if config["device"] == "ddp" else config["device"]

    torch_device = torch.device("cuda", gpu_idx)
    torch.cuda.set_device(torch_device)

    if config["device"] == "ddp":
        logger.info(
            f"Using Distributed Data Parallelism with {config['world_size']} GPUs."
        )
    else:
        logger.info(f"Using GPU index {config['device']} in PCI_BUS_ID order.")

    return torch_device


def create_folder_structure(config: dict, logger: Logger):
    """
    Creates the folder structure for the experiments. At config['project_directory'] the following structure will be
    created:
        config['project_directory']
            |- config['experiment_name']
                |- config['network']
                    |- checkpoints
                    |- logs
                    |- outputs
                    |- caching (if PersistentDataset is used)

    Args:
        config (Dict): Dictionary that holds the whole configuration
        logger (Logger): Logger for printing the logic
    """
    experiment_directory = (
        config["project_directory"]
        + config["experiment_name"]
        + "/"
        + config["network"]
    )

    checkpoint_directory = experiment_directory + "/checkpoints/"
    logs_directory = experiment_directory + "/logs/"
    outputs_directory = experiment_directory + "/outputs/"

    Path(checkpoint_directory).mkdir(parents=True, exist_ok=True)
    Path(logs_directory).mkdir(parents=True, exist_ok=True)
    Path(outputs_directory).mkdir(parents=True, exist_ok=True)

    cache_dir = experiment_directory + "/caching/"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # If the project exists already and the checkpoint directory is not empty
    # we will assume we want to restart from latest iteration
    if (
        os.path.exists(experiment_directory)
        and config["starting_epoch"] == 0
        and os.listdir(checkpoint_directory)
    ):
        config["starting_epoch"] = -1
        logger.warning("The experiment already exists. Checkpoints were found at: ")
        logger.warning(f"{checkpoint_directory}")
        logger.warning("The starting iteration has been modified from 0 to -1.")

    logger.info("Directory setting:")
    logger.info(f"\t{experiment_directory}")
    logger.info(f"\t{checkpoint_directory}")
    logger.info(f"\t{logs_directory}")
    logger.info(f"\t{outputs_directory}")
    logger.info(f"\t{cache_dir}")

    config["experiment_directory"] = experiment_directory
    config["checkpoint_directory"] = checkpoint_directory
    config["logs_directory"] = logs_directory
    config["outputs_directory"] = outputs_directory
    config["cache_dir"] = cache_dir


def log_network_size(network: torch.nn.Module, logger: Logger):
    """
    Logs the size of the network based on the number of trainable and total parameters.

    Args:
        network (torch.nn.Module): The network that will have its size logged
        logger (Logger): Logger for printing the logic
    """
    parameters = sum(p.numel() for p in network.parameters())
    trainable_parameters = sum(
        p.numel() for p in network.parameters() if p.requires_grad
    )

    logger.info(f"Number of parameters in network {type(network)}")
    logger.info(f"\tTrainable: {trainable_parameters}")
    logger.info(f"\tTotal: {parameters}")


def basic_initialization(config: dict, logger_name: str) -> Tuple[Logger, torch.device]:
    """
    Common initialization across the codebase.

    Create a logger with the name logger_name.

    Logs the details of the configuration.

    Set the GPU that will be used.

    Set deterministic behaviour and/or torch.backends.cudnn.benchmark.

    Creates the folder structure.

    Checks for existing checkpoints to load.

    Args:
        config (Dict): Dictionary that holds the whole configuration
        logger_name (str): The name of the logger

    Returns:
        Logger: The instantiated logger
        torch.device: The GPU device that was set
    """
    logger = setup_logger(name=logger_name, distributed_rank=config["rank"])

    log_basic_info(config=config, logger=logger)

    device = get_device(config=config, logger=logger)

    if config["deterministic"]:
        set_determinism(seed=config["seed"] + config["rank"])

    if config["cuda_benchmark"]:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    create_folder_structure(config=config, logger=logger)

    checkpoint_fp = check_for_checkpoints(config=config, logger=logger)
    config["checkpoint_fp"] = checkpoint_fp

    return logger, device


def quantize_conditionings(
    conditionings_path: str,
    id_column: str,
    chosen_conditionings: Tuple[str, ...],
    chosen_quantiles: Tuple[int, ...],
    output_path: str,
    output_filename: str,
):
    df = pd.read_csv(
        conditionings_path,
        usecols=chosen_conditionings + (id_column,),
        low_memory=False,
    )

    for cc, cq in zip(chosen_conditionings, chosen_quantiles):
        if cq is not None:
            df[cc] = pd.qcut(df[cc], cq, labels=False)

    df = df.dropna(axis=0, how="any")
    df = df.reset_index(drop=True)
    df[list(chosen_conditionings)] = df[list(chosen_conditionings)].astype(int)

    df.to_csv(f"{output_path}/{output_filename}.csv", index=False)


def uniform_sampling(
    conditionings_path: str, chosen_conditionings: Tuple[str, ...], output_path: str
):
    df = pd.read_csv(conditionings_path)

    dfgby = df.groupby(list(chosen_conditionings))

    n = min(dfgby.size())

    samples = [
        dfgby.get_group(key).sample(n=n, replace=False, random_state=0, axis=0)
        for key, item in dfgby
    ]

    uniform_df = pd.concat(samples)

    uniform_df.to_csv(f"{output_path}/uniform_quantized_conditioning.csv", index=False)
