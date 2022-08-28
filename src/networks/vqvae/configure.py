from enum import Enum
from logging import Logger
from typing import List

import numpy as np

from src.handlers.general import ParamSchedulerHandler
from src.networks.vqvae.baseline import BaselineVQVAE
from src.networks.vqvae.vqvae import VQVAEBase
from src.utils.general import get_max_decay_epochs
from src.utils.vqvae import DecayWarmups


class VQVAENetworks(Enum):
    BASELINE_VQVAE = "baseline_vqvae"


def get_vqvae_network(config: dict) -> VQVAEBase:
    if config["network"] == VQVAENetworks.BASELINE_VQVAE.value:
        network = BaselineVQVAE(
            n_levels=config["no_levels"],
            downsample_parameters=config["downsample_parameters"],
            upsample_parameters=config["upsample_parameters"],
            n_embed=config["num_embeddings"][0],
            embed_dim=config["embedding_dim"][0],
            commitment_cost=config["commitment_cost"][0],
            n_channels=config["no_channels"],
            n_res_channels=config["no_channels"],
            n_res_layers=config["no_res_layers"],
            p_dropout=config["dropout"],
            vq_decay=config["decay"][0],
            use_subpixel_conv=config["use_subpixel_conv"],
        )
    else:
        raise ValueError(
            f"VQVAE unknown. Was given {config['network']} but choices are {[vqvae.value for vqvae in VQVAENetworks]}."
        )

    return network


def add_vqvae_network_handlers(
    train_handlers: List, vqvae: VQVAEBase, config: dict, logger: Logger
) -> List:

    if config["decay_warmup"] == DecayWarmups.STEP.value:
        delta_step = (0.99 - config["decay"]) / 4
        stair_steps = np.linspace(0, config["max_decay_epochs"], 5)[1:]

        def decay_anealing(current_step: int) -> float:
            if (current_step + 1) >= stair_steps[3]:
                return config["decay"] + 4 * delta_step
            if (current_step + 1) >= stair_steps[2]:
                return config["decay"] + 3 * delta_step
            if (current_step + 1) >= stair_steps[1]:
                return config["decay"] + 2 * delta_step
            if (current_step + 1) >= stair_steps[0]:
                return config["decay"] + delta_step
            return config["decay"]

        train_handlers += [
            ParamSchedulerHandler(
                parameter_setter=vqvae.set_ema_decay,
                value_calculator=decay_anealing,
                vc_kwargs={},
                epoch_level=True,
            )
        ]
    elif config["decay_warmup"] == DecayWarmups.LINEAR.value:
        train_handlers += [
            ParamSchedulerHandler(
                parameter_setter=vqvae.set_ema_decay,
                value_calculator="linear",
                vc_kwargs={
                    "initial_value": config["decay"],
                    "step_constant": 0,
                    "step_max_value": config["max_decay_epochs"]
                    if isinstance(config["max_decay_epochs"], int)
                    else get_max_decay_epochs(config=config, logger=logger),
                    "max_value": 0.99,
                },
                epoch_level=True,
            )
        ]

    return train_handlers
