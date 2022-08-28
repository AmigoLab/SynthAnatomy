from typing import Union, List

from torch.nn import Module
from torch.nn.modules.loss import _Loss

from src.handlers.general import ParamSchedulerHandler
from src.losses.vqvae.utils import VQVAELosses
from src.losses.vqvae.vqvae import (
    BaselineLoss,
    BaurLoss,
    MSELoss,
    SpectralLoss,
    HartleyLoss,
    JukeboxLoss,
    WaveGANLoss,
    PerceptualLoss,
    JukeboxPerceptualLoss,
    HartleyPerceptualLoss,
)


def get_vqvae_loss(config: dict) -> Union[_Loss, Module]:
    """
    Configures and returns the loss.

    Expects a 'loss' field in the dictionary which can have any of the vlaues found in src.losses.configure.VQVAELosses.
    """
    if config["loss"] == VQVAELosses.BAUR.value:
        loss = BaurLoss()
    elif config["loss"] == VQVAELosses.MSE.value:
        loss = MSELoss()
    elif config["loss"] == VQVAELosses.SPECTRAL.value:
        loss = SpectralLoss(dimensions=3)
    elif config["loss"] == VQVAELosses.HARTLEY.value:
        loss = HartleyLoss(dimensions=3)
    elif config["loss"] == VQVAELosses.JUKEBOX.value:
        loss = JukeboxLoss(dimensions=3)
    elif config["loss"] == VQVAELosses.WAVEGAN.value:
        loss = WaveGANLoss(dimensions=3)
    elif config["loss"] == VQVAELosses.PERCEPTUAL.value:
        loss = PerceptualLoss(dimensions=3, drop_ratio=0.50)
    elif config["loss"] == VQVAELosses.JUKEBOX_PERCEPTUAL.value:
        loss = JukeboxPerceptualLoss(dimensions=3, drop_ratio=0.5)
    elif config["loss"] == VQVAELosses.HARTLEY_PERCEPTUAL.value:
        loss = HartleyPerceptualLoss(dimensions=3, drop_ratio=0.5)
    elif config["loss"] == VQVAELosses.BASELINE.value:
        loss = BaselineLoss()
    else:
        raise ValueError(
            f"Loss function unknown. Was given {config['loss']} but choices are {[loss.value for loss in VQVAELosses]}."
        )

    return loss


def add_vqvae_loss_handlers(
    train_handlers: List, loss_function: Union[_Loss, Module], config: dict
) -> List:
    """
    Configures the required handlers for each loss. Please see implementation for details.
    """
    if config["loss"] == "baur":
        train_handlers += [
            ParamSchedulerHandler(
                parameter_setter=loss_function.set_gdl_factor,
                value_calculator="linear",
                vc_kwargs={
                    "initial_value": config["initial_factor_value"],
                    "step_constant": config["initial_factor_steps"],
                    "step_max_value": config["max_factor_steps"],
                    "max_value": config["max_factor_value"],
                },
                epoch_level=True,
            )
        ]
    return train_handlers
