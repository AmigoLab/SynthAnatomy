from typing import Union, List

from torch.nn import Module
from torch.nn.modules.loss import _Loss

from src.losses.adversarial.utils import AdversarialLosses
from src.losses.adversarial.adversarial import AdversarialLoss


def get_discriminator_loss(config: dict) -> Union[_Loss, Module]:
    d_losses = [d_loss.value for d_loss in AdversarialLosses]

    if not config["discriminator_loss"] in d_losses:
        raise ValueError(
            f"Unknown discriminator loss. Available losses are {d_losses} but received {config['discriminator_loss']}"
        )

    loss = AdversarialLoss(
        criterion=config["discriminator_loss"], is_discriminator=True, weight=0.005
    )

    return loss


def get_generator_loss(config: dict):
    d_losses = [d_loss.value for d_loss in AdversarialLosses]

    if not config["generator_loss"] in d_losses:
        raise ValueError(
            f"Unknown generator loss. Available losses are {d_losses} but received {config['generator_loss']}"
        )

    loss = AdversarialLoss(
        criterion=config["generator_loss"], is_discriminator=False, weight=0.005
    )

    return loss


def get_criterion(criterion: str):
    return AdversarialLoss.get_criterion_function(criterion=criterion)
