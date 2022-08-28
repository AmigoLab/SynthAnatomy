import torch.nn as nn

from src.networks.discriminator.utils import DiscriminatorNetworks
from src.networks.discriminator.baseline import BaselineDiscriminator


def get_discriminator_network(config: dict) -> nn.Module:
    if config["discriminator_network"] == DiscriminatorNetworks.BASELINE_DISCRIMINATOR.value:
        network = BaselineDiscriminator(
            input_nc=1,
            ndf=64,
            n_layers=3
        )
    else:
        raise ValueError(
            f"Discriminator unknown. Was given {config['discriminator_network']} but choices are"
            f" {[discriminator.value for discriminator in DiscriminatorNetworks]}."
        )

    return network
