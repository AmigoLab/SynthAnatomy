from typing import List, Any

import torch
from monai.inferers import Inferer

from src.networks.vqvae.vqvae import VQVAEBase


class VQVAEExtractionInferer(Inferer):
    def __init__(self, d_network: torch.nn.Module = None) -> None:
        Inferer.__init__(self)
        self.d_network = d_network

    def __call__(
        self, inputs: torch.Tensor, network: VQVAEBase, *args: Any, **kwargs: Any
    ):
        """
        Inferer for the VQVAE models that extracts the reconstructions and quantizations.

        Args:
            inputs: model input data for inference.
            network: trained VQVAE network on which .index_quantize and .decode_samples will be called.
            args: optional args to be passed to ``network``. It is ignored.
            kwargs: optional keyword args to be passed to ``network``. It is ignored.
        """
        return_dict = {}

        n = (
            network.module
            if isinstance(network, torch.nn.parallel.DistributedDataParallel)
            else network
        )

        embedding_indices = n.index_quantize(images=inputs)
        reconstructions = n.decode_samples(embedding_indices=embedding_indices)

        return_dict["reconstruction"] = reconstructions

        for idx, embedding_index in enumerate(embedding_indices):
            return_dict[f"quantization_{idx}"] = embedding_index

        if self.d_network:
            return_dict["adversarial_logits"] = self.d_network(return_dict["reconstruction"])

        return return_dict


class VQVAEDecodingInferer(Inferer):
    def __init__(
        self, num_quantization_levels: int, d_network: torch.nn.Module = None
    ) -> None:
        """
            Args:
                num_quantization_levels (int): How many quantization layers the network has.
        """
        Inferer.__init__(self)

        self.num_quantization_levels = num_quantization_levels
        self.d_network = d_network

    def __call__(
        self, inputs: List[torch.Tensor], network: VQVAEBase, *args: Any, **kwargs: Any
    ):
        """
        Inferer for the VQVAE models that decodes autoregressive samples and if possible saves the adversarial loss.

        Args:
            inputs: model input data for inference.
            network: trained VQVAE network on which .decode_samples will be called.
            args: optional args to be passed to ``network``. It is ignored.
            kwargs: optional keyword args to be passed to ``network``. It is ignored.

        """
        return_dict = {}

        n = (
            network.module
            if isinstance(network, torch.nn.parallel.DistributedDataParallel)
            else network
        )

        return_dict["sample"] = n.decode_samples(embedding_indices=inputs)

        if self.d_network:
            return_dict["adversarial_logits"] = self.d_network(return_dict["sample"])

        return return_dict
