"""
A collection of "vanilla" transforms for transformers functions
"""
import torch
from monai.transforms.compose import Transform
import torch.nn.functional as F
from typing import Union
from src.networks.transformers.img2seq_ordering import Ordering


class ConvertToSequence(Transform):
    """ Transforms 2D/3D images into a 1D sequence."""

    def __init__(self, ordering: Ordering) -> None:
        self.ordering = ordering

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        img = img.reshape(-1)
        return self.ordering(img)


class AddBOS(Transform):
    """ Add Beginning Of Sentence (<BOS>) token to the sequences."""

    def __init__(self, prefix: int) -> None:
        if not (isinstance(prefix, int)):
            raise AssertionError("invalid prefix value.")
        self.prefix = prefix

    def __call__(self, seq: torch.Tensor) -> torch.Tensor:
        return F.pad(seq, (1, 0), "constant", self.prefix)


class QuantiseImage(Transform):
    """"""

    def __init__(self, vqvae, level) -> None:
        self.level = level
        self.vqvae = vqvae

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            quantizations = self.vqvae.index_quantize(img)
        return quantizations[self.level]
