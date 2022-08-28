"""
A collection of dictionary-based wrappers around the "vanilla" transforms for  transformers functions

Class names are ended with 'd' to denote dictionary-based transforms.
"""

from monai.config import KeysCollection
from monai.transforms.compose import MapTransform

from src.transforms.transformer.array import ConvertToSequence, AddBOS, QuantiseImage


class ConvertToSequenced(MapTransform):
    def __init__(self, keys: KeysCollection, ordering) -> None:
        super().__init__(keys)
        self.converter = ConvertToSequence(ordering)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.converter(d[key])
        return d


class AddBOSd(MapTransform):
    def __init__(self, keys: KeysCollection, prefix) -> None:
        super().__init__(keys)
        self.padder = AddBOS(prefix)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.padder(d[key])
        return d

class QuantiseImaged(MapTransform):
    def __init__(self, keys: KeysCollection, vqvae, level) -> None:
        super().__init__(keys)
        self.quantiser = QuantiseImage(
            vqvae=vqvae,
            level=level
        )

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.quantiser(d[key])
        return d


AddBOSD = AddBOSDict = AddBOSd
ConvertToSequenceD = ConvertToSequenceDict = ConvertToSequenced
QuantiseImageD = QuantiseImageDict = QuantiseImaged
