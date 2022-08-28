from typing import Tuple, Dict


class AdversarialKeys:
    """
    A set of common keys for adversarial networks.
    """

    REALS = "reals"
    FAKES = "fakes"
    GLOSS = "g_loss"
    DLOSS = "d_loss"


def batch_decomposition(batch: Tuple) -> Tuple:
    if len(batch) == 2:
        inputs, targets = batch
        args: Tuple = tuple()
        kwargs: Dict = dict()
    else:
        inputs, targets, args, kwargs = batch

    return inputs, targets, args, kwargs
