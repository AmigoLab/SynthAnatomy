import torch
import torch.nn.functional as F

from typing import Sequence

from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from ignite.exceptions import NotComputableError


class CE(Metric):
    def __init__(self, output_transform=lambda x: x, weight=None):
        self._accumulator = None
        self._count = None
        self._weight = weight
        super(CE, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self._accumulator = 0
        self._count = 0
        super(CE, self).reset()

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]):
        y_pred, y = output
        y = y.long()
        y_pred = y_pred.float()

        if y.shape[2:] != y_pred.shape[3:]:
            raise ValueError("y_pred and y should have same spatial shape.")

        # We use reduction="mean" and then multiply by y.shape[0] because we want the average CE of each sample.
        # If we use "sum" we get the sum of all classes, not the sumt of the average CE.
        self._accumulator += (
            F.cross_entropy(
                input=y_pred, target=y, reduction="mean", weight=self._weight
            )
        ).item() * y.shape[0]

        self._count += y.shape[0]

    @sync_all_reduce("_accumulator", "_count")
    def compute(self):
        if self._count == 0:
            raise NotComputableError(
                "CE must have at least one example before it can be computed."
            )
        return self._accumulator / self._count
