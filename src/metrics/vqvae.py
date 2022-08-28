import pytorch_msssim
import torch
import torch.nn.functional as F

from typing import Sequence

from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from ignite.exceptions import NotComputableError


class MultiScaleSSIM(Metric):
    def __init__(self, output_transform=lambda x: x, ms_ssim_kwargs=None):
        self._accumulator = None
        self._count = None

        self._ms_ssim_kwargs = {
            "data_range": 1,
            "win_size": 11,
            "win_sigma": 1.5,
            "size_average": False,
            "weights": None,
            "K": (0.01, 0.03),
        }

        if ms_ssim_kwargs:
            self._ms_ssim_kwargs.update(ms_ssim_kwargs)

        super(MultiScaleSSIM, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self._accumulator = 0
        self._count = 0
        super(MultiScaleSSIM, self).reset()

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]):
        y_pred, y = output
        y = y.float()
        y_pred = y_pred.float()

        if y.shape != y_pred.shape:
            raise ValueError("y_pred and y should have same shapes.")

        self._accumulator += torch.sum(
            pytorch_msssim.ms_ssim(X=y, Y=y_pred, **self._ms_ssim_kwargs)
        ).item()

        self._count += y.shape[0]

    @sync_all_reduce("_accumulator", "_count")
    def compute(self):
        if self._count == 0:
            raise NotComputableError(
                "MultiScaleSSIM must have at least one example before it can be computed."
            )
        return self._accumulator / self._count


class MAE(Metric):
    def __init__(self, output_transform=lambda x: x):
        self._accumulator = None
        self._count = None
        super(MAE, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self._accumulator = 0
        self._count = 0
        super(MAE, self).reset()

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]):
        y_pred, y = output
        y = y.float()
        y_pred = y_pred.float()

        if y.shape != y_pred.shape:
            raise ValueError("y_pred and y should have same shapes.")

        self._accumulator += (
            F.l1_loss(input=y_pred, target=y, reduction="mean")
        ).item() * y.shape[0]

        self._count += y.shape[0]

    @sync_all_reduce("_accumulator", "_count")
    def compute(self):
        if self._count == 0:
            raise NotComputableError(
                "MAE must have at least one example before it can be computed."
            )
        return self._accumulator / self._count


class MSE(Metric):
    def __init__(self, output_transform=lambda x: x):
        self._accumulator = None
        self._count = None
        super(MSE, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self._accumulator = 0
        self._count = 0
        super(MSE, self).reset()

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]):
        y_pred, y = output
        y = y.float()
        y_pred = y_pred.float()

        if y.shape != y_pred.shape:
            raise ValueError("y_pred and y should have same shapes.")

        self._accumulator += (
            F.mse_loss(input=y_pred, target=y, reduction="mean")
        ).item() * y.shape[0]

        self._count += y.shape[0]

    @sync_all_reduce("_accumulator", "_count")
    def compute(self):
        if self._count == 0:
            raise NotComputableError(
                "MSE must have at least one example before it can be computed."
            )
        return self._accumulator / self._count
