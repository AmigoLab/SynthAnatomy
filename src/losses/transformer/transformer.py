from typing import Dict

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from src.handlers.general import TBSummaryTypes


class CELoss(_Loss):
    def __init__(
        self, weight=None, size_average: bool = None, reduce: bool = None, reduction: str = "mean"
    ):
        super(CELoss, self).__init__(size_average, reduce, reduction)

        try:
            assert reduction in ["sum", "mean"]
        except AssertionError:
            raise ValueError("Reduction must be either 'sum' or 'mean'")

        self.summaries: Dict = {TBSummaryTypes.SCALAR: dict()}
        self._weight = weight

    def forward(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = y.long()
        y_pred = y_pred.float()

        loss = F.cross_entropy(
            input=y_pred, target=y, reduction=self.reduction, weight=self._weight
        )
        self.summaries[TBSummaryTypes.SCALAR]["Loss-CE-Prediction"] = loss

        return loss

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries
