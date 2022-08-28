from typing import Dict, Callable

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from src.handlers.general import TBSummaryTypes
from src.losses.adversarial.utils import AdversarialLosses


class AdversarialLoss(_Loss):
    def __init__(
        self,
        criterion: str = AdversarialLosses.LEAST_SQUARE.value,
        is_discriminator: bool = True,
        weight=None,
        size_average: bool = None,
        reduce: bool = None,
        reduction: str = "mean",
    ):
        super(AdversarialLoss, self).__init__(size_average, reduce, reduction)

        try:
            assert reduction in ["sum", "mean"]
        except AssertionError:
            raise ValueError("Reduction must be either 'sum' or 'mean'")

        self.criterion = criterion
        self.is_discriminator = is_discriminator
        self.criterion_function = AdversarialLoss.get_criterion_function(self.criterion)

        self._weight = weight

        self.summaries: Dict = {TBSummaryTypes.SCALAR: dict()}

    def forward(
        self, logits_fake: torch.Tensor, logits_real: torch.Tensor = None
    ) -> torch.Tensor:
        logits_fake = logits_fake.float()

        loss_fake = torch.mean(
            self.criterion_function(
                logits_fake, False if self.is_discriminator else True
            )
        )

        self.summaries[TBSummaryTypes.SCALAR][
            f"Loss-Adversarial_{'Discriminator' if self.is_discriminator else 'Generator'}-Reconstruction"
        ] = loss_fake

        loss = loss_fake

        if self.is_discriminator:
            logits_real = logits_real.float()
            loss_real = torch.mean(self.criterion_function(logits_real, True))
            self.summaries[TBSummaryTypes.SCALAR][
                f"Loss-Adversarial_Discriminator-Originals"
            ] = loss_real

            loss = 0.5 * (loss + loss_real)

        loss = self._weight * loss

        return loss

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries

    def get_weight(self) -> float:
        return self._weight

    def set_weight(self, weight: float) -> float:
        self._weight = weight

        return self.get_weight()

    @staticmethod
    def get_criterion_function(
        criterion
    ) -> Callable[[torch.Tensor, bool], torch.Tensor]:
        if criterion == AdversarialLosses.VANILLA.value:

            def criterion(logits: torch.Tensor, is_real: bool) -> torch.Tensor:
                # An equivalent explicit implementation would be
                #     loss_real = torch.mean(F.relu(1.0 - logits_real))
                #     loss_fake = torch.mean(F.relu(1.0 + logits_fake))
                return F.relu(1.0 + (-1 if is_real else 1) * logits)

        elif criterion == AdversarialLosses.HINGE.value:

            def criterion(logits: torch.Tensor, is_real: bool) -> torch.Tensor:
                # An equivalent explicit implementation would be
                #     loss_real = torch.mean(torch.nn.functional.softplus(-logits_real))
                #     loss_fake = torch.mean(torch.nn.functional.softplus(logits_fake))
                return torch.nn.functional.softplus((-1 if is_real else 1) * logits)

        elif criterion == AdversarialLosses.LEAST_SQUARE.value:

            def criterion(logits: torch.Tensor, is_real: bool) -> torch.Tensor:
                # An equivalent explicit implementation would be
                #     loss_real = torch.mean((logits_real - 1) ** 2)
                #     loss_fake = torch.mean(logits_fake ** 2)
                return (logits - (1 if is_real else 0)) ** 2

        return criterion
