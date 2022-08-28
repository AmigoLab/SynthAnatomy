from typing import Any, Optional

import numpy as np
import torch
from torch.nn import functional as F


class TransformerBase(torch.nn.Module):
    """ Abstract class for transformers."""

    @staticmethod
    def __top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
        """ From https://github.com/karpathy/minGPT/blob/master/mingpt/utils.py """
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[:, [-1]]] = -float("Inf")
        return out

    @torch.no_grad()
    def sample_next_index(
        self,
        x: torch.Tensor,
        conditioning: torch.Tensor = None,
        temperature: float = 1.0,
        sample: bool = True,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample single element of the sequence

        Based on https://github.com/karpathy/minGPT/blob/master/mingpt/utils.py

        Args:
            x: input sequence to condition the sampling of the next element.
            temperature: temperature value to scale the logits.
            sample: flag to define if the values are sampled from the distribution or take the most likely.
            top_k: optionally crop probabilities to only the top k options

        Returns:
            Next value of the sequence
        """
        self.eval()
        logits = self(x, conditioning)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            logits = self.__top_k_logits(logits, top_k)

        probs = F.softmax(logits, dim=-1)

        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)

        return ix

    @torch.no_grad()
    def sample(
        self,
        prefix: torch.Tensor,
        conditioning: torch.Tensor = None,
        temperature: float = 1.0,
        sample: bool = True,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate a new sequence sample and reshape it into an image format using the
        ordering attribute from the transformer.

        Based on https://github.com/karpathy/minGPT/blob/master/mingpt/utils.py

        Args:
            prefix: Torch tensor containing the initial sequence object to be completed. For example, this tensor can
                have the [n_samples, 1] format where the Begining Of Sequence (<BOS>) is repeated across all rows.
            temperature: temperature value to scale the logits.
            sample: flag to define if the values are sampled from the distribution or take the most likely.
            top_k: optionally crop probabilities to only the top k options
        Returns:
            Generated image.
        """
        steps = np.prod(self.ordering.dimensions)

        x = prefix
        for k in range(steps):
            ix = self.sample_next_index(
                x,
                conditioning=conditioning,
                temperature=temperature,
                sample=sample,
                top_k=top_k,
            )
            x = torch.cat((x, ix), dim=1)

        x = x[:, prefix.shape[1] :]
        x = x[:, self.ordering.get_revert_sequence_ordering()]
        x = x.reshape(x.shape[0], *self.ordering.dimensions)
        # Squeezing so that it can be cleanly passed to torch.nn.Embedding
        x = torch.squeeze(x, 1)

        return x

    def forward(self, x: torch.Tensor) -> Any:
        return self(x)
