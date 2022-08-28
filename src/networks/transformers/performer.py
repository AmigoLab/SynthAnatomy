import math
from typing import Optional, Tuple, Union, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from axial_positional_embedding import AxialPositionalEmbedding
from performer_pytorch.performer_pytorch import Performer as BasePerformer
from performer_pytorch.performer_pytorch import (
    cast_tuple,
    exists,
    AbsolutePositionalEmbedding,
    FixedPositionalEmbedding,
    Always,
    default,
)
from torch import nn
from src.networks.transformers.img2seq_ordering import Ordering
from src.networks.transformers.transformer import TransformerBase
from src.utils.transformer import TransformerConditioningType


class AbsoluteSpatialPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, spatial_indices_sequence: torch.Tensor):
        super().__init__()

        self.register_buffer("spatial_indices_sequence", spatial_indices_sequence)
        # Eliminating the last element as it is the predicted one
        self.spatial_indices_sequence = self.spatial_indices_sequence[:-1]

        self.padding = lambda x: F.pad(x, (0, 0, 1, 0, 0, 0), "constant", 0)

        self.emb = nn.Embedding(len(self.spatial_indices_sequence), dim)

    def forward(self, x):
        sc = self.emb(self.spatial_indices_sequence)
        sc = sc[None, : x.shape[1] - 1, :].to(x)
        sc = self.padding(sc)

        return sc


class FixedSpatialPositionalEmbedding(nn.Module):
    def __init__(self, dim, spatial_indices_sequence):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

        max_position = torch.max(spatial_indices_sequence)
        # The + 1 is for torch.arange to include the max_position as well
        position = torch.arange(0, max_position + 1, dtype=torch.float)

        sinusoid_inp = torch.einsum("d,j->ij", position, inv_freq)
        sinusoid_inp = sinusoid_inp[spatial_indices_sequence, :]

        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        # Eliminating the last element as it is the predicted one
        emb = emb[:-1]
        self.register_buffer("emb", emb)

        self.padding = lambda x: F.pad(x, (0, 0, 1, 0, 0, 0), "constant", 0)

    def forward(self, x):
        sc = self.emb
        # The sc is cropped for just the amount of spatial information
        sc = sc[None, : x.shape[1] - 1, :].to(x)
        sc = self.padding(sc)
        return sc


class Performer(TransformerBase):
    """
        NOTE: All tensor logic assumes the following ordering [Batch, Length, Channel]
    """

    def __init__(
        self,
        *,
        num_tokens: int,
        max_seq_len: int,
        dim: int,
        depth: int,
        heads: int,
        ordering: Ordering,
        dim_head: int = 64,
        local_attn_heads: int = 0,
        local_window_size: int = 256,
        causal: bool = True,
        ff_mult: int = 4,
        nb_features: Optional[int] = None,
        feature_redraw_interval: int = 1000,
        reversible: bool = False,
        ff_chunks: int = 1,
        ff_glu: bool = False,
        emb_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        generalized_attention: bool = False,
        kernel_fn: torch.nn.Module = nn.ReLU(),
        use_scalenorm: bool = False,
        use_rezero: bool = False,
        cross_attend: bool = False,
        no_projection: bool = False,
        tie_embed: bool = False,
        rotary_position_emb: bool = False,
        fixed_position_emb: bool = False,
        axial_position_emb: bool = False,
        axial_position_shape: Tuple[int, int] = None,
        auto_check_redraw: bool = True,
        qkv_bias: bool = False,
        attn_out_bias: bool = False,
        spatial_position_emb: str = None,
        spatial_shape: Union[Tuple[int, int], Tuple[int, int, int]] = None,
        conditioning_num_tokens: Optional[Tuple[int, ...]] = None,
        conditioning_type: str = TransformerConditioningType.NONE.value,
    ):
        super().__init__()
        local_attn_heads = cast_tuple(local_attn_heads)

        # Accounting for the number of prepended conditionings
        self.max_seq_len = max_seq_len + (
            len(conditioning_num_tokens)
            if conditioning_num_tokens
            and conditioning_type == TransformerConditioningType.PREPENDING.value
            else 0
        )
        self.token_emb = nn.Embedding(num_tokens, dim)

        assert (
            0 <= sum([rotary_position_emb, fixed_position_emb, axial_position_emb]) <= 1
        ), (
            f"rotary_position_emb, fixed_position_emb and axial_position_emb are exclusive, but received "
            f"{rotary_position_emb} {fixed_position_emb} and {axial_position_emb}."
        )
        if rotary_position_emb:
            self.pos_emb = FixedPositionalEmbedding(dim, self.max_seq_len)
            self.layer_pos_emb = FixedPositionalEmbedding(dim_head, self.max_seq_len)
        elif fixed_position_emb:
            self.pos_emb = FixedPositionalEmbedding(dim, self.max_seq_len)
            self.layer_pos_emb = Always(None)
        elif axial_position_emb:
            axial_position_shape = default(
                axial_position_shape, (math.ceil(self.max_seq_len / 64), 64)
            )
            self.pos_emb = AxialPositionalEmbedding(dim, axial_position_shape)
            self.layer_pos_emb = Always(None)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(dim, self.max_seq_len)
            self.layer_pos_emb = Always(None)

        self.ordering = ordering

        self.spatial_position_emb = nn.ModuleList()
        if spatial_position_emb:
            assert spatial_position_emb in [
                "fixed",
                "absolute",
            ], f"spatial_position_emb must be either 'fixed' or  'absolute', but got {spatial_position_emb}"

            axis = (0, 1, 2) if len(spatial_shape) == 3 else (0, 1)
            coord_channels = np.array(
                np.meshgrid(
                    *tuple(np.arange(0, s) for s in spatial_shape), indexing="ij"
                )
            )
            coord_channels = coord_channels[[s for s in axis]]

            for i in axis:
                spatial_indices_sequence = torch.from_numpy(
                    coord_channels[i, ...].flatten()
                )
                spatial_indices_sequence = self.ordering(spatial_indices_sequence)

                if spatial_position_emb == "fixed":
                    self.spatial_position_emb.append(
                        FixedSpatialPositionalEmbedding(
                            dim=dim, spatial_indices_sequence=spatial_indices_sequence
                        )
                    )
                elif spatial_position_emb == "absolute":
                    self.spatial_position_emb.append(
                        AbsoluteSpatialPositionalEmbedding(
                            dim=dim, spatial_indices_sequence=spatial_indices_sequence
                        )
                    )

        self.conditioning_emb = nn.ModuleList()
        self.conditioning_type = conditioning_type
        if conditioning_num_tokens:
            for cnt in conditioning_num_tokens:
                self.conditioning_emb.append(nn.Embedding(cnt, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.performer = BasePerformer(
            dim,
            depth,
            heads,
            dim_head,
            local_attn_heads,
            local_window_size,
            causal,
            ff_mult,
            nb_features,
            feature_redraw_interval,
            reversible,
            ff_chunks,
            generalized_attention,
            kernel_fn,
            use_scalenorm,
            use_rezero,
            ff_glu,
            ff_dropout,
            attn_dropout,
            cross_attend,
            no_projection,
            auto_check_redraw,
            qkv_bias,
            attn_out_bias,
        )
        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, num_tokens) if not tie_embed else None

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(
        self,
        x: torch.tensor,
        conditionings: Sequence[torch.Tensor] = None,
        return_encodings: bool = False,
        **kwargs,
    ):
        b, n, device = *x.shape, x.device
        assert (
            n <= self.max_seq_len
        ), f"sequence length {n} must be less than the max sequence length {self.max_seq_len}"

        x = self.token_emb(x)

        for spatial_pos_emb in self.spatial_position_emb:
            x += spatial_pos_emb(x)

        layer_pos_emb = self.layer_pos_emb(x)

        if (
            conditionings
            and self.conditioning_type != TransformerConditioningType.NONE.value
        ):
            if (
                self.conditioning_type
                == TransformerConditioningType.BOSREPLACEMENT.value
            ):
                c = torch.unsqueeze(torch.zeros_like(x[:, 0, :]), 1)

                for idx, conditioning_emb in enumerate(self.conditioning_emb):
                    c += conditioning_emb(conditionings[idx])

                x[:, 0, :] = c[:, 0, :]
            elif self.conditioning_type == TransformerConditioningType.PREPENDING.value:
                for idx, conditioning_emb in enumerate(self.conditioning_emb):
                    x = torch.cat((conditioning_emb(conditionings[idx]), x), dim=1)

        x += self.pos_emb(x)

        x = self.dropout(x)

        x = self.performer(x, pos_emb=layer_pos_emb, **kwargs)

        # norm and to logits
        x = self.norm(x)

        if (
            conditionings
            and self.conditioning_type != TransformerConditioningType.NONE.value
        ):
            if self.conditioning_type == TransformerConditioningType.PREPENDING.value:
                x = x[:, len(conditionings) :, :]

        if return_encodings:
            return x

        if exists(self.to_out):
            return self.to_out(x)

        return x @ self.token_emb.weight.t()
