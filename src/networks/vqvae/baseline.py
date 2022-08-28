import torch
import torch.nn as nn
import torch.distributed.distributed_c10d as dist

from typing import Dict, List, Sequence, Union, Tuple
from monai.networks.blocks import SubpixelUpsample
from torch.nn import functional as F

from src.networks.vqvae.vqvae import VQVAEBase

# Isolated baseline implementation provided by @warvito and should be used as minimum performance
# for any further improvement. This implementation is 3D only. It was extracted from his codebase
# at commit https://github.com/Warvito/Dan_Data/blob/97ccba41d8ee1e811b66e79254fcce6bf3a8d0f9/
# Based on following files
# - Network - https://github.com/Warvito/Dan_Data/blob/97ccba41d8ee1e811b66e79254fcce6bf3a8d0f9/models/vqvae.py
# - Quantizer - https://github.com/Warvito/Dan_Data/blob/97ccba41d8ee1e811b66e79254fcce6bf3a8d0f9/layers/quantizer.py
# - Default values - https://github.com/Warvito/Dan_Data/blob/97ccba41d8ee1e811b66e79254fcce6bf3a8d0f9/runai/submit_train_vqgan.sh
# Modifications were done to make it compliant with the VQVAE interface


# This is an implementation trick to work around a DDP and AMP where FP16 were encountered during high batch size
# training. Encapsulating the forward function in a method was tried but it resulted in AMP mishandling the output
# type.
class Quantizer_impl(nn.Module):
    def __init__(self, n_embed, embed_dim, eps):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.eps = eps

        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.requires_grad = False
        self.weight = self.embedding.weight

        self.register_buffer("N", torch.zeros(n_embed))
        self.register_buffer("embed_avg", self.weight.data.clone())

    @torch.cuda.amp.autocast(enabled=False)
    def forward(
        self, x: torch.Tensor, decay: float, commitment_cost: float
    ) -> List[torch.Tensor]:
        b, c, h, w, d = x.shape
        x = x.float()

        # convert inputs from BCHW -> BHWC and flatten input
        flat_inputs = x.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.embed_dim)

        # Calculate distances
        distances = (
            (flat_inputs ** 2).sum(dim=1, keepdim=True)
            - 2 * torch.mm(flat_inputs, self.weight.t())
            + (self.weight ** 2).sum(dim=1, keepdim=True).t()
        )

        # Encoding
        embed_idx = torch.max(-distances, dim=1)[1]
        embed_onehot = F.one_hot(embed_idx, self.n_embed).type_as(flat_inputs)

        # Quantize and unflatten
        embed_idx = embed_idx.view(b, h, w, d)

        # Embed
        quantized = self.embedding(embed_idx).permute(0, 4, 1, 2, 3).contiguous()

        # Use EMA to update the embedding vectors
        if self.training:
            with torch.no_grad():
                encodings_sum = embed_onehot.sum(0)
                dw = torch.mm(embed_onehot.t(), flat_inputs)
                if dist.is_initialized():
                    dist.all_reduce(tensor=encodings_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(tensor=dw, op=dist.ReduceOp.SUM)

                # Laplace smoothing of the cluster size
                self.N.data.mul_(decay).add_(torch.mul(encodings_sum, 1 - decay))
                self.embed_avg.data.mul_(decay).add_(torch.mul(dw, 1 - decay))

                n = self.N.sum()
                W = (self.N + self.eps) / (n + self.n_embed * self.eps) * n
                self.weight.data.copy_(self.embed_avg / W.unsqueeze(1))

        latent_loss = commitment_cost * F.mse_loss(quantized.detach(), x)

        # Stop optimization from accessing the embedding
        quantized_st = (quantized - x).detach() + x

        return quantized_st, latent_loss, embed_idx

    @torch.cuda.amp.autocast(enabled=False)
    def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(embedding_indices).permute(0, 4, 1, 2, 3).contiguous()


class Quantizer(nn.Module):
    def __init__(self, n_embed, embed_dim, commitment_cost=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.impl = Quantizer_impl(n_embed, embed_dim, eps)

        self.n_embed = n_embed
        self.commitment_cost = commitment_cost
        self.decay = decay

        self.perplexity_code: torch.Tensor = torch.rand(1)

    def forward(self, x):
        quantized_st, latent_loss, embed_idx = self.impl(
            x, self.decay, self.commitment_cost
        )

        avg_probs = (
            lambda e: torch.histc(e.float(), bins=self.n_embed, max=self.n_embed)
            .float()
            .div(e.numel())
        )

        perplexity = lambda avg_probs: torch.exp(
            -torch.sum(avg_probs * torch.log(avg_probs + 1e-10))
        )

        self.perplexity_code = perplexity(avg_probs(embed_idx))

        return quantized_st, latent_loss

    def get_ema_decay(self) -> float:
        return self.decay

    def set_ema_decay(self, decay: float) -> float:
        self.decay = decay

        return self.get_ema_decay()

    def get_commitment_cost(self) -> float:
        return self.commitment_cost

    def set_commitment_cost(self, commitment_cost) -> float:
        self.commitment_cost = commitment_cost

        return self.get_commitment_cost()

    def get_perplexity(self) -> torch.Tensor:
        return self.perplexity_code

    def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        return self.impl.embed(embedding_indices=embedding_indices)

    def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
        return self.impl(encodings, self.decay, self.commitment_cost)


class ResidualLayer(nn.Sequential):
    def __init__(self, n_channels, n_res_channels, p_dropout):
        super().__init__(
            nn.Conv3d(n_channels, n_res_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Dropout3d(p_dropout),
            nn.Conv3d(n_res_channels, n_channels, kernel_size=1),
        )

    def forward(self, x):
        return F.relu(x + super().forward(x), True)


class BaselineVQVAE(VQVAEBase, nn.Module):
    def __init__(
        self,
        n_levels: int = 3,
        downsample_parameters: Tuple[Tuple[int, int, int, int], ...] = (
            (4, 2, 1, 1),
            (4, 2, 1, 1),
            (4, 2, 1, 1),
        ),
        upsample_parameters: Tuple[Tuple[int, int, int, int, int], ...] = (
            (4, 2, 1, 0, 1),
            (4, 2, 1, 0, 1),
            (4, 2, 1, 0, 1),
        ),
        n_embed: int = 256,
        embed_dim: int = 256,
        n_channels: int = 144,
        n_res_channels: int = 144,
        n_res_layers: int = 3,
        p_dropout: float = 0.0,
        commitment_cost: float = 0.25,
        vq_decay: float = 0.5,
        use_subpixel_conv: bool = False,
    ):
        super().__init__()
        assert n_levels == len(downsample_parameters) and n_levels == len(
            upsample_parameters
        ), (
            f"downsample_parameters, upsample_parameters must have the same number of elements as n_levels. "
            f"But got {len(downsample_parameters)} and {len(upsample_parameters)}, instead of {n_levels}."
        )
        self.n_levels = n_levels
        self.downsample_parameters = downsample_parameters
        self.upsample_parameters = upsample_parameters
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.use_subpixel_conv = use_subpixel_conv
        self.n_channels = n_channels
        self.n_res_channels = n_res_channels
        self.n_res_layers = n_res_layers
        self.p_dropout = p_dropout
        self.commitment_cost = commitment_cost
        self.vq_decay = vq_decay

        self.encoder = self.construct_encoder()
        self.quantizer = self.construct_quantizer()
        self.decoder = self.construct_decoder()

        self.n_embed = n_embed

    def construct_encoder(self) -> nn.ModuleList:
        modules = []

        for i in range(self.n_levels):
            modules.append(
                nn.Conv3d(
                    in_channels=1 if i == 0 else self.n_channels // 2,
                    out_channels=self.n_channels
                    // (1 if i == self.n_levels - 1 else 2),
                    kernel_size=self.downsample_parameters[i][0],
                    stride=self.downsample_parameters[i][1],
                    padding=self.downsample_parameters[i][2],
                    dilation=self.downsample_parameters[i][3],
                )
            )
            modules.append(nn.ReLU())
            modules.append(
                nn.Sequential(
                    *[
                        ResidualLayer(
                            self.n_channels // (1 if i == self.n_levels - 1 else 2),
                            self.n_res_channels // (1 if i == self.n_levels - 1 else 2),
                            self.p_dropout,
                        )
                        for _ in range(self.n_res_layers)
                    ]
                )
            )

        modules.append(
            nn.Conv3d(self.n_channels, self.embed_dim, 3, stride=1, padding=1)
        )

        return nn.ModuleList([nn.Sequential(*modules)])

    def construct_quantizer(self) -> nn.ModuleList:
        quantizer = Quantizer(
            self.n_embed,
            self.embed_dim,
            commitment_cost=self.commitment_cost,
            decay=self.vq_decay,
        )
        return nn.ModuleList([quantizer])

    def construct_decoder(self) -> nn.ModuleList:
        modules = [nn.Conv3d(self.embed_dim, self.n_channels, 3, stride=1, padding=1)]

        for i in range(self.n_levels):
            modules.append(
                nn.Sequential(
                    *[
                        ResidualLayer(
                            self.n_channels // (1 if i == 0 else 2),
                            self.n_res_channels // (1 if i == 0 else 2),
                            self.p_dropout,
                        )
                        for _ in range(self.n_res_layers)
                    ]
                )
            )
            modules.append(
                SubpixelUpsample(
                    dimensions=3,
                    in_channels=self.n_channels // 2,
                    out_channels=1,
                    scale_factor=self.upsample_parameters[i][1],
                    apply_pad_pool=True,
                    bias=True,
                )
                if i == self.n_levels - 1 and self.use_subpixel_conv
                else nn.ConvTranspose3d(
                    in_channels=self.n_channels // (1 if i == 0 else 2),
                    out_channels=(
                        1 if i == self.n_levels - 1 else self.n_channels // 2
                    ),
                    kernel_size=self.upsample_parameters[i][0],
                    stride=self.upsample_parameters[i][1],
                    padding=self.upsample_parameters[i][2],
                    output_padding=self.upsample_parameters[i][3],
                    dilation=self.upsample_parameters[i][4],
                )
            )
            # We do not have an output activation
            if i != self.n_levels - 1:
                modules.append(nn.ReLU())

        return nn.ModuleList([nn.Sequential(*modules)])

    def get_ema_decay(self) -> Sequence[float]:
        return [self.quantizer[0].get_ema_decay()]

    def set_ema_decay(self, decay: Union[Sequence[float], float]) -> Sequence[float]:
        self.quantizer[0].set_ema_decay(decay[0] if isinstance(decay, list) else decay)

        return self.get_ema_decay()

    def get_commitment_cost(self) -> Sequence[float]:
        return [self.quantizer[0].get_commitment_cost()]

    def set_commitment_cost(
        self, commitment_factor: Union[Sequence[float], float]
    ) -> Sequence[float]:
        self.quantizer[0].set_commitment_cost(
            commitment_factor[0]
            if isinstance(commitment_factor, list)
            else commitment_factor
        )

        return self.get_commitment_cost()

    def get_perplexity(self) -> Sequence[float]:
        return [self.quantizer[0].get_perplexity()]

    def get_last_layer(self) -> nn.parameter.Parameter:
        return list(self.decoder.modules())[-1].weight

    def encode(self, images: torch.Tensor) -> List[torch.Tensor]:
        return [self.encoder[0](images)]

    def quantize(
        self, encodings: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        x, x_loss = self.quantizer[0](encodings[0])
        return [x], [x_loss]

    def decode(self, quantizations: List[torch.Tensor]) -> torch.Tensor:
        x = self.decoder[0](quantizations[0])
        return x

    def index_quantize(self, images: torch.Tensor) -> List[torch.Tensor]:
        encodings = self.encode(images)
        _, _, encoding_indices = self.quantizer[0].quantize(encodings[0])

        return [encoding_indices]

    def decode_samples(self, embedding_indices: List[torch.Tensor]) -> torch.Tensor:
        samples_codes = self.quantizer[0].embed(embedding_indices[0])
        samples_images = self.decode([samples_codes])

        return samples_images

    def forward(self, images: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        encodings = self.encode(images)
        quantizations, quantization_losses = self.quantize(encodings)
        reconstruction = self.decode(quantizations)

        return {
            "reconstruction": [reconstruction],
            "quantization_losses": quantization_losses,
        }
