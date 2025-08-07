from __future__ import annotations
from typing import Any, Optional, Tuple
import torch
from torch import nn
from torch import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm
from . import modules
from .commons import init_weights
from .spectral_modules import ISTFT


class Generator(nn.Module):
    def __init__(
        self: Generator,
        initial_channel: int,
        resblock: Optional[str],
        resblock_kernel_sizes: Tuple[int, ...],
        resblock_dilation_sizes: Tuple[Tuple[int, ...], ...],
        upsample_rates: Tuple[int, ...],
        upsample_initial_channel: int,
        upsample_kernel_sizes: Tuple[int, ...],
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        self.LRELU_SLOPE = 0.1
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = nn.Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock_module = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock_module(ch, k, d))

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(
        self: Generator,
        x: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            x = up(x)
            xs = torch.zeros(1)
            for j, resblock in enumerate(self.resblocks):
                index = j - (i * self.num_kernels)
                if index == 0:
                    xs = resblock(x)
                elif (index > 0) and (index < self.num_kernels):
                    xs += resblock(x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self: Generator) -> None:
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()



class VocosBackbone(nn.Module):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional model. Defaults to None.
    """

    def __init__(
        self: VocosBackbone,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
        adanorm_num_embeddings: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = adanorm_num_embeddings is not None
        if adanorm_num_embeddings:
            self.norm = modules.AdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                modules.ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=adanorm_num_embeddings,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self: VocosBackbone, m: nn.Module) -> None:
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(
        self: VocosBackbone,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, L), where B is the batch size,
                C denotes output features, and L is the sequence length.

        Returns:
            Tensor: Output of shape (B, L, H), where B is the batch size, L is the sequence length,
                and H denotes the model dimension.
        """
        bandwidth_id = kwargs.get('bandwidth_id', None)
        x = self.embed(x)
        if self.adanorm:
            assert bandwidth_id is not None
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, cond_embedding_id=bandwidth_id)
        x = self.final_layer_norm(x.transpose(1, 2))
        return x


class ISTFTHead(nn.Module):
    """
    ISTFT Head module for predicting STFT complex coefficients.

    Args:
        dim (int): Hidden dimension of the model.
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames, which should align with
            the resolution of the input features.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(
        self: ISTFTHead,
        dim: int,
        n_fft: int,
        hop_length: int,
        padding: str = "same",
    ) -> None:
        super().__init__()
        out_dim = n_fft + 2
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def forward(
        self: ISTFTHead,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the ISTFTHead module.

        Args:
            x (Tensor): Input tensor of shape (B, L, H), where B is the batch size,
                L is the sequence length, and H denotes the model dimension.

        Returns:
            Tensor: Reconstructed time-domain audio signal of shape (B, T), where T is the length of the output signal.
        """
        x = self.out(x).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        # recalculating phase here does not produce anything new
        # only costs time
        # phase = torch.atan2(y, x)
        # S = mag * torch.exp(phase * 1j)
        # better directly produce the complex value
        S = mag * (x + 1j * y)
        audio = self.istft(S)
        return audio


class VocosDecoder(torch.nn.Module):
    def __init__(
        self: VocosDecoder,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        isft_n_fft: int,
        isft_hop_length: int,
        isft_padding: str = "same",
        layer_scale_init_value: Optional[float] = None,
        adanorm_num_embeddings: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.backbone = VocosBackbone(
            input_channels,
            dim,
            intermediate_dim,
            num_layers,
            layer_scale_init_value=layer_scale_init_value,
            adanorm_num_embeddings=adanorm_num_embeddings,
        )
        self.head = ISTFTHead(
            dim,
            isft_n_fft,
            isft_hop_length,
            padding=isft_padding
        )

    def forward(
        self: VocosDecoder,
        features_input: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Method to decode audio waveform from already calculated features. The features input is passed through
        the backbone and the head to reconstruct the audio output.

        Args:
            features_input (Tensor): The input tensor of features of shape (B, C, L), where B is the batch size,
                C denotes the feature dimension, and L is the sequence length.

        Returns:
            Tensor: The output tensor representing the reconstructed audio waveform of shape (B, T).
        """
        x = self.backbone(features_input, **kwargs)
        audio_output = self.head(x)
        return audio_output
