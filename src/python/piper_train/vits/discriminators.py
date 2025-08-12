from __future__ import annotations
from typing import List, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm
from .commons import get_padding
from .spectral_modules import mag_stft


class SpecDiscriminator(nn.Module):
    LRELU_SLOPE: float = 0.1

    def __init__(
        self: SpecDiscriminator,
        fft_size: int = 1024,
        shift_size: int = 120,
        win_length: int = 600,
        window: str = "hann_window",
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()

        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.discriminators = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1,2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1,2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1,2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1,1), padding=(1, 1))),
        ])

        self.out = norm_f(nn.Conv2d(32, 1, 3, 1, 1))

    def forward(self: SpecDiscriminator, y: torch.Tensor) -> torch.Tensor:
        fmap = []
        y = y.squeeze(1)

        y = mag_stft(y, self.fft_size, self.shift_size, self.win_length, self.window.to(y.get_device()))
        y = y.unsqueeze(1)

        for _i, d in enumerate(self.discriminators):
            y = d(y)
            y = F.leaky_relu(y, self.LRELU_SLOPE)
            fmap.append(y)

        y = self.out(y)
        fmap.append(y)

        return torch.flatten(y, 1, -1), fmap


class MultiResSpecDiscriminator(nn.Module):

    def __init__(
        self: MultiResSpecDiscriminator,
        fft_sizes: Union[Tuple[int, ...], List[int]] = [1024, 2048, 512],
        hop_sizes: Union[Tuple[int, ...], List[int]] = [120, 240, 50],
        win_lengths: Union[Tuple[int, ...], List[int]] = [600, 1200, 240],
        window: str = "hann_window",
    ) -> None:
        super().__init__()

        self.discriminators = nn.ModuleList([
            SpecDiscriminator(fft_sizes[0], hop_sizes[0], win_lengths[0], window),
            SpecDiscriminator(fft_sizes[1], hop_sizes[1], win_lengths[1], window),
            SpecDiscriminator(fft_sizes[2], hop_sizes[2], win_lengths[2], window)
            ])

    def forward(
        self: MultiResSpecDiscriminator,
        y: torch.Tensor,
        y_hat: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], ...]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for _i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorP(nn.Module):
    LRELU_SLOPE: float = 0.1

    def __init__(
        self: DiscriminatorP,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()

        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    nn.Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(
        self: DiscriminatorP,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(nn.Module):
    LRELU_SLOPE: float = 0.1

    def __init__(self: DiscriminatorS, use_spectral_norm: bool = False) -> None:
        super(DiscriminatorS, self).__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self: DiscriminatorS, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    PERIODS: List[int] = [2, 3, 5, 7, 11]

    def __init__(self: MultiPeriodDiscriminator, use_spectral_norm: bool = False, simplified: bool = True) -> None:
        super().__init__()

        modules = [] if simplified else [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        modules += [
            DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in self.PERIODS
        ]
        self.discriminators = nn.ModuleList(modules)

    def forward(
        self: MultiPeriodDiscriminator,
        y: torch.Tensor,
        y_hat: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], ...]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for _i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class WavLMDiscriminator(nn.Module):
    """WavLM-based Discriminator."""
    LRELU_SLOPE: float = 0.1

    def __init__(
        self: WavLMDiscriminator,
        slm_hidden: int = 768,
        slm_layers: int = 13,
        initial_channel: int = 64,
        use_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.pre = norm_f(nn.Conv1d(slm_hidden * slm_layers, initial_channel, 1, 1, padding=0))

        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(initial_channel, initial_channel * 2, kernel_size=5, padding=2)),
                norm_f(nn.Conv1d(initial_channel * 2, initial_channel * 4, kernel_size=5, padding=2)),
                norm_f(nn.Conv1d(initial_channel * 4, initial_channel * 4, 5, 1, padding=2)),
            ]
        )

        self.conv_post = norm_f(nn.Conv1d(initial_channel * 4, 1, 3, 1, padding=1))

    def forward(self: WavLMDiscriminator, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)

        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)

        return x
