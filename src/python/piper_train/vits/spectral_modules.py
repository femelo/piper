from __future__ import annotations
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F


def mag_stft(
    x: torch.Tensor,
    fft_size: int,
    hop_size: int,
    win_length: int,
    window: str,
) -> torch.Tensor:
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(
        x, fft_size, hop_size, win_length, window, return_complex=True
    )
    # real = x_stft[..., 0]
    # imag = x_stft[..., 1]

    return torch.abs(x_stft).transpose(2, 1)


class ISTFT(nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self: ISTFT, n_fft: int, hop_length: int, win_length: int, padding: str = "same") -> None:
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        arg_matrix = (2.0 * torch.pi / n_fft) * (
            torch.arange(n_fft, dtype=torch.float32)[:, None] * torch.arange(n_fft, dtype=torch.float32)[None, :]
        )
        cos_matrix = torch.cos(arg_matrix)
        sin_matrix = torch.sin(arg_matrix)
        self.register_buffer("cos_matrix", cos_matrix)
        self.register_buffer("sin_matrix", sin_matrix)
        self.register_buffer("window", window)


    def istft(self: ISTFT, y_real: torch.Tensor, y_imag: torch.Tensor) -> torch.Tensor:
        assert y_real.shape == y_imag.shape, "Real and imaginary parts must have the same shape"
        assert self.win_length == self.n_fft
        B, _N, T = y_real.shape[-1]

        device = y_real.device
        istft_window = self.window.to(device).view(1, -1)

        output_size =  self.n_fft + self.hop_length * (T - 1)

        x_hat = torch.zeros(B, output_size, device=device)
        for i in range(T):
            sample = i * self.hop_length
            y_real_ = y_real[:, :, i]
            y_imag_ = y_imag[:, :, i]
            x_ifft = self.irfft(y_real_, y_imag_, dim=-1)  # [B, n_fft]
            x_win = istft_window *  x_ifft
            x_hat[:, sample:(sample + self.n_fft)] += x_win

        # x_hat = x_hat[:, (self.n_fft // 2):]
        # coeff = self.n_fft / float(self.hop_length) / 2.0
        coeff = self.n_fft / float(self.hop_length)
        return x_hat / coeff

    def irfft(self: ISTFT, y_real: torch.Tensor, y_imag: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert y_real.shape == y_imag.shape, "Real and imaginary parts must have the same shape"
        if dim != -1:
            y_real = y_real.swapaxes(-1, dim)
            y_imag = y_imag.swapaxes(-1, dim)
        z_real, z_imag = self.extend_input(y_real, y_imag)
        x = (
            torch.matmul(self.cos_matrix, z_real.swapaxes(1, -1))
            - torch.matmul(self.sin_matrix, z_imag.swapaxes(1, -1))
        ) / self.n_fft
        x = x.swapaxes(1, -1)
        if dim != -1:
            x = x.swapaxes(-1, dim)
        return x

    def extend_input(
        self: ISTFT,
        y_real: torch.Tensor,
        y_imag: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert y_real.shape == y_imag.shape, "Real and imaginary parts must have the same shape"
        shape = y_real.shape
        n = shape[-1]
        m = 2 * n - 2
        r = self.n_fft - m
        real = []
        imag = []
        if r > 0:
            i = n
            j = n
            s = r - 1
        else:
            i = n + (r // 2)
            j = self.n_fft - i + 1
            s = 0
        real.append(y_real[..., :i])
        imag.append(y_imag[..., :i])
        if s > 0:
            real.append(torch.zeros((*shape[:-1], s)))
            imag.append(torch.zeros((*shape[:-1], s)))
        real.append(y_real[..., 1:j].flip(-1))
        imag.append(-y_imag[..., 1:j].flip(-1))
        y_real_full = torch.cat(real, dim=-1)
        y_imag_full = torch.cat(imag, dim=-1)
        return y_real_full, y_imag_full


    def forward(self: ISTFT, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T, 2), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames,
                            and the last dimension contains real and imaginary parts.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if self.padding == "center":
            # Fallback to pytorch native implementation
            # return torch.istft(spec, self.n_fft, self.hop_length, self.win_length, self.window, center=True)
            return self.istft(spec[..., 0], spec[..., 1])
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 4, "Expected a 4D tensor as input"
        _B, _N, T, _ = spec.shape

        # Inverse FFT
        # ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        x_ifft = self.irfft(spec[..., 0], spec[..., 1], dim=1) # [B, n_fft, T]
        x_win = x_ifft * self.window.view(1, -1, 1)

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        x_hat = F.fold(
            x_win,
            output_size=(1, int(output_size)),
            kernel_size=(1, int(self.win_length)),
            stride=(1, int(self.hop_length)),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = F.fold(
            window_sq,
            output_size=(1, int(output_size)),
            kernel_size=(1, int(self.win_length)),
            stride=(1, int(self.hop_length)),
        ).squeeze()[pad:-pad]

        # Normalize
        # FIXME: changed to allow onnx export
        # assert (window_envelope > 1e-11).all()
        # window_envelope = window_envelope.maximum(
        #     torch.Tensor([1e-11]).to(window_envelope.device)
        # )
        x_hat = x_hat / window_envelope

        return x_hat
