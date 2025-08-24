from __future__ import annotations
import math
from typing import Optional, Tuple
# from dataclasses import dataclass

import torch
from torch import nn
from .encoders import TextEncoder, PosteriorEncoder
from .decoders import VitsDecoder
from .decoders import VocosDecoder
from .flow import InvertibleNormalizingFlow
from .duration_predictors import DurationPredictor, StochasticDurationPredictor


from . import commons, monotonic_align


# @dataclass
# class Text:
#     data: torch.Tensor
#     mask: Optional[torch.Tensor] = None
#     length: Optional[torch.Tensor] = None


# @dataclass
# class Audio:
#     data: torch.Tensor
#     mask: Optional[torch.Tensor] = None
#     length: Optional[torch.Tensor] = None


# @dataclass
# class LatentVariable:
#     data: torch.Tensor
#     mean: torch.Tensor
#     log_std_dev: torch.Tensor
#     mask: Optional[torch.Tensor] = None
#     length: Optional[torch.Tensor] = None


class VitsGenerator(nn.Module):
    """
    Vits Synthesizer/Generator
    """

    def __init__(
        self: VitsGenerator,
        n_vocab: int,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: Tuple[int, ...],
        resblock_dilation_sizes: Tuple[Tuple[int, ...], ...],
        upsample_rates: Tuple[int, ...],
        upsample_initial_channel: int,
        upsample_kernel_sizes: Tuple[int, ...],
        n_speakers: int = 1,
        gin_channels: int = 0,
        use_sdp: bool = True,
    ) -> None:

        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        self.use_sdp = use_sdp

        # Text encoder
        self.enc_p = TextEncoder(
            n_vocab,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        # Decoder
        # self.dec = VitsDecoder(
        #     inter_channels,
        #     resblock,
        #     resblock_kernel_sizes,
        #     resblock_dilation_sizes,
        #     upsample_rates,
        #     upsample_initial_channel,
        #     upsample_kernel_sizes,
        #     gin_channels=gin_channels,
        # )
        self.dec = VocosDecoder(
            input_channels=inter_channels,
            dim=384,
            intermediate_dim=1152,
            num_layers=6,
            isft_n_fft=1280,
            isft_hop_length=256,
            isft_padding="same",
        )
        # Posterior encoder (only needed for training)
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        # Normalizing flow (f_theta)
        self.flow = InvertibleNormalizingFlow(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels
        )
        # Duration predictor
        if use_sdp:
            self.dp = StochasticDurationPredictor(
                hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels
            )
        else:
            self.dp = DurationPredictor(
                hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
            )

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def encode_speaker(
        self: VitsGenerator,
        sid: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        g: Optional[torch.Tensor] = None
        if self.n_speakers > 1:
            # If multispeaker, retrieve speaker embeddings
            assert sid is not None, "Missing speaker id"
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        return g

    def forward(
        self: VitsGenerator,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        sid: Optional[int] = None,
    ) -> Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Tuple[torch.Tensor, ...]
        ]:
        """
            Forward step

            Args:
                x:          phoneme ID inputs (phonemized text)
                x_lengths:  phoneme input lengths
                y:          audio targets
                y_lengths:  audio target lengths
                sid:        speaker ID (if any)
        """
        # Encode phonemes
        # Outputs:
        #   x:      masked encoded input
        #   m_p:    mean for encoded inputs
        #   logs_p: log-standard deviation of encoded inputs
        #   x_mask: phoneme mask
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

        # Encode speaker
        g = self.encode_speaker(sid)

        # Encode targets
        # Outputs:
        #   z:      latent state variables
        #   m_q:    mean for latent variables
        #   logs_q: log-standard deviation of latent variables
        #   y_mask: audio mask
        z, m_q, logs_q, z_mask = self.enc_q(y, y_lengths, g=g)

        # Apply normalizing flow (forward transformation)
        z_p = self.flow(z, z_mask, g=g)

        with torch.no_grad():
            # Inverse of variance
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            # Gaussian density log of normalizing constant
            neg_cross_entropy1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
            )  # [b, 1, t_s]
            # Gaussian density exponent (quadratic term on z_p)
            neg_cross_entropy2 = torch.matmul(
                -0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            # Gaussian density exponent (cross term on z_p and m_p)
            neg_cross_entropy3 = torch.matmul(
                z_p.transpose(1, 2), (m_p * s_p_sq_r)
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            # Gaussian density exponent (quadratic term on m_p)
            neg_cross_entropy4 = torch.sum(
                -0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cross_entropy = neg_cross_entropy1 + neg_cross_entropy2 + neg_cross_entropy3 + neg_cross_entropy4
            # Attention mask
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(z_mask, -1)
            # Calculate alignment matrices (per batch)
            attn = (
                monotonic_align.maximum_path(neg_cross_entropy, attn_mask.squeeze(1))
                .unsqueeze(1)
                .detach()
            )
        # Sum of rows
        w = attn.sum(2)
        if self.use_sdp:
            # Stochastic duration prediction
            l_length = self.dp(x, x_mask, w, g=g)
            l_length = l_length / torch.sum(x_mask)
        else:
            # Deterministic duration prediction
            log_w_ = torch.log(w + 1e-6) * x_mask
            log_w = self.dp(x, x_mask, g=g)
            l_length = torch.sum((log_w - log_w_) ** 2, [1, 2]) / torch.sum(
                x_mask
            )  # for averaging

        # Expand prior values (upsample) according to alignment masks
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        # Get random slices of latent variables
        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        # Decode slices to produce predicted audio slices
        o = self.dec(z_slice, g=g)
        return (
            o,         # audio slices
            l_length,  # predicted durations
            attn,      # alignment matrices
            ids_slice, # slices indices
            x_mask,    # phoneme masks
            z_mask,    # latent variable masks
            (
                z,      # latent variables
                z_p,    # transformed latent variables
                m_p,    # phoneme-encoded mean
                logs_p, # phoneme-encoded log-standard-deviation
                m_q,    # latent variable mean
                logs_q, # latent variable log-standard-deviation
            ),
        )

    def infer(
        self: VitsGenerator,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        sid: Optional[int] = None,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
        noise_scale_w: float = 0.8,
        max_len: Optional[int] = None,
    ) -> Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Tuple[torch.Tensor, ...]
        ]:
        # Encode phonemes
        # Outputs:
        #   x:      masked encoded input
        #   m_p:    mean for encoded inputs
        #   logs_p: log-standard deviation of encoded inputs
        #   x_mask: phoneme mask
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
    
        # Encode speaker
        g = self.encode_speaker(sid)

        if self.use_sdp:
            # Stochastic duration likelihoods
            log_w = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            # Deterministic duration likelihoods
            log_w = self.dp(x, x_mask, g=g)
        # Duration prediction
        w = torch.exp(log_w) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        z_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        # Predicted masks
        z_mask = commons.sequence_mask(z_lengths, z_lengths.max()).unsqueeze(1).type_as(x_mask)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(z_mask, -1)
        # Predicted alignment matrices
        attn = commons.generate_path(w_ceil, attn_mask)

        # Expand prior (upsample)
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        # Sample in the normalizing flow space
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        # Apply inverse of normalizing flow (backward transformation)
        z = self.flow(z_p, z_mask, g=g, reverse=True)
        # Decode latent variables into audio
        o = self.dec((z * z_mask)[:, :, :max_len], g=g)

        return (
            o,          # predicted audio
            attn,       # predicted alignment matrices
            z_mask,     # latent variable mask
            (
                z,      # latent variables
                z_p,    # sampled transformed variable
                m_p,    # mean of transformed variable
                logs_p, # log-standard-deviation of transformed variable
            ),
        )

    def voice_conversion(
        self: VitsGenerator,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        sid_src: int,
        sid_tgt: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        assert self.n_speakers > 1, "n_speakers have to be larger than 1."
        g_src = self.encode_speaker(sid_src)
        g_tgt = self.encode_speaker(sid_tgt)
        z, _m_q, _logs_q, z_mask = self.enc_q(y, y_lengths, g=g_src)
        z_p = self.flow(z, z_mask, g=g_src)
        z_hat = self.flow(z_p, z_mask, g=g_tgt, reverse=True)
        o_hat = self.dec(z_hat * z_mask, g=g_tgt)
        return o_hat, z_mask, (z, z_p, z_hat)
