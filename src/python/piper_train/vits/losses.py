from __future__ import annotations
from typing import List, Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F
import torchaudio
from transformers import AutoModel
from .discriminators import (
    MultiPeriodDiscriminator,
    MultiResSpecDiscriminator,
    WavLMDiscriminator,
)


def feature_loss(
    fmap_r: List[torch.Tensor],
    fmap_g: List[torch.Tensor],
) -> torch.Tensor:
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(
    disc_real_outputs: List[torch.Tensor],
    disc_generated_outputs: List[torch.Tensor],
) -> Tuple[torch.Tensor, ...]:
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1.0 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l_dg = torch.mean((1.0 - dg) ** 2)
        gen_losses.append(l_dg)
        loss += l_dg

    return loss, gen_losses


""" https://dl.acm.org/doi/abs/10.1145/3573834.3574506 """
def discriminator_tprls_loss(
    disc_real_outputs: List[torch.Tensor],
    disc_generated_outputs: List[torch.Tensor],
) -> torch.Tensor:
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_DG = torch.median((dr-dg))
        L_rel = torch.mean((((dr - dg) - m_DG) ** 2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss


def generator_tprls_loss(
    disc_real_outputs: List[torch.Tensor],
    disc_generated_outputs: List[torch.Tensor],
) -> torch.Tensor:
    loss = 0
    for dg, dr in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_DG = torch.median((dr - dg))
        L_rel = torch.mean((((dr - dg) - m_DG) ** 2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss


def kl_loss(
    z_p: torch.Tensor,
    logs_q: torch.Tensor,
    m_p: torch.Tensor,
    logs_p: torch.Tensor,
    z_mask: torch.Tensor,
) -> torch.Tensor:
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l_kl = kl / torch.sum(z_mask)
    return l_kl


class GeneratorLoss(nn.Module):

    def __init__(
        self: GeneratorLoss,
        mpd: MultiPeriodDiscriminator,
        msd: Optional[MultiResSpecDiscriminator] = None,
    ) -> None:
        super().__init__()
        self.mpd = mpd
        self.msd = msd

    def forward(self: GeneratorLoss, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        # MPD
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_gen_f, _losses_gen_f = generator_loss(y_df_hat_g)
        loss_rel = generator_tprls_loss(y_df_hat_r, y_df_hat_g)
        loss_gen_all = loss_gen_f + loss_fm_f

        # MSD
        if self.msd is not None:
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_hat)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_s, _losses_gen_s = generator_loss(y_ds_hat_g)
            loss_rel += generator_tprls_loss(y_ds_hat_r, y_ds_hat_g)
            loss_gen_all += loss_gen_s + loss_fm_s

        loss_gen_all += loss_rel

        return loss_gen_all.mean()


class DiscriminatorLoss(torch.nn.Module):

    def __init__(
        self: DiscriminatorLoss,
        mpd: MultiPeriodDiscriminator,
        msd: Optional[MultiResSpecDiscriminator] = None,
    ) -> None:
        super().__init__()
        self.mpd = mpd
        self.msd = msd
        
    def forward(self: DiscriminatorLoss, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_hat)
        loss_disc_f, _losses_disc_f_r, _losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
        loss_rel = discriminator_tprls_loss(y_df_hat_r, y_df_hat_g)
        d_loss = loss_disc_f

        # MSD
        if self.msd is not None:
            y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_hat)
            loss_disc_s, _losses_disc_s_r, _losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
            loss_rel += discriminator_tprls_loss(y_ds_hat_r, y_ds_hat_g)
            d_loss += loss_disc_s

        d_loss += loss_rel

        return d_loss.mean()


class WavLMLoss(nn.Module):

    def __init__(
        self: WavLMLoss,
        model: str,
        wd: WavLMDiscriminator,
        model_sr: int,
        slm_sr: int = 16000,
    ) -> None:
        super().__init__()
        self.wavlm = AutoModel.from_pretrained(model)
        self.wd = wd
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)

    def forward(self: WavLMLoss, wav: torch.Tensor, y_rec: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(input_values=wav_16, output_hidden_states=True).hidden_states
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(input_values=y_rec_16.squeeze(), output_hidden_states=True).hidden_states

        floss = 0.0
        for er, eg in zip(wav_embeddings, y_rec_embeddings):
            floss += torch.mean(torch.abs(er - eg))

        return floss.mean()

    def generator(self: WavLMLoss, y_rec: torch.Tensor) -> torch.Tensor:
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(input_values=y_rec_16, output_hidden_states=True).hidden_states
        y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
        y_df_hat_g = self.wd(y_rec_embeddings)
        loss_gen = torch.mean((1.0 - y_df_hat_g) ** 2)

        return loss_gen

    def discriminator(self: WavLMLoss, wav: torch.Tensor, y_rec: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(input_values=wav_16, output_hidden_states=True).hidden_states
            y_rec_16 = self.resample(y_rec)
            y_rec_embeddings = self.wavlm(input_values=y_rec_16, output_hidden_states=True).hidden_states

            y_embeddings = torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
            y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

        y_d_rs = self.wd(y_embeddings)
        y_d_gs = self.wd(y_rec_embeddings)

        y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs

        r_loss = torch.mean((1.0 - y_df_hat_r) ** 2)
        g_loss = torch.mean((y_df_hat_g) ** 2)
        loss_disc_f = r_loss + g_loss

        return loss_disc_f.mean()

    def discriminator_forward(self: WavLMLoss, wav: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(input_values=wav_16, output_hidden_states=True).hidden_states
            y_embeddings = torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

        y_d_rs = self.wd(y_embeddings)

        return y_d_rs
