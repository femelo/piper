from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
from argparse import ArgumentParser

import lightning as L
from lightning.fabric import Fabric
from wandb.integration.lightning.fabric import WandbLogger
import torch
from torch import nn
from torch import autocast
from torch.utils.data import DataLoader, Dataset, random_split

from .commons import slice_segments
from .dataset import Batch, PiperDataset, UtteranceCollate

from .losses import (
    MultiResolutionSTFTLoss,
    DiscriminatorLoss,
    GeneratorLoss,
    WavLMLoss,
    kl_loss,
)

from .discriminators import (
    MultiPeriodDiscriminator,
    MultiResSpecDiscriminator,
    WavLMDiscriminator,
)
from .models import SynthesizerTrn
from .config import SLMModelConfig


_LOGGER = logging.getLogger("vits.lightning_model")


class VitsModel(L.LightningModule):
    def __init__(
        self: VitsModel,
        num_symbols: int,
        num_speakers: int,
        *,
        # run_id
        run_id: Optional[str] = None,
        # audio
        resblock: str = "2",
        resblock_kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        resblock_dilation_sizes: Tuple[Tuple[int, ...], ...] = (
            (1, 2),
            (2, 6),
            (3, 12),
        ),
        upsample_rates: Tuple[int, ...] = (8, 8, 4),
        upsample_initial_channel: int = 256,
        upsample_kernel_sizes: Tuple[int, ...] = (16, 16, 8),
        # mel
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        mel_channels: int = 80,
        sample_rate: int = 22050,
        sample_bytes: int = 2,
        channels: int = 1,
        mel_fmin: float = 0.0,
        mel_fmax: Optional[float] = None,
        # model
        inter_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        n_layers_q: int = 3,
        use_spectral_norm: bool = False,
        gin_channels: int = 0,
        use_sdp: bool = True,
        segment_size: int = 8192,
        # training
        dataset: Optional[List[Union[str, Path]]] = None,
        learning_rate: float = 2e-4,
        betas: Tuple[float, float] = (0.8, 0.99),
        eps: float = 1e-9,
        batch_size: int = 1,
        lr_decay: float = 0.999875,
        init_lr_ratio: float = 1.0,
        warmup_epochs: int = 0,
        slm: SLMModelConfig = SLMModelConfig(),
        c_dur: float = 1.0,
        c_mel: float = 5.0,
        c_kl: float = 1.0,
        c_gen: float = 1.0,
        c_dsc: float = 1.0,
        c_slm: float = 1.0,
        grad_clip: Optional[float] = None,
        num_workers: int = 1,
        seed: int = 1234,
        num_test_examples: int = 5,
        validation_split: float = 0.1,
        max_phoneme_ids: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.wnb_logger: Optional[WandbLogger] = WandbLogger(project=run_id) if run_id else None
        self.fabric: Optional[Fabric] = Fabric(loggers=self.wnb_logger) if run_id else None

        if (self.hparams.num_speakers > 1) and (self.hparams.gin_channels <= 0):
            # Default gin_channels for multi-speaker model
            self.hparams.gin_channels = 512

        # Set up models
        # Generator (only needed for training)
        self.model_g = SynthesizerTrn(
            n_vocab=self.hparams.num_symbols,
            spec_channels=self.hparams.filter_length // 2 + 1,
            segment_size=self.hparams.segment_size // self.hparams.hop_length,
            inter_channels=self.hparams.inter_channels,
            hidden_channels=self.hparams.hidden_channels,
            filter_channels=self.hparams.filter_channels,
            n_heads=self.hparams.n_heads,
            n_layers=self.hparams.n_layers,
            kernel_size=self.hparams.kernel_size,
            p_dropout=self.hparams.p_dropout,
            resblock=self.hparams.resblock,
            resblock_kernel_sizes=self.hparams.resblock_kernel_sizes,
            resblock_dilation_sizes=self.hparams.resblock_dilation_sizes,
            upsample_rates=self.hparams.upsample_rates,
            upsample_initial_channel=self.hparams.upsample_initial_channel,
            upsample_kernel_sizes=self.hparams.upsample_kernel_sizes,
            n_speakers=self.hparams.num_speakers,
            gin_channels=self.hparams.gin_channels,
            use_sdp=self.hparams.use_sdp,
        )

        # Discriminators
        self.mpd = MultiPeriodDiscriminator(
            use_spectral_norm=self.hparams.use_spectral_norm
        ).to(self.device)
        self.msd = MultiResSpecDiscriminator().to(self.device)
        self.wd = WavLMDiscriminator(
            use_spectral_norm=self.hparams.use_spectral_norm
        ).to(self.device)
        # Losses wrappers
        self.stft_loss = MultiResolutionSTFTLoss().to(self.device)
        self.discriminator_loss = DiscriminatorLoss(
            mpd=self.mpd,
            msd=self.msd,
        ).to(self.device)
        self.generator_loss = GeneratorLoss(
            mpd=self.mpd,
            msd=self.msd,
        ).to(self.device)
        self.wavlm_loss = WavLMLoss(
            model=self.hparams.slm.model,
            wd=self.wd,
            model_sr=self.hparams.sample_rate,
            slm_sr=self.hparams.slm.sr,
        ).to(self.device)
        # Generation group
        self.gen_group = nn.ModuleList([self.model_g, self.mpd, self.msd, self.wd])
        # Discrimination group
        self.dsc_group = nn.ModuleList([self.mpd, self.msd, self.wd])

        # Dataset splits
        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None
        self._load_datasets(validation_split, num_test_examples, max_phoneme_ids)

        # State kept between training optimizers
        self._y = None
        self._y_hat = None
        self._y_embed = None
        self._y_hat_embed = None

    def _load_datasets(
        self: VitsModel,
        validation_split: float,
        num_test_examples: int,
        max_phoneme_ids: Optional[int] = None,
    ) -> None:
        if self.hparams.dataset is None:
            _LOGGER.debug("No dataset to load")
            return

        full_dataset = PiperDataset(
            self.hparams.dataset, max_phoneme_ids=max_phoneme_ids
        )
        valid_set_size = int(len(full_dataset) * validation_split)
        train_set_size = len(full_dataset) - valid_set_size - num_test_examples

        self._train_dataset, self._test_dataset, self._val_dataset = random_split(
            full_dataset, [train_set_size, num_test_examples, valid_set_size]
        )

    def forward(
        self: VitsModel,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        scales: Union[Tuple[float, ...], List[float]],
        sid: Optional[int] = None,
    ) -> torch.Tensor:
        noise_scale = scales[0]     # normalizing flow noise scale
        length_scale = scales[1]    # duration scale
        noise_scale_w = scales[2]   # duration noise scale
        audio, *_ = self.model_g.infer(
            text,
            text_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sid=sid,
        )

        return audio

    def train_dataloader(self: VitsModel) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            collate_fn=UtteranceCollate(
                is_multispeaker=self.hparams.num_speakers > 1,
                segment_size=self.hparams.segment_size,
            ),
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
        )

    def val_dataloader(self: VitsModel) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            collate_fn=UtteranceCollate(
                is_multispeaker=self.hparams.num_speakers > 1,
                segment_size=self.hparams.segment_size,
            ),
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
        )

    def test_dataloader(self: VitsModel) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            collate_fn=UtteranceCollate(
                is_multispeaker=self.hparams.num_speakers > 1,
                segment_size=self.hparams.segment_size,
            ),
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
        )

    def training_step(self: VitsModel, batch: Batch, batch_idx: int) -> torch.Tensor:
        optimizer_g, optimizer_d = self.optimizers()

        self.toggle_optimizer(optimizer_g)
        loss_gen = self.training_step_g(batch)
        self.log("loss_gen_all", loss_gen, prog_bar=True)
        self.manual_backward(loss_gen)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        self.toggle_optimizer(optimizer_d)
        loss_dsc = self.training_step_d(batch)
        self.log("loss_dsc_all", loss_dsc, prog_bar=True)
        self.manual_backward(loss_dsc)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def training_step_g(self: VitsModel, batch: Batch) -> torch.Tensor:
        """
            Generation training step

            Variables:
                x:    phoneme-related inputs
                y:    audio-related targets
                spec: spectrogram-related targets from audio targets
        """
        x, x_lengths, y, _, spec, spec_lengths, speaker_ids = (
            batch.phoneme_ids,
            batch.phoneme_lengths,
            batch.audios,
            batch.audio_lengths,
            batch.spectrograms,
            batch.spectrogram_lengths,
            batch.speaker_ids if batch.speaker_ids is not None else None,
        )
        # Forward generation
        (
            y_hat,
            l_length,
            _attn,
            ids_slice,
            _x_mask,
            z_mask,
            (_z, z_p, m_p, logs_p, _m_q, logs_q),
        ) = self.model_g(x, x_lengths, spec, spec_lengths, speaker_ids)

        self._y_hat = y_hat

        # Collect predicted Mel spectrogram segments
        y = slice_segments(
            y,
            ids_slice * self.hparams.hop_length,
            self.hparams.segment_size,
        )  # slice

        # Save for discriminator training step (training_step_d)
        self._y = y

        with torch.no_grad():
            y_embed = self.wavlm_loss.encode(y.detach())
        y_hat_embed = self.wavlm_loss.encode(y_hat)
        self._y_embed = y_embed
        self._y_hat_embed = y_hat_embed

        with autocast(self.device.type, enabled=False):
            # Duration loss
            loss_dur = torch.sum(l_length.float())
            # Likelihood loss (Mel spectrogram multi-resolution STFT loss)
            loss_mel = self.stft_loss(y.detach().squeeze(), y_hat.squeeze()).mean()
            # KL-divergence loss
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)
            # SLM loss
            loss_slm = self.wavlm_loss.loss_from_embeddings(y_embed, y_hat_embed).mean()
            # SLM generation loss
            loss_gen_slm = self.wavlm_loss.generator_from_embeddings(y_hat_embed).mean()
            # Generator loss
            loss_gen = self.generator_loss(y.detach(), y_hat).mean()
            # Total generation loss
            loss_gen_all = \
                self.hparams.c_gen * loss_gen + \
                self.hparams.c_slm * (loss_slm + loss_gen_slm) + \
                self.hparams.c_mel * loss_mel + \
                self.hparams.c_dur * loss_dur + \
                self.hparams.c_kl * loss_kl

            if self.fabric is not None:
                self.fabric.log_dict(
                    {
                        "loss_gen": loss_gen,
                        "loss_gen_slm": loss_gen_slm,
                        "loss_slm": loss_slm,
                        "loss_mel": loss_mel,
                        "loss_dur": loss_dur,
                        "loss_kl": loss_kl,
                        "loss_gen_all": loss_gen_all,
                    }
                )

            return loss_gen_all

    def training_step_d(self: VitsModel, batch: Batch) -> torch.Tensor:
        # Discrimination training step
        # From training_step_g
        y = self._y.detach()
        y_hat = self._y_hat.detach()
        y_embed = [e.detach() for e in self._y_embed]
        y_hat_embed = [e.detach() for e in self._y_hat_embed]

        with autocast(self.device.type, enabled=False):
            # Discrimination adversarial loss
            loss_dsc = self.discriminator_loss(y, y_hat).mean()
            # SLM loss
            loss_dsc_slm = self.wavlm_loss.discriminator_from_embeddings(y_embed, y_hat_embed).mean()
            # Total discrimination loss
            loss_dsc_all = self.hparams.c_dsc * loss_dsc + self.hparams.c_slm * loss_dsc_slm

            if self.fabric is not None:
                self.fabric.log_dict(
                    {
                        "loss_dsc": loss_dsc,
                        "loss_dsc_slm": loss_dsc_slm,
                        "loss_dsc_all": loss_dsc_all,
                    }
                )

            return loss_dsc_all

    def validation_step(self: VitsModel, batch: Batch, batch_idx: int) -> torch.Tensor:
        with torch.no_grad():
            loss_gen = self.training_step_g(batch)
            loss_dsc = self.training_step_d(batch)
        val_loss = loss_gen + loss_dsc
        self.log("val_loss", val_loss)

        # Generate audio examples
        for utt_idx, test_utt in enumerate(self._test_dataset):
            text = test_utt.phoneme_ids.unsqueeze(0).to(self.device)
            text_lengths = torch.LongTensor([len(test_utt.phoneme_ids)]).to(self.device)
            scales = [
                0.667,  # normalizing flow noise scale
                1.0,    # duration scale
                0.8,    # duration noise scale
            ]
            sid = (
                test_utt.speaker_id.to(self.device)
                if test_utt.speaker_id is not None
                else None
            )
            test_audio = self(text, text_lengths, scales, sid=sid).detach()

            # Scale to make louder in [-1, 1]
            test_audio = test_audio * (1.0 / max(0.01, abs(test_audio.max())))

            tag = test_utt.text or str(utt_idx)
            self.logger.experiment.add_audio(
                tag, test_audio, sample_rate=self.hparams.sample_rate
            )

        return val_loss

    def configure_optimizers(
        self: VitsModel
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LRScheduler]]:
        optimizers = [
            torch.optim.AdamW(
                group.parameters(),
                lr=self.hparams.learning_rate,
                betas=self.hparams.betas,
                eps=self.hparams.eps,
            ) for group in (self.gen_group, self.dsc_group)
        ]
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.hparams.lr_decay
            ) for optimizer in optimizers
        ]
        return optimizers, schedulers

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = parent_parser.add_argument_group("VitsModel")
        parser.add_argument("--batch-size", type=int, required=True)
        parser.add_argument("--validation-split", type=float, default=0.1)
        parser.add_argument("--num-test-examples", type=int, default=5)
        parser.add_argument(
            "--max-phoneme-ids",
            type=int,
            help="Exclude utterances with phoneme id lists longer than this",
        )
        #
        parser.add_argument("--hidden-channels", type=int, default=192)
        parser.add_argument("--inter-channels", type=int, default=192)
        parser.add_argument("--filter-channels", type=int, default=768)
        parser.add_argument("--n-layers", type=int, default=6)
        parser.add_argument("--n-heads", type=int, default=2)
        #
        return parent_parser
