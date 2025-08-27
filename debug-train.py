#!/usr/bin/env python3

from piper_train.__main__ import main as main_train
import argparse


args = argparse.Namespace(
    dataset_dir="/home/flavio/Dev/tts/training/",
    default_root_dir="",
    accelerator='cpu',
    devices=1,
    batch_size=16,
    validation_split=0.05,
    num_test_examples=0,
    max_epochs=2,
    checkpoint_epochs=1,
    precision=32,
    quality="medium",
    resume_from_single_speaker_checkpoint=False,
    hidden_channels=192,
    inter_channels=192,
    filter_channels=768,
    n_layers=6,
    n_heads=2,
    seed=1234,
)

main_train(args)
