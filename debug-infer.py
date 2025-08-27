#!/usr/bin/env python3

from piper.__main__ import main as main_infer
import argparse


args = argparse.Namespace(
    model="/home/flavio/Dev/tts/checkpoint/epoch-9-vocos-decoder-860.ckpt",
    config="/home/flavio/Dev/tts/checkpoint/config.json",
    input="Era uma vez um menino, livre, levado e contente.",
    output_file="/home/flavio/Dev/tts/checkpoint/output.wav",
    debug=False,
    download_dir="/home/flavio/Dev/tts/checkpoint",
    data_dir=["/home/flavio/Dev/tts/checkpoint"],
    update_voices=False,
    noise_scale=0.667,
    noise_w=0.8,
    length_scale=1.0,
    sentence_silence=0.0,
    output_raw=False,
    output_dir=None,
    cuda=False,
    speaker=None,
)

main_infer(args)
