#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch

from .vits.lightning_model import VitsModel

_LOGGER = logging.getLogger("piper_train.export_onnx")

# OPSET_VERSION = 15


def main() -> None:
    """Main entry point"""
    torch.manual_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to model checkpoint (.ckpt)")
    parser.add_argument("output", help="Path to output model (.onnx)")

    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    # -------------------------------------------------------------------------

    args.checkpoint = Path(args.checkpoint)
    args.output = Path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = VitsModel.load_from_checkpoint(
        args.checkpoint,
        map_location=torch.device("cpu"),
        dataset=None,
    )
    model_g = model.model_g

    num_symbols = model_g.n_vocab
    num_speakers = model_g.n_speakers

    # Inference only
    model_g.eval()

    with torch.no_grad():
        model_g.dec.remove_weight_norm()

    # old_forward = model_g.infer

    def infer_forward(
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        scales: torch.Tensor,
        sid: Optional[int] = None,
    ) -> torch.Tensor:
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]

        audio = model_g.infer(
            text,
            text_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sid=sid,
        )[0].unsqueeze(1)

        return audio

    model_g.forward = infer_forward

    dummy_input_length = 122
    sequences = torch.randint(
        low=0, high=num_symbols, size=(1, dummy_input_length), dtype=torch.long
    )
    sequence_lengths = torch.LongTensor([sequences.size(1)])

    sid: Optional[torch.LongTensor] = None
    if num_speakers > 1:
        sid = torch.LongTensor([0])

    # noise, length, noise_w
    scales = torch.FloatTensor([0.667, 1.0, 0.8])
    dummy_input = (sequences, sequence_lengths, scales, sid)

    batch_size = torch.export.Dim("batch_size", min=1)
    phone_seq_len = torch.export.Dim("phon_len", min=1)
    scale_size = torch.export.Dim("scale_size", min=3)
    # time_seq_len = torch.export.Dim("time_seq_len", min=1)

    # Export
    torch.onnx.export(
        model=model_g,
        args=dummy_input,
        f=str(args.output),
        verbose=False,
        # opset_version=OPSET_VERSION,
        input_names=["text", "text_lengths", "scales", "sid"],
        output_names=["output"],
        dynamic_shapes={
            "text": (batch_size, phone_seq_len),
            "text_lengths": (batch_size, ),
            "scales": (scale_size, ),
            "sid": None,
            # "output": (batch_size, time_seq_len),
        },
        dynamo=True,
        # report=True,
    )

    _LOGGER.info("Exported model to %s", args.output)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
