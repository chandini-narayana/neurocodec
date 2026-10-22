#!/usr/bin/env python3
# Copyright (c) 2026 Chandini Narayana
# SPDX-License-Identifier: MIT
"""
Export the ConvLSTM decoder to ONNX (single step with explicit recurrent state).

The exported graph takes ``voxel`` plus the ConvLSTM states ``h0, c0, h1, c1, ...``
and returns ``frame`` plus the updated states, so the C++ runtime can decode a
clip frame by frame exactly like the PyTorch path (see ``DecoderStep``).

Usage:
    python eval/export_onnx.py --weights neurocodec/weights/decoder_v1.pt \
                                --output  decoder.onnx \
                                --height  240 --width 320

Then use with the C++ runtime:
    cd cpp_runtime && mkdir build && cd build
    cmake .. && cmake --build . --config Release
    ./nc_decode --model ../../decoder.onnx --input clip.nec --benchmark
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def export(
    weights_path: str | None,
    output_path: str,
    n_bins: int = 5,
    height: int = 240,
    width:  int = 320,
    opset:  int = 17,
    allow_random: bool = False,
) -> None:
    from neurocodec.decoder import DecoderStep, export_onnx, load_decoder

    model = load_decoder(weights_path, n_bins=n_bins, device=torch.device("cpu"),
                         require_weights=not allow_random)
    logger.info("Exporting to ONNX (opset %d) ...", opset)
    in_names = export_onnx(model, output_path, height=height, width=width, opset=opset)
    logger.info("Saved: %s (inputs: %s)", output_path, ", ".join(in_names))

    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed - skipping verification. "
                       "Install: pip install onnxruntime")
        return

    # Verify: run 3 frames through PyTorch (stateful) and ONNX (explicit state).
    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    step = DecoderStep(model)
    rng = np.random.default_rng(0)
    state = {n: np.zeros((1, ch, height, width), np.float32)
             for i, ch in enumerate(step.state_channels) for n in (f"h{i}", f"c{i}")}
    out_names = [o.name for o in sess.get_outputs()]
    model.reset_state()
    max_err = 0.0
    with torch.inference_mode():
        for _ in range(3):
            v = rng.standard_normal((1, n_bins, height, width)).astype(np.float32)
            ref = model(torch.from_numpy(v)).numpy()
            outs = sess.run(out_names, {"voxel": v, **state})
            max_err = max(max_err, float(np.abs(outs[0] - ref).max()))
            state = dict(zip(in_names[1:], outs[1:]))
    logger.info("Verification: max |ONNX - PyTorch| over 3 stateful frames = %.2e", max_err)
    if max_err > 1e-3:
        raise SystemExit("ONNX output does not match PyTorch")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Export ConvLSTM decoder to ONNX")
    p.add_argument("--weights", default=None, help="Path to decoder_v1.pt")
    p.add_argument("--output",  default="decoder.onnx",
                   help="Output ONNX file (default: decoder.onnx)")
    p.add_argument("--n-bins",  type=int, default=5,
                   help="Voxel time bins (must match training, default 5)")
    p.add_argument("--height",  type=int, default=240,
                   help="Dummy input height for tracing (default 240)")
    p.add_argument("--width",   type=int, default=320,
                   help="Dummy input width for tracing (default 320)")
    p.add_argument("--opset",   type=int, default=17,
                   help="ONNX opset version (default 17)")
    p.add_argument("--allow-random", action="store_true",
                   help="Export an untrained model (for testing the toolchain)")
    args = p.parse_args()

    export(args.weights, args.output, n_bins=args.n_bins, height=args.height,
           width=args.width, opset=args.opset, allow_random=args.allow_random)


if __name__ == "__main__":
    main()
