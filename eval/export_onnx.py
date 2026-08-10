#!/usr/bin/env python3
# Copyright (c) 2026 Chandini Narayana
# SPDX-License-Identifier: MIT
"""
Export ConvLSTM decoder to ONNX format.

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
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def export(
    weights_path: str | None,
    output_path: str,
    n_bins: int = 5,
    height: int = 240,
    width:  int = 320,
    opset:  int = 17,
) -> None:
    from neurocodec.decoder import ConvLSTMDecoder, load_decoder

    device = torch.device("cpu")
    model  = load_decoder(weights_path, n_bins=n_bins, device=device)
    model.eval()
    model.reset_state()

    dummy = torch.zeros(1, n_bins, height, width)

    logger.info("Exporting to ONNX (opset %d) ...", opset)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["voxel"],
        output_names=["frame"],
        dynamic_axes={
            "voxel": {0: "batch", 2: "height", 3: "width"},
            "frame": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=opset,
        verbose=False,
    )
    logger.info("Saved: %s", output_path)

    # Verify
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path)
        out  = sess.run(["frame"], {"voxel": dummy.numpy()})
        logger.info("Verification OK — output shape: %s", out[0].shape)
    except ImportError:
        logger.warning("onnxruntime not installed — skipping verification.")
        logger.warning("Install: pip install onnxruntime")

    # Measure PyTorch vs ONNX latency
    try:
        import time, onnxruntime as ort
        sess = ort.InferenceSession(output_path)
        inp  = dummy.numpy()

        # Warm-up
        for _ in range(5):
            with torch.no_grad():
                model(dummy)
            sess.run(["frame"], {"voxel": inp})

        N = 50
        t0 = time.perf_counter()
        for _ in range(N):
            with torch.no_grad():
                model(dummy)
        pt_ms = (time.perf_counter() - t0) / N * 1000

        t0 = time.perf_counter()
        for _ in range(N):
            sess.run(["frame"], {"voxel": inp})
        ort_ms = (time.perf_counter() - t0) / N * 1000

        logger.info("PyTorch CPU:     %.1f ms/frame", pt_ms)
        logger.info("ONNX Runtime:    %.1f ms/frame", ort_ms)
        logger.info("Speedup:         %.1fx", pt_ms / ort_ms)
    except Exception as e:
        logger.warning("Could not benchmark: %s", e)


def main() -> None:
    p = argparse.ArgumentParser(description="Export ConvLSTM decoder to ONNX")
    p.add_argument("--weights", default=None,
                   help="Path to decoder_v1.pt (optional)")
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
    args = p.parse_args()

    export(
        args.weights, args.output,
        n_bins=args.n_bins, height=args.height,
        width=args.width, opset=args.opset,
    )


if __name__ == "__main__":
    main()
