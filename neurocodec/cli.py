# Copyright (c) 2026 Chandini Narayana
# SPDX-License-Identifier: MIT
"""
NeuroCodec CLI
==============
Usage:
    neurocodec encode input.mp4 output.nec [--threshold 0.2] [--bins 5] [--kfrac 0.1]
    neurocodec decode input.nec output.mp4 [--weights decoder.pt] [--device cpu]
    neurocodec simulate input.mp4           [--threshold 0.2] [--output events.mp4]
    neurocodec info    input.nec
    neurocodec benchmark input.mp4          [--thresholds 0.1,0.2,0.3]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("neurocodec.cli")


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_encode(args: argparse.Namespace) -> None:
    from neurocodec.encoder import encode
    stats = encode(
        args.input,
        args.output,
        threshold=args.threshold,
        n_bins=args.bins,
        k_frac=args.kfrac,
    )
    print("\n── Encoding complete ──────────────────")
    print(f"  Events captured : {stats['n_events']:,}")
    print(f"  Event density   : {stats['event_density']:.4f}")
    print(f"  Frames          : {stats['n_frames']}")
    print(f"  Output size     : {stats['file_size_kb']:.1f} KB")
    print(f"  Compression     : {stats['compression_ratio']:.1f}×")
    print("──────────────────────────────────────\n")


def cmd_decode(args: argparse.Namespace) -> None:
    from neurocodec.decoder import decode
    result = decode(
        args.input,
        args.output,
        weights_path=args.weights,
        device=args.device,
    )
    print("\n── Decoding complete ──────────────────")
    print(f"  Frames decoded  : {result['n_frames']}")
    print(f"  Device used     : {result['device']}")
    print(f"  Output          : {result['output_path']}")
    print("──────────────────────────────────────\n")


def cmd_simulate(args: argparse.Namespace) -> None:
    from neurocodec.simulator import DVSSimulator, save_event_video
    sim    = DVSSimulator(threshold=args.threshold)
    stream = sim.simulate_video(args.input)
    print(f"\nSimulated {len(stream):,} events from {Path(args.input).name}")
    print(f"  Density  : {stream.density():.4f}")
    print(f"  Frames   : {stream.n_frames}")
    if args.output:
        save_event_video(stream, args.output)
        print(f"  Saved to : {args.output}")
    print()


def cmd_info(args: argparse.Namespace) -> None:
    from neurocodec.encoder import NECReader
    r = NECReader(args.input)
    size_kb = Path(args.input).stat().st_size / 1024
    print(f"\n── .nec file info: {args.input} ──")
    print(f"  Resolution  : {r.width} × {r.height}")
    print(f"  Time bins   : {r.n_bins}")
    print(f"  Threshold C : {r.threshold}")
    print(f"  Frames      : {r.n_frames}")
    print(f"  File size   : {size_kb:.1f} KB")
    print()


def cmd_benchmark(args: argparse.Namespace) -> None:
    import json
    thresholds = [float(t) for t in args.thresholds.split(",")]
    from neurocodec.simulator import DVSSimulator
    print(f"\n── Threshold sweep: {Path(args.input).name} ──")
    print(f"{'C':>8}  {'Events':>12}  {'Density':>10}")
    print("─" * 36)
    results = {}
    for c in thresholds:
        sim    = DVSSimulator(threshold=c)
        stream = sim.simulate_video(args.input)
        print(f"{c:>8.3f}  {len(stream):>12,}  {stream.density():>10.4f}")
        results[str(c)] = {
            "threshold": c,
            "n_events":  len(stream),
            "density":   stream.density(),
            "n_frames":  stream.n_frames,
        }
    print()
    if getattr(args, "output", None):
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neurocodec",
        description="NeuroCodec — Neuromorphic-Inspired Spike-Driven Video Codec",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--version", action="version",
                   version=f"%(prog)s {_get_version()}")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # encode
    enc = sub.add_parser("encode", help="Encode video → .nec")
    enc.add_argument("input",  help="Input video path")
    enc.add_argument("output", help="Output .nec path")
    enc.add_argument("--threshold", type=float, default=0.2,
                     metavar="C", help="DVS threshold (default 0.2)")
    enc.add_argument("--bins",  type=int,   default=5,
                     help="Voxel time bins (default 5)")
    enc.add_argument("--kfrac", type=float, default=0.1,
                     help="Top-K sparsity fraction (default 0.1)")
    enc.set_defaults(func=cmd_encode)

    # decode
    dec = sub.add_parser("decode", help="Decode .nec → video")
    dec.add_argument("input",  help="Input .nec path")
    dec.add_argument("output", help="Output video path")
    dec.add_argument("--weights", default=None,
                     help="Path to model weights (.pt)")
    dec.add_argument("--device",  default=None,
                     help="'cpu' or 'cuda' (default: auto)")
    dec.set_defaults(func=cmd_decode)

    # simulate
    sim_ = sub.add_parser("simulate", help="Simulate DVS events from video")
    sim_.add_argument("input",  help="Input video path")
    sim_.add_argument("--threshold", type=float, default=0.2)
    sim_.add_argument("--output", default=None,
                      help="Optional: save event visualisation video")
    sim_.set_defaults(func=cmd_simulate)

    # info
    inf = sub.add_parser("info", help="Show .nec file metadata")
    inf.add_argument("input", help=".nec file path")
    inf.set_defaults(func=cmd_info)

    # benchmark
    bm = sub.add_parser("benchmark", help="Threshold sweep for event density")
    bm.add_argument("input", help="Input video path")
    bm.add_argument("--thresholds", default="0.05,0.1,0.2,0.3,0.5",
                    help="Comma-separated threshold values")
    bm.add_argument("--output", default=None,
                    help="Optional: save results as JSON file")
    bm.set_defaults(func=cmd_benchmark)

    return p


def _get_version() -> str:
    try:
        from importlib.metadata import version
        return version("neurocodec")
    except Exception:
        return "0.1.0-dev"


def main(argv=None) -> None:
    parser = build_parser()
    args   = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
