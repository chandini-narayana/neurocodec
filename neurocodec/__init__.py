# Copyright (c) 2026 Chandini Narayana
# SPDX-License-Identifier: MIT
"""
NeuroCodec — Neuromorphic-Inspired Spike-Driven Video Codec
===========================================================
A software DVS event simulator + learned voxel-grid entropy encoder +
ConvLSTM frame reconstructor.  Works on any standard camera input;
no $5,000 event-camera hardware required.

Quick start
-----------
>>> from neurocodec import simulate, encode, decode
>>> events = simulate("clip.mp4", threshold=0.2)
>>> encode("clip.mp4", "clip.nec", threshold=0.2)
>>> decode("clip.nec", "clip_reconstructed.mp4")
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("neurocodec")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

__author__ = "Chandini Narayana"
__license__ = "MIT"
__year__ = "2026"

from neurocodec.simulator import DVSSimulator, simulate
from neurocodec.encoder import VoxelEncoder, encode
from neurocodec.decoder import ConvLSTMDecoder, decode

__all__ = [
    "DVSSimulator", "simulate",
    "VoxelEncoder", "encode",
    "ConvLSTMDecoder", "decode",
    "__version__",
]
