# Copyright (c) 2026 Chandini Narayana
# SPDX-License-Identifier: MIT
"""Tests for voxel encoder and .nec format."""

import os
import tempfile

import numpy as np
import pytest

from neurocodec.encoder import (
    VoxelGrid, NECWriter, NECReader, topk_sparsify, HuffmanCoder
)
from neurocodec.simulator import DVSSimulator, EventStream
from tests.conftest import make_moving_frames, make_static_frames


class TestVoxelGrid:
    def test_output_shape(self):
        sim    = DVSSimulator(threshold=0.1)
        frames = make_moving_frames()
        stream = sim.simulate_array(frames)
        vg     = VoxelGrid(n_bins=5)
        grids  = vg.build(stream)
        T      = stream.n_frames
        assert grids.shape == (T, 5, stream.height, stream.width)

    def test_values_in_minus1_plus1(self):
        sim    = DVSSimulator(threshold=0.1)
        stream = sim.simulate_array(make_moving_frames())
        grids  = VoxelGrid(5).build(stream)
        assert grids.min() >= -1.0 - 1e-6
        assert grids.max() <=  1.0 + 1e-6

    def test_static_stream_all_zeros(self):
        stream = EventStream(width=16, height=16, n_frames=5, threshold=0.2)
        grids  = VoxelGrid(3).build(stream)
        assert np.all(grids == 0)

    def test_moving_stream_nonzero(self):
        sim    = DVSSimulator(threshold=0.05)
        stream = sim.simulate_array(make_moving_frames())
        grids  = VoxelGrid(5).build(stream)
        assert grids.max() > 0, "Moving frames must produce nonzero voxel"


class TestTopKSparsify:
    def test_keeps_k_elements(self):
        arr = np.random.randn(100).astype(np.float32)
        idx, vals = topk_sparsify(arr.reshape(10, 10), k_frac=0.2)
        assert len(idx) == 20  # 20% of 100

    def test_values_match_original(self):
        arr = np.arange(100, dtype=np.float32)
        idx, vals = topk_sparsify(arr.reshape(10, 10), k_frac=0.1)
        np.testing.assert_allclose(arr.ravel()[idx], vals)

    def test_minimum_one_element(self):
        arr = np.ones((2, 2), dtype=np.float32)
        idx, vals = topk_sparsify(arr, k_frac=0.001)
        assert len(idx) >= 1


class TestHuffmanCoder:
    def test_roundtrip(self):
        symbols = [1, 2, 1, 3, 1, 2, 4, 1, 1]
        codes   = HuffmanCoder.build_tree(symbols)
        encoded = HuffmanCoder.encode(symbols, codes)
        decoded = HuffmanCoder.decode(encoded, codes, len(symbols))
        assert decoded == symbols

    def test_single_symbol(self):
        symbols = [7, 7, 7]
        codes   = HuffmanCoder.build_tree(symbols)
        encoded = HuffmanCoder.encode(symbols, codes)
        decoded = HuffmanCoder.decode(encoded, codes, 3)
        assert decoded == symbols

    def test_empty_symbols(self):
        codes = HuffmanCoder.build_tree([])
        assert codes == {}

    def test_large_alphabet(self):
        import random
        random.seed(42)
        symbols = [random.randint(0, 255) for _ in range(500)]
        codes   = HuffmanCoder.build_tree(symbols)
        encoded = HuffmanCoder.encode(symbols, codes)
        decoded = HuffmanCoder.decode(encoded, codes, len(symbols))
        assert decoded == symbols


class TestNECFormat:
    def test_write_read_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".nec", delete=False) as f:
            path = f.name
        try:
            W, H, B, T = 16, 16, 5, 4
            writer = NECWriter(path, W, H, B, 0.2, T, k_frac=0.5)
            for _ in range(T):
                voxel = np.random.randn(B, H, W).astype(np.float32)
                writer.add_voxel(voxel)
            writer.write()

            reader = NECReader(path)
            assert reader.width    == W
            assert reader.height   == H
            assert reader.n_bins   == B
            assert reader.n_frames == T
            dense = reader.to_dense()
            assert dense.shape == (T, B, H, W)
        finally:
            os.unlink(path)

    def test_magic_bytes_validated(self):
        with tempfile.NamedTemporaryFile(suffix=".nec", delete=False) as f:
            f.write(b"BADMAGIC")
            path = f.name
        try:
            with pytest.raises(ValueError):
                NECReader(path)
        finally:
            os.unlink(path)

    def test_file_size_nonzero(self):
        with tempfile.NamedTemporaryFile(suffix=".nec", delete=False) as f:
            path = f.name
        try:
            writer = NECWriter(path, 8, 8, 3, 0.2, 2, k_frac=0.5)
            for _ in range(2):
                writer.add_voxel(np.random.randn(3, 8, 8).astype(np.float32))
            size = writer.write()
            assert size > 0
        finally:
            os.unlink(path)

    def test_iter_frames_count(self):
        with tempfile.NamedTemporaryFile(suffix=".nec", delete=False) as f:
            path = f.name
        try:
            T = 6
            writer = NECWriter(path, 8, 8, 3, 0.2, T, k_frac=0.3)
            for _ in range(T):
                writer.add_voxel(np.random.randn(3, 8, 8).astype(np.float32))
            writer.write()
            reader = NECReader(path)
            count = sum(1 for _ in reader.iter_frames())
            assert count == T
        finally:
            os.unlink(path)
