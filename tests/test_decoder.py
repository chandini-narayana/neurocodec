# Copyright (c) 2026 Chandini Narayana
# SPDX-License-Identifier: MIT
"""Tests for ConvLSTM decoder."""

import torch
import pytest

from neurocodec.decoder import ConvLSTMDecoder, CombinedLoss
from neurocodec.convlstm import ConvLSTMCell, ConvLSTMStack


class TestConvLSTMCell:
    def test_output_shape(self):
        cell = ConvLSTMCell(in_channels=8, hidden_channels=16, kernel_size=3)
        x    = torch.randn(2, 8, 16, 16)
        h, c = cell(x, None, None)
        assert h.shape == (2, 16, 16, 16)
        assert c.shape == (2, 16, 16, 16)

    def test_deterministic_with_seed(self):
        torch.manual_seed(0)
        cell = ConvLSTMCell(8, 16)
        x    = torch.randn(1, 8, 8, 8)
        h1, _ = cell(x, None, None)
        torch.manual_seed(0)
        cell2 = ConvLSTMCell(8, 16)
        h2, _ = cell2(x, None, None)
        assert torch.allclose(h1, h2, atol=1e-6)


class TestConvLSTMStack:
    def test_sequence_output_shape(self):
        stack  = ConvLSTMStack(in_channels=5, hidden_channels=[16, 8])
        x_seq  = torch.randn(2, 4, 5, 12, 12)   # (B, T, C, H, W)
        out, _ = stack(x_seq)
        assert out.shape == (2, 4, 8, 12, 12)


class TestConvLSTMDecoder:
    def test_single_frame_output_shape(self):
        model = ConvLSTMDecoder(n_bins=5)
        voxel = torch.randn(1, 5, 32, 32)
        out   = model(voxel)
        assert out.shape == (1, 1, 32, 32)

    def test_sequence_output_shape(self):
        model = ConvLSTMDecoder(n_bins=5)
        voxel = torch.randn(2, 6, 5, 32, 32)  # (B, T, C, H, W)
        out   = model(voxel)
        assert out.shape == (2, 6, 1, 32, 32)

    def test_output_in_zero_one(self):
        model = ConvLSTMDecoder(n_bins=5)
        voxel = torch.randn(1, 5, 16, 16)
        out   = model(voxel)
        assert out.min().item() >= 0.0 - 1e-6
        assert out.max().item() <= 1.0 + 1e-6

    def test_reset_state(self):
        model = ConvLSTMDecoder(n_bins=5)
        voxel = torch.randn(1, 5, 16, 16)
        model(voxel)                  # sets hidden state
        model.reset_state()
        assert model._hidden is None  # properly reset

    def test_parameter_count_reasonable(self):
        model  = ConvLSTMDecoder(n_bins=5)
        n_params = model.count_parameters()
        # Should be < 5M for a compact model
        assert n_params < 5_000_000, f"Too many params: {n_params:,}"
        assert n_params > 10_000,    f"Too few params: {n_params:,}"

    def test_no_nan_in_output(self):
        model = ConvLSTMDecoder(n_bins=5)
        for _ in range(3):
            voxel = torch.randn(1, 5, 16, 16)
            out   = model(voxel)
            assert not torch.isnan(out).any(), "NaN detected in decoder output"
