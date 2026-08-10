# Copyright (c) 2026 Chandini Narayana
# SPDX-License-Identifier: MIT
"""
ConvLSTM Cell and Stack
=======================
Convolutional LSTM that processes sequences of 2-D feature maps while
maintaining spatial structure through the hidden state.

Reference: Shi et al., "Convolutional LSTM Network: A Machine Learning
Approach for Precipitation Nowcasting", NeurIPS 2015.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell.

    Parameters
    ----------
    in_channels  : int  Number of input channels.
    hidden_channels : int  Number of hidden (and cell) state channels.
    kernel_size  : int  Spatial kernel size (default 3).
    bias         : bool Whether to use bias in convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels     = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size     = kernel_size
        padding              = kernel_size // 2

        # One convolution that computes all 4 gates at once
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
            bias=bias,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.gates.weight)
        if self.gates.bias is not None:
            nn.init.zeros_(self.gates.bias)
            # Initialise forget gate bias to 1 for better gradient flow
            nn.init.ones_(self.gates.bias[self.hidden_channels : 2 * self.hidden_channels])

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor],
        c: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, in_channels, H, W)
        h : (B, hidden_channels, H, W)  or None (zeros)
        c : (B, hidden_channels, H, W)  or None (zeros)

        Returns
        -------
        h_new, c_new
        """
        B, _, H, W = x.shape
        device = x.device
        if h is None:
            h = torch.zeros(B, self.hidden_channels, H, W, device=device)
        if c is None:
            c = torch.zeros(B, self.hidden_channels, H, W, device=device)

        combined = torch.cat([x, h], dim=1)
        gates    = self.gates(combined)

        HC = self.hidden_channels
        i_gate = torch.sigmoid(gates[:, :HC])           # input
        f_gate = torch.sigmoid(gates[:, HC:2*HC])       # forget
        g_gate = torch.tanh  (gates[:, 2*HC:3*HC])     # cell candidate
        o_gate = torch.sigmoid(gates[:, 3*HC:])         # output

        c_new = f_gate * c + i_gate * g_gate
        h_new = o_gate * torch.tanh(c_new)
        return h_new, c_new

    def init_hidden(
        self, batch_size: int, height: int, width: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        return h, c


class ConvLSTMStack(nn.Module):
    """
    A stack of ConvLSTM cells with residual connections between layers.

    Parameters
    ----------
    in_channels     : int
    hidden_channels : list[int]  Hidden dim for each layer.
    kernel_size     : int
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.n_layers = len(hidden_channels)
        cells = []
        c_in  = in_channels
        for h in hidden_channels:
            cells.append(ConvLSTMCell(c_in, h, kernel_size))
            c_in = h
        self.cells = nn.ModuleList(cells)
        self.out_channels = hidden_channels[-1]

    def forward(
        self,
        x_seq: torch.Tensor,
        hidden: Optional[List[Tuple]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple]]:
        """
        Parameters
        ----------
        x_seq  : (B, T, C, H, W)  Input sequence.
        hidden : list of (h, c) tuples per layer, or None.

        Returns
        -------
        out_seq : (B, T, hidden[-1], H, W)
        hidden  : updated list of (h, c) per layer
        """
        B, T, C, H, W = x_seq.shape
        device = x_seq.device

        if hidden is None:
            hidden = [cell.init_hidden(B, H, W, device) for cell in self.cells]

        out_seq = []
        for t in range(T):
            x_t = x_seq[:, t]
            new_hidden = []
            for i, cell in enumerate(self.cells):
                h, c        = hidden[i]
                h_new, c_new = cell(x_t, h, c)
                new_hidden.append((h_new, c_new))
                x_t = h_new
            hidden  = new_hidden
            out_seq.append(x_t)

        return torch.stack(out_seq, dim=1), hidden
