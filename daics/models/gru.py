"""Separate GRU temporal prediction model for the proposed DAICS extension.

Input follows the existing DAICS convention: (batch, 44 features, Win=60).
Outputs are three section tensors, each shaped (batch, section_sensors, Wout=4).
This module does not modify or wrap the frozen WDNN.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from daics.config import DATA, WINDOW


class GRUPredictor(nn.Module):
    """One-layer GRU with independent sensor-section prediction heads."""

    def __init__(
        self,
        input_size: int = DATA.n_features,
        hidden_size: int = 64,
        section_sizes: Sequence[int] = DATA.mgse_per_section,
        Win: int = WINDOW.Win,
        Wout: int = WINDOW.Wout,
    ):
        super().__init__()
        if sum(section_sizes) != DATA.n_sensors:
            raise ValueError("Section sizes must cover all retained sensors exactly")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.section_sizes = tuple(int(size) for size in section_sizes)
        self.Win = Win
        self.Wout = Wout

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.output_heads = nn.ModuleList(
            nn.Linear(hidden_size, size * Wout) for size in self.section_sizes
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Map (batch, features, Win) to ordered per-section forecasts."""
        if x.ndim != 3:
            raise ValueError(f"Expected a 3-D input (batch,features,time), got {tuple(x.shape)}")
        if x.shape[1] != self.input_size or x.shape[2] != self.Win:
            raise ValueError(
                f"Expected (batch,{self.input_size},{self.Win}), got {tuple(x.shape)}"
            )
        sequence = x.transpose(1, 2).contiguous()  # (batch, Win, features)
        _, hidden = self.gru(sequence)              # (1, batch, hidden_size)
        final_hidden = hidden[-1]                   # (batch, hidden_size)
        return [
            head(final_hidden).reshape(x.shape[0], sensors, self.Wout)
            for head, sensors in zip(self.output_heads, self.section_sizes)
        ]


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
