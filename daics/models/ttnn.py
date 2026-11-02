"""
DAICS TTNN (Threshold Tuning Neural Network) Implementation
Paper: "DAICS: A Deep Learning Solution for Anomaly Detection in Industrial Control Systems"
Architecture per Figure 4 and Table IV (SWaT column).

One TTNN per WDNN output section (G=3).
Input: (batch, 1, Win=60) - univariate MSE time series of length Win
Output: (batch, 1) - estimated prediction error for threshold tuning

TTNN Structure (Table IV SWaT):
CL1: 2 kernels, kernel_size=2, stride=1
MP1: pool_size=2
CL2: 4 kernels (Wout), kernel_size=2, stride=1
MP2: pool_size=Wout=4
DL: 1 neuron
Activation: LeakyReLU(alpha=0.01)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from daics.config import TTNNConfig, WINDOW, DATA, WDNN


class LeakyReLU(nn.Module):
    """LeakyReLU with alpha=0.01 as per paper"""
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return F.leaky_relu(x, negative_slope=self.alpha)


class TTNN(nn.Module):
    """
    Threshold Tuning Neural Network.
    One per WDNN output section.

    Paper (Section VI-D, Figure 4, Table IV):
    - Input: univariate MSE time series of length Win (60)
    - "We treat the prediction error MSE as a univariate time series"
    - CL1: 2 kernels, kernel size 2, stride 1
    - MP1: pool size 2
    - CL2: Wout kernels (4), kernel size 2
    - MP2: pool size Wout (4)
    - DL: 1 neuron
    - Only first element of output is kept (Algorithm 1 line 6)
    """
    def __init__(
        self,
        Win: int = WINDOW.Win,
        cl1_kernels: int = 2,
        cl1_kernel_size: int = 2,
        mp1_pool_size: int = 2,
        cl2_kernels: int = 4,  # = Wout
        cl2_kernel_size: int = 2,
        mp2_pool_size: int = 4,  # = Wout
        dl_neurons: int = 1,
        alpha: float = 0.01
    ):
        super().__init__()
        self.Win = Win

        # Input is univariate: (batch, 1, Win)
        # CL1: Conv1d with 1 input channel
        cl1_padding = cl1_kernel_size // 2
        self.cl1 = nn.Conv1d(
            in_channels=1,
            out_channels=cl1_kernels,
            kernel_size=cl1_kernel_size,
            padding=cl1_padding,
            stride=1
        )
        self.act_cl1 = LeakyReLU(alpha)
        self.mp1 = nn.MaxPool1d(kernel_size=mp1_pool_size)

        # CL2: Conv1d
        cl2_padding = cl2_kernel_size // 2
        self.cl2 = nn.Conv1d(
            in_channels=cl1_kernels,
            out_channels=cl2_kernels,
            kernel_size=cl2_kernel_size,
            padding=cl2_padding,
            stride=1
        )
        self.act_cl2 = LeakyReLU(alpha)
        self.mp2 = nn.MaxPool1d(kernel_size=mp2_pool_size)

        # Calculate flattened size
        # Input: (batch, 1, Win=60)
        # CL1: padding=1 -> 60 -> MP1(2) -> 30
        # CL2: padding=1 -> 30 -> MP2(4) -> 7 (floor)
        l1 = Win // mp1_pool_size  # 60 // 2 = 30
        l2 = l1 // mp2_pool_size   # 30 // 4 = 7
        self.flat_size = cl2_kernels * l2  # 4 * 7 = 28

        # DL: Single output neuron
        self.dl = nn.Linear(self.flat_size, dl_neurons)
        # No activation on output (linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input: (batch, 1, Win) - univariate MSE sequence
        Output: (batch, 1) - estimated prediction error (only first element kept per Algorithm 1)
        """
        batch_size = x.shape[0]
        if batch_size == 0:
            return torch.zeros(0, 1, device=x.device)

        # CL1 -> MP1
        x = self.cl1(x)  # (batch, cl1_kernels, Win)
        x = self.act_cl1(x)
        x = self.mp1(x)  # (batch, cl1_kernels, Win/2)

        # CL2 -> MP2
        x = self.cl2(x)
        x = self.act_cl2(x)
        x = self.mp2(x)  # (batch, cl2_kernels, Win/2/4)

        # Flatten and DL
        x = x.view(batch_size, -1)
        x = self.dl(x)  # (batch, 1)

        return x


class TTNNEnsemble(nn.Module):
    """
    Ensemble of G TTNNs, one per WDNN output section.
    Each TTNN takes univariate MSE sequence of length Win.
    """
    def __init__(
        self,
        G: int = WDNN.G,
        Win: int = WINDOW.Win,
        ttnn_config = TTNNConfig
    ):
        super().__init__()
        self.G = G
        self.Win = Win

        self.ttnns = nn.ModuleList([
            TTNN(
                Win=Win,
                cl1_kernels=ttnn_config.CL1_kernels,
                cl1_kernel_size=ttnn_config.CL1_kernel_size,
                mp1_pool_size=ttnn_config.MP1_pool_size,
                cl2_kernels=ttnn_config.CL2_kernels,
                cl2_kernel_size=ttnn_config.CL2_kernel_size,
                mp2_pool_size=ttnn_config.MP2_pool_size,
                dl_neurons=ttnn_config.DL_neurons,
                alpha=ttnn_config.leaky_relu_alpha
            )
            for _ in range(G)
        ])

    def forward(self, mse_sequences: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass for all sections.
        Input: List of G tensors, each (batch, 1, Win)
        Output: List of G tensors, each (batch, 1)
        """
        outputs = []
        for i, ttnn in enumerate(self.ttnns):
            out = ttnn(mse_sequences[i])  # (batch, 1)
            outputs.append(out)
        return outputs

    def forward_section(self, section_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single section."""
        return self.ttnns[section_idx](x)


# For testing
if __name__ == "__main__":
    # Test individual TTNN with correct input shape (batch, 1, Win)
    model = TTNN()
    print(f"TTNN (Win={WINDOW.Win}):")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.randn(32, 1, WINDOW.Win)
    out = model(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")

    print()
    # Test ensemble
    ensemble = TTNNEnsemble()
    print(f"TTNN Ensemble (G={WDNN.G}):")
    print(f"  Total parameters: {sum(p.numel() for p in ensemble.parameters()):,}")

    mse_sequences = [
        torch.randn(32, 1, WINDOW.Win)
        for _ in range(WDNN.G)
    ]
    outputs = ensemble(mse_sequences)
    for i, out in enumerate(outputs):
        print(f"  Section {i}: {out.shape}")