"""
DAICS WDNN (Wide and Deep Neural Network) Implementation
Paper: "DAICS: A Deep Learning Solution for Anomaly Detection in Industrial Control Systems"
Architecture per Figure 2 and Table IV.

Input: (batch, n_features=44, Win=60) -> sensors + actuators
Output per section g: (batch, mgse_g, Wout=4)

WDNN Structure:
- Wide branch: DL1 (cross-product layer) -> 4 neurons
- Deep branch: DL2(180) -> CL1(64, k=2) -> MP1(2) -> CL2(128, k=2) -> MP2(2) -> DL3(4)
- Aggregation: DL4(80)
- Per-section output (G=3):
    DL5(2.25*mgse) -> DL6(1.5*mgse) -> DL7(mgse) -> Output mgse*Wout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from daics.config import WDNN, WINDOW, DATA, WDNNConfig


class LeakyReLU(nn.Module):
    """LeakyReLU with alpha=0.01 as per paper"""
    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return F.leaky_relu(x, negative_slope=self.alpha)


class WideBranch(nn.Module):
    """
    Wide branch: Cross-product layer (DL1)
    Input: (batch, n_features, Win)
    Output: (batch, Wout=4)
    """
    def __init__(self, n_features: int, Win: int, Wout: int):
        super().__init__()
        self.n_features = n_features
        self.Win = Win
        self.Wout = Wout

        # Cross-product layer: all pairwise products of input features
        # For each time step, compute pairwise products
        # This is implemented as a fully connected layer that learns cross-products
        self.dl1 = nn.Linear(n_features * Win, Wout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features, Win)
        batch_size = x.shape[0]
        # Flatten features and time
        x_flat = x.view(batch_size, -1)  # (batch, n_features * Win)
        out = self.dl1(x_flat)  # (batch, Wout)
        return out


class DeepBranch(nn.Module):
    """
    Deep branch: DL2 -> CL1 -> MP1 -> CL2 -> MP2 -> DL3
    Input: (batch, n_features, Win)
    Output: (batch, Wout=4)
    """
    def __init__(
        self,
        n_features: int,
        Win: int,
        Wout: int,
        dl2_neurons: int = 180,
        cl1_kernels: int = 64,
        cl1_kernel_size: int = 2,
        mp1_pool_size: int = 2,
        cl2_kernels: int = 128,
        cl2_kernel_size: int = 2,
        mp2_pool_size: int = 2,
        dl3_neurons: int = 4,
        alpha: float = 0.01
    ):
        super().__init__()

        # DL2: Dense layer
        self.dl2 = nn.Linear(n_features * Win, dl2_neurons)
        self.act_dl2 = LeakyReLU(alpha)

        # CL1: Conv1d over time dimension
        # Input needs to be reshaped to (batch, channels, length)
        self.cl1 = nn.Conv1d(
            in_channels=1,
            out_channels=cl1_kernels,
            kernel_size=cl1_kernel_size,
            padding=0
        )
        self.act_cl1 = LeakyReLU(alpha)
        self.mp1 = nn.MaxPool1d(kernel_size=mp1_pool_size)

        # CL2: Conv1d
        self.cl2 = nn.Conv1d(
            in_channels=cl1_kernels,
            out_channels=cl2_kernels,
            kernel_size=cl2_kernel_size,
            padding=0
        )
        self.act_cl2 = LeakyReLU(alpha)
        self.mp2 = nn.MaxPool1d(kernel_size=mp2_pool_size)

        # Calculate output size after conv/pool
        # DL2 output: dl2_neurons -> reshape to (batch, 1, dl2_neurons)
        # CL1: (dl2_neurons - cl1_kernel_size + 1) -> pool -> // mp1_pool_size
        l1 = (dl2_neurons - cl1_kernel_size + 1) // mp1_pool_size
        # CL2: (l1 - cl2_kernel_size + 1) -> pool -> // mp2_pool_size
        l2 = (l1 - cl2_kernel_size + 1) // mp2_pool_size
        self.flat_size = cl2_kernels * l2

        self.dl3 = nn.Linear(self.flat_size, dl3_neurons)
        self.act_dl3 = LeakyReLU(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # DL2
        x_flat = x.view(batch_size, -1)  # (batch, n_features * Win)
        x = self.dl2(x_flat)
        x = self.act_dl2(x)

        # Reshape for Conv1d: (batch, 1, dl2_neurons)
        x = x.unsqueeze(1)

        # CL1 -> MP1
        x = self.cl1(x)
        x = self.act_cl1(x)
        x = self.mp1(x)

        # CL2 -> MP2
        x = self.cl2(x)
        x = self.act_cl2(x)
        x = self.mp2(x)

        # Flatten and DL3
        x = x.view(batch_size, -1)
        x = self.dl3(x)
        x = self.act_dl3(x)

        return x


class OutputSection(nn.Module):
    """
    Per-section output branch: DL5 -> DL6 -> DL7
    Input: (batch, DL4_neurons=80)
    Output: (batch, mgse * Wout) -> reshape to (batch, mgse, Wout)
    """
    def __init__(
        self,
        mgse: int,
        Wout: int,
        dl4_neurons: int = 80,
        dl5_factor: float = 2.25,
        dl6_factor: float = 1.5,
        dl7_factor: float = 1.0,
        alpha: float = 0.01
    ):
        super().__init__()
        self.mgse = mgse
        self.Wout = Wout

        dl5_neurons = int(round(dl5_factor * mgse))
        dl6_neurons = int(round(dl6_factor * mgse))
        dl7_neurons = int(round(dl7_factor * mgse))  # = mgse

        self.dl5 = nn.Linear(dl4_neurons, dl5_neurons)
        self.act5 = LeakyReLU(alpha)
        self.dl6 = nn.Linear(dl5_neurons, dl6_neurons)
        self.act6 = LeakyReLU(alpha)
        self.dl7 = nn.Linear(dl6_neurons, dl7_neurons * Wout)
        # No activation on final output (linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, DL4_neurons)
        x = self.dl5(x)
        x = self.act5(x)
        x = self.dl6(x)
        x = self.act6(x)
        x = self.dl7(x)
        # Reshape to (batch, mgse, Wout)
        x = x.view(-1, self.mgse, self.Wout)
        return x


class WDNN(nn.Module):
    """
    Complete WDNN model with G output sections.
    """
    def __init__(
        self,
        n_features: int = DATA.n_features,
        n_sensors: int = DATA.n_sensors,
        Win: int = WINDOW.Win,
        Wout: int = WINDOW.Wout,
        mgse_per_section: Tuple[int, ...] = DATA.mgse_per_section,
        G: int = WDNN.G,
        wdnn_config = WDNN
    ):
        super().__init__()
        self.n_features = n_features
        self.n_sensors = n_sensors
        self.Win = Win
        self.Wout = Wout
        self.mgse_per_section = mgse_per_section
        self.G = G

        # Wide branch
        self.wide = WideBranch(n_features, Win, Wout)

        # Deep branch
        self.deep = DeepBranch(
            n_features=n_features,
            Win=Win,
            Wout=Wout,
            dl2_neurons=wdnn_config.DL2_neurons,
            cl1_kernels=wdnn_config.CL1_kernels,
            cl1_kernel_size=wdnn_config.CL1_kernel_size,
            mp1_pool_size=wdnn_config.MP1_pool_size,
            cl2_kernels=wdnn_config.CL2_kernels,
            cl2_kernel_size=wdnn_config.CL2_kernel_size,
            mp2_pool_size=wdnn_config.MP2_pool_size,
            dl3_neurons=wdnn_config.DL3_neurons,
            alpha=wdnn_config.leaky_relu_alpha
        )

        # Aggregation: DL4
        self.dl4 = nn.Linear(Wout + Wout, wdnn_config.DL4_neurons)  # wide + deep = 8
        self.act4 = LeakyReLU(wdnn_config.leaky_relu_alpha)

        # Per-section output branches
        self.output_sections = nn.ModuleList([
            OutputSection(
                mgse=mgse,
                Wout=Wout,
                dl4_neurons=wdnn_config.DL4_neurons,
                dl5_factor=wdnn_config.DL5_factor,
                dl6_factor=wdnn_config.DL6_factor,
                dl7_factor=wdnn_config.DL7_factor,
                alpha=wdnn_config.leaky_relu_alpha
            )
            for mgse in mgse_per_section
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass.
        Input: (batch, n_features, Win)
        Output: List of G tensors, each (batch, mgse_g, Wout)
        """
        batch_size = x.shape[0]

        # Wide branch
        wide_out = self.wide(x)  # (batch, Wout)

        # Deep branch
        deep_out = self.deep(x)  # (batch, Wout)

        # Aggregation
        combined = torch.cat([wide_out, deep_out], dim=1)  # (batch, 2*Wout=8)
        aggregated = self.dl4(combined)  # (batch, DL4_neurons=80)
        aggregated = self.act4(aggregated)

        # Per-section outputs
        outputs = []
        for section in self.output_sections:
            out = section(aggregated)  # (batch, mgse, Wout)
            outputs.append(out)

        return outputs

    def get_section_output(self, x: torch.Tensor, section_idx: int) -> torch.Tensor:
        """Get output for a specific section only (for TTNN training)"""
        outputs = self.forward(x)
        return outputs[section_idx]


# For testing
if __name__ == "__main__":
    # Test model creation and forward pass
    model = WDNN()
    print("WDNN Model Structure:")
    print(f"  Wide branch: {sum(p.numel() for p in model.wide.parameters())} params")
    print(f"  Deep branch: {sum(p.numel() for p in model.deep.parameters())} params")
    print(f"  DL4: {sum(p.numel() for p in model.dl4.parameters())} params")
    for i, section in enumerate(model.output_sections):
        print(f"  Section {i}: {sum(p.numel() for p in section.parameters())} params")
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total: {total:,} params")

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, DATA.n_features, WINDOW.Win)
    outputs = model(x)
    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")
    for i, out in enumerate(outputs):
        print(f"  Section {i} output: {out.shape} (expected mgse={DATA.mgse_per_section[i]}, Wout={WINDOW.Wout})")