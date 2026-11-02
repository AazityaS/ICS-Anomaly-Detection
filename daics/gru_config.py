"""Configuration for the proposed GRU predictor; separate from DAICS settings."""

from dataclasses import asdict, dataclass

from daics.config import DATA, WINDOW


@dataclass(frozen=True)
class GRUTrainingConfig:
    input_size: int = DATA.n_features
    hidden_size: int = 64
    num_layers: int = 1
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 15
    patience: int = 3
    gradient_clip_norm: float = 1.0
    seed: int = 20261101
    Win: int = WINDOW.Win
    H: int = WINDOW.H
    Wout: int = WINDOW.Wout
    section_sizes: tuple[int, ...] = DATA.mgse_per_section

    def to_dict(self):
        return asdict(self)


GRU_CONFIG = GRUTrainingConfig()
