"""Definição da rede neural MLP para previsão de Churn."""

import torch
import torch.nn as nn


class ChurnMLP(nn.Module):
    """Multi-Layer Perceptron para classificação binária de Churn.

    Args:
        input_dim: Número de features de entrada.
        hidden_dims: Lista com o número de neurônios em cada camada oculta.
        dropout: Taxa de dropout entre camadas.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int] = None, dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)
