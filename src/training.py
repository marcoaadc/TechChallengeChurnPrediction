"""Utilitários de treinamento: Dataset, training loop e early stopping."""

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class ChurnDataset(Dataset):
    """Dataset PyTorch para dados tabulares de Churn."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EarlyStopping:
    """Interrompe o treinamento quando a loss de validação para de melhorar.

    Args:
        patience: Épocas sem melhoria antes de parar.
        min_delta: Melhoria mínima para considerar progresso.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    pos_weight: float = 1.0,
    patience: int = 10,
    device: str = "cpu",
):
    """Treina o modelo MLP com early stopping.

    Args:
        model: Modelo PyTorch.
        train_loader: DataLoader de treino.
        val_loader: DataLoader de validação.
        epochs: Número máximo de épocas.
        lr: Learning rate.
        pos_weight: Peso da classe positiva na BCEWithLogitsLoss.
        patience: Épocas para early stopping.
        device: Dispositivo (cpu/cuda).

    Returns:
        Dicionário com histórico de loss de treino e validação.
    """
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Treino
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validação
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if (epoch + 1) % 10 == 0:
            logger.info("Epoch %d/%d — train_loss: %.4f | val_loss: %.4f", epoch + 1, epochs, train_loss, val_loss)

        if early_stopping.step(val_loss, model):
            logger.info("Early stopping na época %d", epoch + 1)
            early_stopping.restore(model)
            break

    return history


def predict_proba(model: nn.Module, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Retorna probabilidades de churn para os dados de entrada."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(X_tensor)
        proba = torch.sigmoid(logits).cpu().numpy()
    return proba
