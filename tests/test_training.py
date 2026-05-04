"""Testes unitários para o módulo src.training."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.model import ChurnMLP
from src.training import ChurnDataset, EarlyStopping, find_optimal_threshold, predict_proba, train_model


class TestChurnDataset:
    """Testes do ChurnDataset."""

    def test_len(self):
        """__len__ retorna o número de amostras."""
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.random.randint(0, 2, size=50).astype(np.float32)
        ds = ChurnDataset(X, y)
        assert len(ds) == 50

    def test_getitem_shapes(self):
        """__getitem__ retorna tensores com shapes corretos."""
        X = np.random.randn(20, 5).astype(np.float32)
        y = np.random.randint(0, 2, size=20).astype(np.float32)
        ds = ChurnDataset(X, y)
        x_sample, y_sample = ds[0]
        assert x_sample.shape == torch.Size([5])
        assert y_sample.shape == torch.Size([])

    def test_dtype_is_float32(self):
        """Tensores internos devem ser float32."""
        X = np.random.randn(10, 3).astype(np.float64)
        y = np.random.randint(0, 2, size=10).astype(np.int64)
        ds = ChurnDataset(X, y)
        assert ds.X.dtype == torch.float32
        assert ds.y.dtype == torch.float32


class TestEarlyStopping:
    """Testes do EarlyStopping."""

    def test_no_stop_when_improving(self):
        """Não para enquanto a loss de validação melhora."""
        model = ChurnMLP(input_dim=5, hidden_dims=[8])
        es = EarlyStopping(patience=3, min_delta=0.01)
        assert es.step(1.0, model) is False
        assert es.step(0.9, model) is False
        assert es.step(0.8, model) is False

    def test_stops_after_patience(self):
        """Para após patience épocas sem melhoria."""
        model = ChurnMLP(input_dim=5, hidden_dims=[8])
        es = EarlyStopping(patience=2, min_delta=0.0)
        es.step(0.5, model)  # nova melhor loss
        es.step(0.6, model)  # piora — counter=1
        stopped = es.step(0.7, model)  # piora — counter=2 >= patience
        assert stopped is True

    def test_restore_best_state(self):
        """restore() carrega o state_dict da melhor época."""
        model = ChurnMLP(input_dim=5, hidden_dims=[8])
        es = EarlyStopping(patience=5)

        # Registrar o melhor estado
        es.step(1.0, model)
        best_weight = model.network[0].weight.data.clone()

        # Modificar o modelo
        with torch.no_grad():
            model.network[0].weight.fill_(999.0)

        es.restore(model)
        assert torch.allclose(model.network[0].weight.data, best_weight)


class TestTrainModel:
    """Testes da função train_model com dados sintéticos pequenos."""

    @pytest.fixture()
    def small_loaders(self):
        """Cria DataLoaders pequenos para teste rápido."""
        np.random.seed(42)
        X_train = np.random.randn(40, 5).astype(np.float32)
        y_train = (X_train[:, 0] > 0).astype(np.float32)
        X_val = np.random.randn(10, 5).astype(np.float32)
        y_val = (X_val[:, 0] > 0).astype(np.float32)

        train_ds = ChurnDataset(X_train, y_train)
        val_ds = ChurnDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=16)
        return train_loader, val_loader

    def test_returns_history_dict(self, small_loaders):
        """train_model retorna dicionário com train_loss e val_loss."""
        train_loader, val_loader = small_loaders
        model = ChurnMLP(input_dim=5, hidden_dims=[8])
        history = train_model(model, train_loader, val_loader, epochs=5, patience=3)
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) > 0
        assert len(history["val_loss"]) > 0

    def test_loss_decreases(self, small_loaders):
        """A menor loss de treino deve ser menor que a primeira."""
        train_loader, val_loader = small_loaders
        model = ChurnMLP(input_dim=5, hidden_dims=[8])
        history = train_model(model, train_loader, val_loader, epochs=30, patience=25)
        assert min(history["train_loss"]) < history["train_loss"][0]

    def test_early_stopping_triggers(self, small_loaders):
        """Com patience=1 e epochs alto, early stopping deve encurtar o treino."""
        train_loader, val_loader = small_loaders
        model = ChurnMLP(input_dim=5, hidden_dims=[8])
        history = train_model(model, train_loader, val_loader, epochs=200, patience=1)
        # Com patience=1, deve parar bem antes de 200 épocas
        assert len(history["train_loss"]) < 200


class TestTrainModelWithScheduler:
    """Testes do train_model com LR scheduler."""

    @pytest.fixture()
    def small_loaders(self):
        np.random.seed(42)
        X_train = np.random.randn(40, 5).astype(np.float32)
        y_train = (X_train[:, 0] > 0).astype(np.float32)
        X_val = np.random.randn(10, 5).astype(np.float32)
        y_val = (X_val[:, 0] > 0).astype(np.float32)
        train_ds = ChurnDataset(X_train, y_train)
        val_ds = ChurnDataset(X_val, y_val)
        return DataLoader(train_ds, batch_size=16, shuffle=True), DataLoader(val_ds, batch_size=16)

    def test_train_with_scheduler(self, small_loaders):
        """train_model com scheduler executa sem erro."""
        train_loader, val_loader = small_loaders
        model = ChurnMLP(input_dim=5, hidden_dims=[8])
        history = train_model(
            model,
            train_loader,
            val_loader,
            epochs=10,
            patience=5,
            scheduler_patience=3,
            scheduler_factor=0.5,
        )
        assert len(history["train_loss"]) > 0

    def test_train_with_weight_decay(self, small_loaders):
        """train_model com weight_decay executa sem erro."""
        train_loader, val_loader = small_loaders
        model = ChurnMLP(input_dim=5, hidden_dims=[8])
        history = train_model(
            model,
            train_loader,
            val_loader,
            epochs=5,
            patience=3,
            weight_decay=1e-4,
        )
        assert len(history["train_loss"]) > 0


class TestPredictProba:
    """Testes da função predict_proba."""

    def test_output_shape(self):
        """predict_proba retorna array com shape (n_samples,)."""
        model = ChurnMLP(input_dim=5, hidden_dims=[8])
        X = np.random.randn(10, 5).astype(np.float32)
        proba = predict_proba(model, X)
        assert proba.shape == (10,)

    def test_output_range(self):
        """Probabilidades devem estar entre 0 e 1."""
        model = ChurnMLP(input_dim=5, hidden_dims=[8])
        X = np.random.randn(20, 5).astype(np.float32)
        proba = predict_proba(model, X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)


class TestFindOptimalThreshold:
    """Testes da função find_optimal_threshold."""

    def test_returns_valid_threshold(self):
        """Threshold retornado deve estar entre 0 e 1."""
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.4, 0.15, 0.6, 0.85])
        threshold, cost = find_optimal_threshold(y_true, y_proba)
        assert 0.0 < threshold < 1.0
        assert cost >= 0

    def test_lower_threshold_with_high_fn_cost(self):
        """Com FN muito caro, threshold ótimo deve ser mais baixo (detectar mais churn)."""
        y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        y_proba = np.array([0.05, 0.12, 0.18, 0.28, 0.42, 0.52, 0.38, 0.62, 0.78, 0.92])
        t_high_fn, _ = find_optimal_threshold(y_true, y_proba, cost_fn=10000, cost_fp=1)
        t_low_fn, _ = find_optimal_threshold(y_true, y_proba, cost_fn=1, cost_fp=10000)
        assert t_high_fn <= t_low_fn

    def test_returns_tuple(self):
        """Retorna tupla (threshold, custo)."""
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([0.2, 0.8, 0.3, 0.7])
        result = find_optimal_threshold(y_true, y_proba)
        assert isinstance(result, tuple)
        assert len(result) == 2
