"""Testes unitários para o módulo src.model (ChurnMLP)."""

import torch

from src.model import ChurnMLP


class TestChurnMLPInstantiation:
    """Testes de criação do modelo."""

    def test_default_hidden_dims(self):
        """Modelo com hidden_dims padrão ([64, 32]) instancia sem erro."""
        model = ChurnMLP(input_dim=10)
        assert model is not None

    def test_custom_hidden_dims(self):
        """Modelo com hidden_dims customizado instancia sem erro."""
        model = ChurnMLP(input_dim=20, hidden_dims=[128, 64, 32], dropout=0.5)
        assert model is not None

    def test_single_hidden_layer(self):
        """Modelo com uma única camada oculta instancia sem erro."""
        model = ChurnMLP(input_dim=5, hidden_dims=[16])
        assert model is not None


class TestChurnMLPForwardPass:
    """Testes do forward pass."""

    def test_output_shape_single_sample(self):
        """Forward pass com batch=1 retorna shape correto (scalar por amostra)."""
        model = ChurnMLP(input_dim=10)
        model.eval()
        x = torch.randn(1, 10)
        with torch.no_grad():
            out = model(x)
        assert out.shape == torch.Size([1])

    def test_output_shape_batch(self):
        """Forward pass com batch > 1 retorna shape (batch_size,)."""
        model = ChurnMLP(input_dim=15, hidden_dims=[32, 16])
        model.eval()
        x = torch.randn(8, 15)
        with torch.no_grad():
            out = model(x)
        assert out.shape == torch.Size([8])

    def test_output_is_logits(self):
        """Saída são logits (valores reais, sem sigmoid aplicado)."""
        model = ChurnMLP(input_dim=10)
        model.eval()
        x = torch.randn(16, 10)
        with torch.no_grad():
            out = model(x)
        # Logits podem ser negativos — se fosse sigmoid, estariam em [0, 1].
        # Com 16 amostras aleatórias, é muito improvável que todos caiam em [0,1].
        assert out.dtype == torch.float32

    def test_sigmoid_output_range(self):
        """Após sigmoid, saída fica no intervalo [0, 1]."""
        model = ChurnMLP(input_dim=10)
        model.eval()
        x = torch.randn(32, 10)
        with torch.no_grad():
            proba = torch.sigmoid(model(x))
        assert proba.min().item() >= 0.0
        assert proba.max().item() <= 1.0

    def test_deterministic_eval_mode(self):
        """Duas chamadas em eval mode produzem saída idêntica."""
        model = ChurnMLP(input_dim=10, dropout=0.5)
        model.eval()
        x = torch.randn(4, 10)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)
