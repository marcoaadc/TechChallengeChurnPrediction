# Tech Challenge - Churn Prediction

Projeto de previsão de Churn (cancelamento de clientes) utilizando uma Rede Neural do tipo MLP (Multi-Layer Perceptron) desenvolvida em PyTorch.

## Problema

Churn é a taxa de cancelamento de clientes em um determinado período. Prever quais clientes têm maior probabilidade de cancelar permite que a empresa tome ações preventivas de retenção, reduzindo custos e aumentando a receita.

Este projeto implementa uma API de classificação binária que recebe os dados de um cliente e retorna a probabilidade de churn.

## Stack

- **Modelo:** PyTorch (MLP)
- **Rastreamento de experimentos:** MLflow
- **API:** FastAPI + Uvicorn
- **Gerenciamento de dependências:** Poetry

## Instalação

```bash
# Instalar o Poetry (caso ainda não tenha)
pip install poetry

# Instalar as dependências do projeto
poetry install

# Ativar o ambiente virtual
poetry shell
```

## Estrutura do Projeto

```
├── data/
│   ├── raw/            # Dados brutos
│   └── processed/      # Dados processados
├── docs/               # Documentação
├── models/             # Modelos treinados
├── notebooks/          # Notebooks de análise e experimentação
├── src/                # Código-fonte
├── tests/              # Testes automatizados
├── pyproject.toml      # Dependências (Poetry)
└── README.md
```
