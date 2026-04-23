from pathlib import Path

import pandas as pd

from src.data_acquisition import download_telco_churn


def load_data(filepath: str) -> pd.DataFrame:
    """Carrega um arquivo CSV e retorna um DataFrame.

    Args:
        filepath: Caminho para o arquivo CSV a ser carregado.

    Returns:
        DataFrame com os dados do CSV.

    Raises:
        FileNotFoundError: Se o arquivo não for encontrado no caminho informado.
    """
    path = Path(filepath)
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(
            f"[ERRO] Arquivo não encontrado: '{path.resolve()}'\n"
            f"Verifique se o CSV está no caminho correto (ex: data/raw/)."
        )
        raise

    print(f"Dataset carregado com sucesso: {df.shape[0]} linhas x {df.shape[1]} colunas")
    return df


def load_telco_churn(force_download: bool = False) -> pd.DataFrame:
    """Baixa (se necessário) e carrega o dataset Telco Customer Churn.

    Args:
        force_download: Se True, baixa novamente mesmo que o arquivo já exista.

    Returns:
        DataFrame com os dados do Telco Customer Churn.
    """
    csv_path = download_telco_churn(force=force_download)
    return load_data(str(csv_path))
