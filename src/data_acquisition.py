"""Módulo de aquisição de dados: download do dataset Telco Customer Churn via kagglehub."""

from pathlib import Path
import shutil

import kagglehub

DATASET_HANDLE = "blastchar/telco-customer-churn"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
EXPECTED_FILENAME = "telco_churn.csv"


def download_telco_churn(
    destination_dir: Path = RAW_DATA_DIR,
    filename: str = EXPECTED_FILENAME,
    force: bool = False,
) -> Path:
    """Baixa o dataset Telco Customer Churn e coloca em data/raw/.

    Args:
        destination_dir: Diretório de destino do CSV.
        filename: Nome do arquivo de destino.
        force: Se True, baixa novamente mesmo que o arquivo já exista.

    Returns:
        Path para o arquivo CSV baixado.
    """
    dest_path = destination_dir / filename

    if dest_path.exists() and not force:
        print(f"Dataset já presente em '{dest_path}'. Use force=True para baixar novamente.")
        return dest_path

    print(f"Baixando dataset '{DATASET_HANDLE}' via kagglehub...")
    cached_dir = Path(kagglehub.dataset_download(DATASET_HANDLE))

    csv_files = list(cached_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nenhum CSV encontrado no cache do kagglehub: '{cached_dir}'")

    source_csv = csv_files[0]
    print(f"Arquivo fonte encontrado: '{source_csv.name}'")

    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_csv, dest_path)
    print(f"Dataset copiado para '{dest_path}'")

    return dest_path
