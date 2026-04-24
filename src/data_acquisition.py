"""Módulo de aquisição de dados: download do dataset Telco Customer Churn via kagglehub."""

import logging
import shutil
from pathlib import Path

import kagglehub

logger = logging.getLogger(__name__)

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
        logger.info("Dataset já presente em '%s'.", dest_path)
        return dest_path

    logger.info("Baixando dataset '%s' via kagglehub...", DATASET_HANDLE)
    cached_dir = Path(kagglehub.dataset_download(DATASET_HANDLE))

    csv_files = list(cached_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Nenhum CSV encontrado no cache do kagglehub: '{cached_dir}'")

    source_csv = csv_files[0]
    logger.info("Arquivo fonte encontrado: '%s'", source_csv.name)

    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_csv, dest_path)
    logger.info("Dataset copiado para '%s'", dest_path)

    return dest_path
