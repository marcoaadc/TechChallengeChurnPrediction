"""Configuração centralizada de logging estruturado em JSON."""

import logging
import sys

from pythonjsonlogger.json import JsonFormatter


def setup_logging(level: int = logging.INFO) -> None:
    """Configura o root logger para saída JSON estruturada.

    Deve ser chamado uma vez na inicialização da aplicação (API ou script).
    Módulos individuais continuam usando `logging.getLogger(__name__)`.
    """
    handler = logging.StreamHandler(sys.stdout)
    formatter = JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level"},
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
