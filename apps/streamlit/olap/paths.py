"""
Resolução de caminhos persistentes para o pacote OLAP (DuckDB).

Todos os módulos do pacote ``olap`` que precisam ler ou gravar o arquivo
``.duckdb`` chamam funções deste módulo — nunca constroem caminhos
diretamente. Isso centraliza a detecção de ambiente (Docker vs. dev local).

Cadeia de prioridade para o diretório de dados DuckDB:
  1. Variável de ambiente ``ASSISTENTE_DUCKDB_DIR`` (explícita — ganha sempre).
  2. ``/data/duckdb`` quando rodando dentro de um contêiner Docker
     (volume declarado no Compose).
  3. ``.duckdb_data/`` ao lado do diretório ``apps/streamlit/``, para
     desenvolvimento local sem Docker.
"""

from __future__ import annotations

import os
from pathlib import Path

from projects_loader import running_inside_docker

ENV_DUCKDB_DIR = "ASSISTENTE_DUCKDB_DIR"
DUCKDB_DATABASE_FILENAME = "assistente_lab.duckdb"


def duckdb_data_root() -> Path:
    """
    Retorna o diretório onde o arquivo ``assistente_lab.duckdb`` é armazenado.

    Resolução (em ordem):
    - ``ASSISTENTE_DUCKDB_DIR`` definido → usa esse valor.
    - Dentro de Docker sem a variável → ``/data/duckdb`` (volume do Compose).
    - Dev local → ``.duckdb_data/`` relativo a ``apps/streamlit/``.
    """
    raw = os.environ.get(ENV_DUCKDB_DIR, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    if running_inside_docker():
        return Path("/data/duckdb").resolve()
    return (Path(__file__).resolve().parent.parent / ".duckdb_data").resolve()


def duckdb_database_path() -> Path:
    """Caminho completo do banco DuckDB em arquivo único no volume."""
    return duckdb_data_root() / DUCKDB_DATABASE_FILENAME


def ensure_duckdb_parent_exists() -> None:
    """Cria o diretório-raiz DuckDB caso ainda não exista (mkdir -p)."""
    duckdb_data_root().mkdir(parents=True, exist_ok=True)
