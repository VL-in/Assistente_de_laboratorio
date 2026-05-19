"""
Resolução de caminhos persistentes para o pacote RAG.

Todos os módulos do pacote `rag` que precisam ler ou gravar arquivos em disco
chamam funções deste módulo — nunca constroem caminhos diretamente. Isso
centraliza a lógica de detecção de ambiente (Docker vs. dev local) em um único
lugar.

Cadeia de prioridade para localizar o diretório de dados txtai:
  1. Variável de ambiente ``ASSISTENTE_TXTAI_DIR`` (explícita — ganha sempre).
  2. ``/data/txtai`` quando rodando dentro de um contêiner Docker
     (volume declarado no Compose).
  3. ``.txtai_data/`` ao lado do diretório ``apps/streamlit/``, para
     desenvolvimento local sem Docker.
"""

from __future__ import annotations

import os
from pathlib import Path

from projects_loader import running_inside_docker

# ── Constante de ambiente ───────────────────────────────────────────────────

ENV_TXTAI_DIR = "ASSISTENTE_TXTAI_DIR"


# ── Resolução de caminhos ───────────────────────────────────────────────────

def txtai_data_root() -> Path:
    """
    Retorna o diretório-raiz onde todos os dados do txtai são armazenados.

    Dentro dessa raiz ficam o índice de embeddings (``embeddings_index/``) e o
    manifesto de hashes (``index_manifest.json``). Qualquer função do pacote que
    precise gravar ou ler dados em disco deve chamar esta função para obter o
    caminho base, garantindo consistência entre ambientes.

    Resolução (em ordem):
    - ``ASSISTENTE_TXTAI_DIR`` definido → usa esse valor.
    - Dentro de Docker sem a variável → ``/data/txtai`` (volume do Compose).
    - Dev local → ``.txtai_data/`` relativo ao diretório ``apps/streamlit/``.
    """
    raw = os.environ.get(ENV_TXTAI_DIR, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    if running_inside_docker():
        return Path("/data/txtai").resolve()
    # Pasta local ignorada pelo git (.gitignore recomendado: `.txtai_data/`)
    return (Path(__file__).resolve().parent.parent / ".txtai_data").resolve()


def txtai_index_path() -> Path:
    """
    Retorna o caminho do diretório de índice de embeddings.

    Este caminho é passado diretamente a ``Embeddings.save()`` e
    ``Embeddings.load()`` do txtai. O txtai exige um diretório (não um arquivo);
    ele cria e gerencia os arquivos internos.
    """
    return txtai_data_root() / "embeddings_index"


def ensure_txtai_parent_exists() -> None:
    """
    Cria o diretório-raiz txtai caso ainda não exista (equivalente a mkdir -p).

    Deve ser chamada antes de qualquer ``Embeddings.save()`` para garantir que o
    caminho pai esteja presente. Falhas de permissão propagam ``OSError``.
    """
    txtai_data_root().mkdir(parents=True, exist_ok=True)
