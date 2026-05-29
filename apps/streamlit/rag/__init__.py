"""
Pacote RAG — pipeline local de indexação semântica e recuperação de contexto.

Subsistemas exportados
----------------------
Indexação (``build_index``, ``BuildStats``)
    Ponto de entrada único para criar ou atualizar o índice txtai a partir do
    inventário produzido por ``projects_loader``. Consumido pela aba
    "Indexação RAG" do ``app.py``.

Busca semântica (``search_chunks``, ``search_with_backend``)
    Consulta o índice por similaridade semântica. ``search_with_backend``
    recebe a instância já carregada do cache do Streamlit para evitar
    recarregar o modelo a cada interação. Consumido pelo chat e pela aba
    "Teste RAG (dev)" do ``app.py``.

Formatação de contexto (``format_context_for_llm``)
    Converte os hits de busca em um bloco de texto citável injetado no
    system prompt do LLM. Consumido pelo chat do ``app.py``.

Utilitários de estado (``index_ready``, ``index_mtime``, ``manifest_exists``,
``manifest_path``, ``txtai_data_root``, ``txtai_index_path``)
    Funções de consulta ao estado do índice em disco. Usadas na UI para exibir
    indicadores de status e como chave de invalidação do cache do Streamlit.

Constantes (``EMBEDDING_MODEL_ID``, ``ENV_TXTAI_DIR``)
    Identificador do modelo e nome da variável de ambiente. Exibidos na UI de
    diagnóstico e na aba de indexação.
"""

from .index_txtai import (
    EMBEDDING_MODEL_ID,
    BuildStats,
    build_index,
    format_context_for_llm,
    index_mtime,
    index_ready,
    search_chunks,
    search_with_backend,
)
from .manifest import manifest_exists, manifest_path
from .paths import ENV_TXTAI_DIR, txtai_data_root, txtai_index_path
from .rerank import (
    RERANKER_MODEL_ID,
    default_retrieve_k,
    env_rerank_enabled,
    load_reranker,
    rerank_hits,
)

__all__ = [
    "EMBEDDING_MODEL_ID",
    "RERANKER_MODEL_ID",
    "BuildStats",
    "ENV_TXTAI_DIR",
    "build_index",
    "default_retrieve_k",
    "env_rerank_enabled",
    "format_context_for_llm",
    "index_mtime",
    "index_ready",
    "load_reranker",
    "manifest_exists",
    "manifest_path",
    "rerank_hits",
    "search_chunks",
    "search_with_backend",
    "txtai_data_root",
    "txtai_index_path",
]
