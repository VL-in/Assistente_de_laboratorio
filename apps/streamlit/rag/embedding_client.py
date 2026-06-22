"""
Geração de embeddings RAG via sentence-transformers in-process.

Substitui o cliente HTTP do TEI (contêiner Docker separado) por uma instância
local do ``SentenceTransformer``, tornando a aplicação auto-contida e compatível
com deploys sem Docker Compose (ex.: Hugging Face Spaces).

O modelo ``intfloat/multilingual-e5-small`` exige prefixos assimétricos:
``passage:`` na indexação e ``query:`` nas buscas. O txtai aplica esses
prefixos via ``instructions`` em ``embeddings_config()`` antes de chamar
``embedding_transform``.
"""

from __future__ import annotations

import os
import threading

import numpy as np

EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-small"
# Caminho importável gravado no índice txtai (``Resolver`` do txtai na carga).
EMBEDDING_TRANSFORM_PATH = "rag.embedding_client.embedding_transform"

_DEFAULT_BATCH_SIZE = 32

_model_lock = threading.Lock()
_model_instance: object | None = None


def _get_model() -> object:
    """
    Retorna a instância singleton do SentenceTransformer, carregando na primeira chamada.

    Thread-safe via Lock: se duas threads chegarem simultaneamente na primeira
    chamada, apenas uma carrega o modelo; a outra aguarda e reutiliza a instância.
    O cache em ``HF_HOME`` / ``SENTENCE_TRANSFORMERS_HOME`` evita re-download.
    """
    global _model_instance
    if _model_instance is not None:
        return _model_instance
    with _model_lock:
        if _model_instance is None:
            from sentence_transformers import SentenceTransformer

            cache_dir = os.environ.get("SENTENCE_TRANSFORMERS_HOME") or os.environ.get("HF_HOME")
            _model_instance = SentenceTransformer(EMBEDDING_MODEL_ID, cache_folder=cache_dir)
    return _model_instance


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Gera embeddings para uma lista de textos via SentenceTransformer in-process.

    Retorna matriz ``(n, dim)`` em float32. Lista vazia retorna ``(0, 0)``.
    O tamanho de lote é controlado por ``EMBEDDING_BATCH_SIZE`` (padrão 32).
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    model = _get_model()
    batch_size = _batch_size()
    vectors = model.encode(  # type: ignore[attr-defined]
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.array(vectors, dtype=np.float32)


def embedding_transform(inputs: list[str] | str) -> np.ndarray:
    """
    Função de vetorização externa do txtai (``method=external``).

    Deve ser referenciada por **string** em ``embeddings_config()`` para que o
    txtai consiga resolver o callable ao carregar o índice do disco.
    """
    batch = [inputs] if isinstance(inputs, str) else list(inputs)
    return embed_texts(batch)


def _batch_size() -> int:
    raw = os.environ.get("EMBEDDING_BATCH_SIZE", str(_DEFAULT_BATCH_SIZE)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_BATCH_SIZE
