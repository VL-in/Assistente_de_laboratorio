"""
Cliente HTTP para o serviço de embeddings (TEI) no contêiner Docker.

O modelo ``intfloat/multilingual-e5-small`` exige prefixos assimétricos:
``passage:`` na indexação e ``query:`` nas buscas. O txtai aplica esses
prefixos via ``instructions`` em ``embeddings_config()`` antes de chamar
``transform``.
"""

from __future__ import annotations

import os
import numpy as np
import requests

EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-small"
# Caminho importável gravado no índice txtai (``Resolver`` do txtai na carga).
EMBEDDING_TRANSFORM_PATH = "rag.embedding_client.embedding_transform"
ENV_EMBEDDING_SERVICE_URL = "EMBEDDING_SERVICE_URL"
ENV_EMBEDDING_HTTP_BATCH_SIZE = "EMBEDDING_HTTP_BATCH_SIZE"
DEFAULT_EMBEDDING_SERVICE_URL = "http://embeddings:80"
# TEI rejeita lotes grandes (413) — sub-lotes HTTP independentes do batch do txtai.
DEFAULT_EMBEDDING_HTTP_BATCH_SIZE = 16
_EMBED_TIMEOUT_S = float(os.environ.get("EMBEDDING_TIMEOUT_S", "120"))


def embedding_http_batch_size() -> int:
    """Quantos textos enviar por POST ``/embed`` (limite do TEI)."""
    raw = os.environ.get(
        ENV_EMBEDDING_HTTP_BATCH_SIZE,
        str(DEFAULT_EMBEDDING_HTTP_BATCH_SIZE),
    ).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_EMBEDDING_HTTP_BATCH_SIZE


def embedding_service_url() -> str:
    """URL base do TEI (sem barra final)."""
    raw = os.environ.get(ENV_EMBEDDING_SERVICE_URL, DEFAULT_EMBEDDING_SERVICE_URL).strip()
    return raw.rstrip("/")


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Gera embeddings via POST ``/embed`` do Text Embeddings Inference.

    O txtai pode pedir dezenas de textos de uma vez (ex.: lote 64 na UI).
    O TEI limita inputs por requisição (``max-client-batch-size``) e o corpo
    HTTP (``payload-limit``), retornando **413** se o lote for grande demais.
    Por isso fragmentamos em sub-lotes HTTP menores e concatenamos os vetores.

    Retorna matriz ``(n, dim)`` em float32.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    url = f"{embedding_service_url()}/embed"
    http_batch = embedding_http_batch_size()
    all_vectors: list[list[float]] = []

    for start in range(0, len(texts), http_batch):
        chunk = texts[start : start + http_batch]
        response = requests.post(
            url,
            json={"inputs": chunk},
            timeout=_EMBED_TIMEOUT_S,
        )
        response.raise_for_status()
        batch_vectors = response.json()
        if not isinstance(batch_vectors, list) or len(batch_vectors) != len(chunk):
            raise RuntimeError(
                f"TEI devolveu {len(batch_vectors) if isinstance(batch_vectors, list) else type(batch_vectors)!r} "
                f"vetores para {len(chunk)} entradas."
            )
        all_vectors.extend(batch_vectors)

    return np.array(all_vectors, dtype=np.float32)


def embedding_transform(inputs: list[str] | str) -> np.ndarray:
    """
    Função de vetorização externa do txtai (``method=external``).

    Deve ser referenciada por **string** em ``embeddings_config()`` para que o
    txtai consiga resolver o callable ao carregar o índice do disco.
    """
    batch = [inputs] if isinstance(inputs, str) else list(inputs)
    return embed_texts(batch)
