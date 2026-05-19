"""Divisão de texto em chunks com sobreposição — alinhado ao limite do modelo de embedding."""

from __future__ import annotations


def chunk_text(text: str, *, max_chars: int, overlap: int) -> list[str]:
    """
    Corta texto em segmentos de até ``max_chars`` caracteres com sobreposição ``overlap``.

    O modelo ``paraphrase-multilingual-mpnet-base-v2`` usa até **128 tokens**; em pt-BR,
    ~450–550 caracteres costuma ser seguro. Os defaults na UI ficam conservadores.
    """
    cleaned = text.strip()
    if not cleaned:
        return []
    if max_chars < 80:
        max_chars = 80
    if overlap < 0:
        overlap = 0
    if overlap >= max_chars:
        overlap = max(0, max_chars // 8)

    if len(cleaned) <= max_chars:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    n = len(cleaned)
    while start < n:
        end = min(start + max_chars, n)
        # Recua até a última quebra de palavra para não cortar no meio de um token.
        # Busca somente na segunda metade do chunk para evitar pedaços muito pequenos.
        if end < n:
            boundary = cleaned.rfind(" ", start + max_chars // 2, end)
            if boundary == -1:
                boundary = cleaned.rfind("\n", start + max_chars // 2, end)
            if boundary > start:
                end = boundary + 1  # inclui o delimitador; .strip() remove depois
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end  # garante avanço mesmo com overlap grande ou trecho vazio após strip
        start = next_start

    return chunks
