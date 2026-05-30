"""
Divisão de texto em chunks com sobreposição para indexação semântica.

O modelo de embedding RAG (``intfloat/multilingual-e5-small``) processa até
**512 tokens** por entrada. Em português, 1 token ≈ 4–5 caracteres; o padrão
na UI é **720 caracteres** (~150–180 tokens) com sobreposição de 150 — bem
dentro da janela do modelo e alinhado ao tamanho de chunk já usado no projeto.

A sobreposição (``overlap``) cria uma janela deslizante entre chunks consecutivos:
o final do chunk N reaparece no início do chunk N+1. Isso preserva o contexto
em torno dos limites de corte, evitando que informações-chave fiquem isoladas
em um único chunk que talvez não seja recuperado.
"""

from __future__ import annotations


def chunk_text(text: str, *, max_chars: int, overlap: int) -> list[str]:
    """
    Divide ``text`` em segmentos de até ``max_chars`` caracteres com sobreposição
    ``overlap``.

    Parâmetros
    ----------
    text:
        Texto bruto já extraído do documento.
    max_chars:
        Limite superior de caracteres por chunk. Valores abaixo de 80 são
        elevados a 80 para evitar chunks sem sentido semântico.
    overlap:
        Quantos caracteres do fim do chunk anterior são repetidos no início do
        próximo. Se ``overlap >= max_chars``, é reduzido a ``max_chars // 8``
        para garantir progresso.

    Retorna
    -------
    list[str]
        Lista de segmentos, cada um com no máximo ``max_chars`` chars (após
        ``.strip()``). Retorna lista vazia se ``text`` for vazio.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    # Garante valores mínimos que produzem chunks com sentido semântico.
    if max_chars < 80:
        max_chars = 80
    if overlap < 0:
        overlap = 0
    # Overlap igual ou maior que o chunk impediria o avanço do ponteiro.
    if overlap >= max_chars:
        overlap = max(0, max_chars // 8)

    # Texto curto o suficiente para caber em um único chunk.
    if len(cleaned) <= max_chars:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    n = len(cleaned)

    while start < n:
        end = min(start + max_chars, n)

        # Tenta recuar o corte até a última fronteira de palavra (espaço ou
        # quebra de linha) dentro da segunda metade do chunk. Buscar na segunda
        # metade evita criar um chunk tão curto que perca significado semântico.
        # Se nenhuma fronteira for encontrada (palavra muito longa), o corte
        # permanece na posição exata — comportamento inevitável.
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

        # Aplica sobreposição: o próximo chunk começa `overlap` chars antes do
        # fim do atual. O guard abaixo garante que o ponteiro sempre avança,
        # mesmo que overlap seja grande ou que o .strip() tenha consumido chars.
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks
