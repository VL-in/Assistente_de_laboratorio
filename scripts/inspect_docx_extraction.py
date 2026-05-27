"""
Script de inspeção: roda o pipeline real de extração ``.docx`` do projeto sobre
todos os documentos de ``PROJETOS_HOST_DIR`` e imprime um relatório de auditoria.

Uso (a partir da raiz do repositório):

    python scripts/inspect_docx_extraction.py

Para cada arquivo lista:
- número de parágrafos e tabelas detectados,
- as primeiras linhas serializadas das tabelas,
- ocorrências de termos-chave do laboratório (Tampão, Validade, Lote, NA).

Objetivo: confirmar se ``_extract_docx`` (em ``apps/streamlit/rag/extract.py``)
está percorrendo corretamente parágrafos, tabelas, sub-tabelas e controles de
conteúdo (``<w:sdt>``) dos documentos reais do laboratório.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Força UTF-8 no stdout/stderr para não quebrar em PowerShell/cp1252 quando
# imprimir caracteres como "→", "·" ou acentos vindos dos extratos.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

REPO_ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_APP_DIR = REPO_ROOT / "apps" / "streamlit"
sys.path.insert(0, str(STREAMLIT_APP_DIR))

# Importa o extrator real usado em produção (mesma função do pipeline RAG).
from rag.extract import _extract_docx  # noqa: E402
from rag.chunking import chunk_text  # noqa: E402

PROJETOS_DIR = Path(
    os.environ.get("PROJETOS_HOST_DIR")
    or "D:/Vanessa/AI_project/Projetos"
)

TERMOS_AUDITORIA = [
    "tampão",
    "tampao",
    "validade",
    "lote",
    "fabricante",
    "NA",
    "N/A",
    "amostra",
    "ELISA",
]


def listar_docx(raiz: Path) -> list[Path]:
    return sorted(raiz.rglob("*.docx"))


def relatorio_arquivo(path: Path) -> None:
    print("\n" + "=" * 88)
    print(f"ARQUIVO: {path}")
    print("=" * 88)

    outcome = _extract_docx(path, max_chars_total=2_000_000)
    print(f"detail        : {outcome.detail}")
    print(f"ok            : {outcome.ok}")
    print(f"len(text)     : {len(outcome.text)} chars")

    text = outcome.text or ""
    print(f"linhas totais : {text.count(chr(10)) + 1 if text else 0}")

    cab_legado = text.count("### Tabela de insumos / materiais")
    cab_dinamico = text.count("### Tabela — ")
    print(
        f"cabeçalhos de tabela: {cab_legado} (genérico) + "
        f"{cab_dinamico} (com título dinâmico)"
    )

    # Mostra um trecho da 1ª tabela (qualquer cabeçalho '### ')
    idx_tab = text.find("\n### ")
    if idx_tab == -1 and text.startswith("### "):
        idx_tab = 0
    if idx_tab != -1:
        amostra = text[idx_tab : idx_tab + 800]
        print("\n--- primeiras ~800 chars de uma tabela ---")
        print(amostra)
        print("--- fim do trecho ---")

    print("\n--- ocorrências de termos-chave ---")
    text_lower = text.lower()
    for termo in TERMOS_AUDITORIA:
        n = text_lower.count(termo.lower())
        if n:
            print(f"  {termo:20s} -> {n} ocorrência(s)")

    if any(t in text_lower for t in ("tampão", "tampao")):
        print("\n--- contexto em torno de 'Tampão' (1ª ocorrência) ---")
        for needle in ("tampão", "tampao"):
            idx = text_lower.find(needle)
            if idx != -1:
                print(text[max(0, idx - 80) : idx + 320])
                break

    chunks = chunk_text(text, max_chars=720, overlap=150)
    print(f"\nchunks gerados (max=720, overlap=150): {len(chunks)}")
    for i, ch in enumerate(chunks):
        low = ch.lower()
        if "tampão" in low or "tampao" in low or "validade" in low:
            print(f"  > chunk #{i} ({len(ch)} chars) -- contém termo-chave")
            print(f"    «{ch[:300]}{'…' if len(ch) > 300 else ''}»")


def main() -> int:
    if not PROJETOS_DIR.exists():
        print(f"ERRO: pasta de projetos não encontrada: {PROJETOS_DIR}")
        return 2
    docs = listar_docx(PROJETOS_DIR)
    print(f"Encontrados {len(docs)} arquivo(s) .docx em {PROJETOS_DIR}")
    for p in docs:
        try:
            relatorio_arquivo(p)
        except Exception as exc:
            print(f"\nERRO ao processar {p}: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
