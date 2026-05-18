"""Construção e consulta do índice txtai com sentence-transformers."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from projects_loader import ProjectScan, ScannedFile

from .chunking import chunk_text
from .extract import extract_from_scanned_file
from .paths import ensure_txtai_parent_exists, txtai_index_path

# Modelo fixo do MVP (multilingual, ~768 dim; max 128 tokens — ver chunking na UI).
EMBEDDING_MODEL_ID = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def embeddings_config() -> dict:
    return {
        "path": EMBEDDING_MODEL_ID,
        "content": True,
    }


def index_ready() -> bool:
    """True se já existe um índice salvo em disco."""
    p = txtai_index_path()
    if not p.exists() or not p.is_dir():
        return False
    return any(p.iterdir())


def chunk_uid(sf: ScannedFile, chunk_index: int) -> str:
    raw = f"{sf.project_id}\x00{sf.relative_path}\x00{chunk_index}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass
class BuildStats:
    files_total: int = 0
    files_extracted: int = 0
    files_empty: int = 0
    chunks_written: int = 0
    errors: list[str] = field(default_factory=list)


Row = tuple[str, dict]


def _flatten_files(
    scans: list[ProjectScan],
    *,
    project_ids: set[str] | None,
) -> list[ScannedFile]:
    out: list[ScannedFile] = []
    for scan in scans:
        if project_ids is not None and scan.project_id not in project_ids:
            continue
        out.extend(scan.files)
    return out


def prepare_rows(
    scans: list[ProjectScan],
    *,
    project_ids: set[str] | None,
    max_chars: int,
    overlap: int,
    max_doc_chars: int,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[Row], BuildStats]:
    """Extrai texto, faz chunking e monta linhas ``(id, {text, ...})`` para o txtai."""
    stats = BuildStats()
    rows: list[Row] = []
    files = _flatten_files(scans, project_ids=project_ids)
    stats.files_total = len(files)

    for sf in files:
        label = f"{sf.project_id}/{sf.relative_path}"
        if progress:
            progress(label)

        outcome = extract_from_scanned_file(sf, max_chars_total=max_doc_chars)
        if not outcome.ok or not outcome.text.strip():
            stats.files_empty += 1
            if outcome.detail and not outcome.ok:
                stats.errors.append(f"{label}: {outcome.detail}")
            continue

        parts = chunk_text(outcome.text, max_chars=max_chars, overlap=overlap)
        if not parts:
            stats.files_empty += 1
            continue

        stats.files_extracted += 1

        for idx, part in enumerate(parts):
            uid = chunk_uid(sf, idx)
            # Busca txtai expõe sobretudo id/text/score — citações ficam no texto indexado.
            cited = (
                f"[Projeto: {sf.project_id}] [Arquivo: {sf.relative_path}] [Chunk {idx}]\n"
                f"{part}"
            )
            row_dict = {
                "text": cited,
                "project_id": sf.project_id,
                "relative_path": sf.relative_path,
                "chunk_index": idx,
                "extract_detail": outcome.detail,
            }
            rows.append((uid, row_dict))

    stats.chunks_written = len(rows)
    return rows, stats


def build_index(
    scans: list[ProjectScan],
    *,
    project_ids: set[str] | None,
    max_chars: int,
    overlap: int,
    max_doc_chars: int,
    batch_size: int,
    replace_existing: bool,
    progress: Callable[[str], None] | None = None,
) -> BuildStats:
    """
    Cria embeddings txtai e persiste em ``txtai_index_path()``.

    ``replace_existing``: remove o diretório do índice anterior antes de gravar.
    Lotes após o primeiro usam ``upsert`` — ``index()`` substitui o índice inteiro.
    """
    from txtai import Embeddings

    rows, stats = prepare_rows(
        scans,
        project_ids=project_ids,
        max_chars=max_chars,
        overlap=overlap,
        max_doc_chars=max_doc_chars,
        progress=progress,
    )

    if not rows:
        stats.errors.append("Nenhum chunk gerado — verifique arquivos vazios ou filtros de projeto.")
        return stats

    ensure_txtai_parent_exists()
    index_path = txtai_index_path()

    if replace_existing and index_path.exists():
        shutil.rmtree(index_path)

    index_path.parent.mkdir(parents=True, exist_ok=True)

    emb = Embeddings(embeddings_config())
    append_mode = not replace_existing and index_ready()

    try:
        if append_mode:
            emb.load(str(index_path))

        batch: list[Row] = []
        n = len(rows)
        first_batch = True
        for i, row in enumerate(rows):
            batch.append(row)
            if len(batch) >= batch_size or i == n - 1:
                if first_batch and not append_mode:
                    emb.index(batch)
                    first_batch = False
                else:
                    emb.upsert(batch)
                batch.clear()
                if progress:
                    progress(f"indexando… {min(i + 1, n)}/{n} chunks")

        emb.save(str(index_path))
    finally:
        emb.close()

    return stats


def index_mtime() -> float:
    """Timestamp do índice salvo (para cache do Streamlit); 0 se inexistente."""
    p = txtai_index_path()
    if not p.exists() or not p.is_dir():
        return 0.0
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def format_search_results(raw: object) -> list[dict]:
    """
    Normaliza saída de ``Embeddings.search`` (com ``content=True`` → dicts;
    sem content → tuplas ``(id, score)``).
    """
    if not raw:
        return []
    normalized: list[dict] = []
    for r in raw:
        if isinstance(r, dict):
            normalized.append(dict(r))
        elif isinstance(r, (list, tuple)) and len(r) >= 2:
            normalized.append({"id": r[0], "score": r[1]})
    return normalized


def search_chunks(query: str, limit: int) -> list[dict]:
    """Busca semântica (abre e fecha o índice — útil para scripts / testes isolados)."""
    from txtai import Embeddings

    if not index_ready():
        return []

    q = (query or "").strip()
    if not q:
        return []

    path = txtai_index_path()
    emb = Embeddings()
    try:
        emb.load(str(path))
        return format_search_results(emb.search(q, limit))
    finally:
        emb.close()


def search_with_backend(backend: object, query: str, limit: int) -> list[dict]:
    """Busca usando instância já carregada (ex.: cache ``st.cache_resource``)."""
    q = (query or "").strip()
    if not q:
        return []
    return format_search_results(backend.search(q, limit))


def format_context_for_llm(hits: list[dict], *, max_chars: int = 12000) -> str:
    """Monta bloco de contexto citável para o prompt do chat."""
    lines: list[str] = []
    used = 0
    for i, h in enumerate(hits, start=1):
        body = (h.get("text") or "").strip()
        if not body:
            continue
        block = f"### Evidência [{i}]\n{body}\n\n"
        if used + len(block) > max_chars:
            lines.append(f"(… contexto truncado em {max_chars} caracteres …)")
            break
        lines.append(block)
        used += len(block)
    return "\n".join(lines).strip()
