"""Construção e consulta do índice txtai com sentence-transformers."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from projects_loader import ProjectScan, ScannedFile, compute_file_sha256

from .chunking import chunk_text
from .extract import extract_from_scanned_file
from .manifest import (
    EXTRACTION_LOGIC_VERSION,
    IndexManifest,
    chunking_signature,
    file_index_key,
    load_manifest,
    manifest_exists,
    remove_manifest_file,
    save_manifest,
)
from .paths import ensure_txtai_parent_exists, txtai_index_path

# Modelo fixo do MVP (multilingual, ~768 dim; max 128 tokens — ver chunking na UI).
EMBEDDING_MODEL_ID = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

Row = tuple[str, dict]


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
    files_skipped_unchanged: int = 0
    files_reindexed: int = 0
    files_removed: int = 0
    chunks_written: int = 0
    chunks_deleted: int = 0
    incremental: bool = False
    errors: list[str] = field(default_factory=list)


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


def resolve_content_hash(sf: ScannedFile) -> tuple[str | None, str | None]:
    """
    Retorna ``(hash, erro)`` — hash do scan ou calculado agora; erro se ilegível.
    """
    if sf.content_hash_sha256:
        return sf.content_hash_sha256, None
    try:
        return compute_file_sha256(sf.absolute_path), None
    except OSError as exc:
        return None, str(exc)


def rows_for_file(
    sf: ScannedFile,
    *,
    max_chars: int,
    overlap: int,
    max_doc_chars: int,
) -> tuple[list[Row], bool, str | None]:
    """
  Retorna (linhas para txtai, teve_texto, mensagem_erro).
    ``teve_texto`` False = arquivo vazio ou falha de extração.
    """
    label = f"{sf.project_id}/{sf.relative_path}"
    outcome = extract_from_scanned_file(sf, max_chars_total=max_doc_chars)
    if not outcome.text.strip():
        err = f"{label}: sem texto extraível"
        if outcome.detail:
            err += f" ({outcome.detail})"
        return [], False, err
    if not outcome.ok:
        return [], False, f"{label}: {outcome.detail}"

    parts = chunk_text(outcome.text, max_chars=max_chars, overlap=overlap)
    if not parts:
        return [], False, None

    rows: list[Row] = []
    for idx, part in enumerate(parts):
        uid = chunk_uid(sf, idx)
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

    return rows, True, None


def prepare_rows(
    scans: list[ProjectScan],
    *,
    project_ids: set[str] | None,
    max_chars: int,
    overlap: int,
    max_doc_chars: int,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[Row], BuildStats]:
    """Extrai texto, faz chunking e monta linhas ``(id, {text, ...})`` para o txtai (todos os arquivos)."""
    stats = BuildStats()
    rows: list[Row] = []
    files = _flatten_files(scans, project_ids=project_ids)
    stats.files_total = len(files)

    for sf in files:
        label = f"{sf.project_id}/{sf.relative_path}"
        if progress:
            progress(label)

        file_rows, had_text, err = rows_for_file(
            sf,
            max_chars=max_chars,
            overlap=overlap,
            max_doc_chars=max_doc_chars,
        )
        if err:
            stats.errors.append(err)
        if not had_text:
            stats.files_empty += 1
            continue

        stats.files_extracted += 1
        rows.extend(file_rows)

    stats.chunks_written = len(rows)
    return rows, stats


def _delete_ids_batched(
    emb: object,
    ids: list[str],
    *,
    batch_size: int,
    errors: list[str] | None = None,
) -> int:
    """Remove IDs duplicados e ignora falhas por lote (registra em ``errors``)."""
    if not ids:
        return 0
    unique_ids = list(dict.fromkeys(ids))
    deleted = 0
    for i in range(0, len(unique_ids), batch_size):
        batch = unique_ids[i : i + batch_size]
        try:
            emb.delete(batch)  # type: ignore[attr-defined]
            deleted += len(batch)
        except Exception as exc:  # noqa: BLE001
            if errors is not None:
                errors.append(f"Falha ao remover {len(batch)} chunk(s) do índice: {exc}")
    return deleted


def _upsert_rows_batched(
    emb: object,
    rows: list[Row],
    *,
    batch_size: int,
    use_initial_index: bool,
    progress: Callable[[str], None] | None,
) -> None:
    if not rows:
        return
    n = len(rows)
    batch: list[Row] = []
    first_batch = use_initial_index
    for i, row in enumerate(rows):
        batch.append(row)
        if len(batch) >= batch_size or i == n - 1:
            if first_batch:
                emb.index(batch)  # type: ignore[attr-defined]
                first_batch = False
            else:
                emb.upsert(batch)  # type: ignore[attr-defined]
            batch.clear()
            if progress:
                progress(f"indexando… {min(i + 1, n)}/{n} chunks")


def build_index(
    scans: list[ProjectScan],
    *,
    project_ids: set[str] | None,
    max_chars: int,
    overlap: int,
    max_doc_chars: int,
    batch_size: int,
    replace_existing: bool,
    incremental: bool | None = None,
    progress: Callable[[str], None] | None = None,
) -> BuildStats:
    """
    Cria ou atualiza embeddings txtai em ``txtai_index_path()``.

    - ``replace_existing=True``: apaga índice + manifesto e reconstrói tudo.
    - ``replace_existing=False`` e índice existente: reindexação **incremental por hash**
      (pula arquivos inalterados; remove chunks de arquivos apagados; atualiza alterados).
    - ``replace_existing=False`` sem índice: indexação completa inicial.
    """
    from txtai import Embeddings

    files = _flatten_files(scans, project_ids=project_ids)
    stats = BuildStats(files_total=len(files))

    if not files:
        stats.errors.append("Nenhum arquivo no inventário — escaneie as pastas antes de indexar.")
        return stats

    ensure_txtai_parent_exists()
    index_path = txtai_index_path()

    use_incremental = (
        incremental
        if incremental is not None
        else (not replace_existing and index_ready())
    )

    if replace_existing:
        use_incremental = False
        if index_path.exists():
            shutil.rmtree(index_path)
        remove_manifest_file()

    index_path.parent.mkdir(parents=True, exist_ok=True)

    if use_incremental:
        return _build_index_incremental(
            files,
            stats=stats,
            max_chars=max_chars,
            overlap=overlap,
            max_doc_chars=max_doc_chars,
            batch_size=batch_size,
            progress=progress,
        )

    rows, prep_stats = prepare_rows(
        scans,
        project_ids=project_ids,
        max_chars=max_chars,
        overlap=overlap,
        max_doc_chars=max_doc_chars,
        progress=progress,
    )
    stats.files_extracted = prep_stats.files_extracted
    stats.files_empty = prep_stats.files_empty
    stats.errors.extend(prep_stats.errors)

    if not rows:
        stats.errors.append("Nenhum chunk gerado — verifique arquivos vazios ou filtros de projeto.")
        return stats

    emb = Embeddings(embeddings_config())
    append_mode = not replace_existing and index_ready()

    try:
        if append_mode:
            emb.load(str(index_path))
            _upsert_rows_batched(
                emb,
                rows,
                batch_size=batch_size,
                use_initial_index=False,
                progress=progress,
            )
        else:
            _upsert_rows_batched(
                emb,
                rows,
                batch_size=batch_size,
                use_initial_index=True,
                progress=progress,
            )
        chunk_sig = chunking_signature(max_chars, overlap, max_doc_chars)
        manifest = IndexManifest(
            embedding_model=EMBEDDING_MODEL_ID,
            extraction_logic_version=EXTRACTION_LOGIC_VERSION,
            chunking_signature=chunk_sig,
        )
        files_by_key = {file_index_key(sf): sf for sf in files}
        chunk_ids_by_key: dict[str, list[str]] = {}
        for uid, row_dict in rows:
            key = f"{row_dict['project_id']}/{row_dict['relative_path']}"
            chunk_ids_by_key.setdefault(key, []).append(uid)
        for key, chunk_ids in chunk_ids_by_key.items():
            sf = files_by_key.get(key)
            if not sf:
                continue
            content_hash, _hash_err = resolve_content_hash(sf)
            if content_hash:
                manifest.set_file(key, content_hash=content_hash, chunk_ids=chunk_ids)
        emb.save(str(index_path))
        save_manifest(manifest)
        stats.chunks_written = len(rows)
    finally:
        emb.close()

    return stats


def _build_index_incremental(
    files: list[ScannedFile],
    *,
    stats: BuildStats,
    max_chars: int,
    overlap: int,
    max_doc_chars: int,
    batch_size: int,
    progress: Callable[[str], None] | None,
) -> BuildStats:
    from txtai import Embeddings

    stats.incremental = True
    manifest = load_manifest()
    had_manifest = manifest_exists() and bool(manifest.files)

    if manifest.embedding_model and manifest.embedding_model != EMBEDDING_MODEL_ID:
        stats.errors.append(
            f"Modelo do manifesto ({manifest.embedding_model}) difere do atual "
            f"({EMBEDDING_MODEL_ID}). Use **Substituir índice** para reconstruir."
        )
        return stats

    if not had_manifest:
        stats.errors.append(
            "Manifesto ausente: todos os arquivos serão processados nesta execução. "
            "Arquivos já removidos do disco podem permanecer no índice até a próxima "
            "indexação incremental (ou use **Substituir índice** uma vez)."
        )

    manifest.embedding_model = EMBEDDING_MODEL_ID
    chunk_sig = chunking_signature(max_chars, overlap, max_doc_chars)
    extraction_ok = manifest.extraction_logic_version == EXTRACTION_LOGIC_VERSION
    chunking_ok = manifest.chunking_signature == chunk_sig
    if had_manifest and not extraction_ok:
        stats.errors.append(
            f"Versão da extração mudou ({manifest.extraction_logic_version} → "
            f"{EXTRACTION_LOGIC_VERSION}): arquivos serão reprocessados."
        )
    if had_manifest and not chunking_ok and manifest.chunking_signature:
        stats.errors.append(
            f"Parâmetros de chunking mudaram ({manifest.chunking_signature} → "
            f"{chunk_sig}): arquivos serão reprocessados."
        )
    manifest.extraction_logic_version = EXTRACTION_LOGIC_VERSION
    manifest.chunking_signature = chunk_sig

    current_keys: set[str] = set()
    rows_to_upsert: list[Row] = []
    ids_to_delete: list[str] = []
    file_updates: list[tuple[str, str, list[str]]] = []  # key, hash, chunk_ids

    for sf in files:
        key = file_index_key(sf)
        current_keys.add(key)
        label = f"{sf.project_id}/{sf.relative_path}"
        if progress:
            progress(f"analisando {label}")

        content_hash, hash_err = resolve_content_hash(sf)
        if not content_hash:
            detail = f" ({hash_err})" if hash_err else ""
            stats.errors.append(f"{label}: não foi possível calcular SHA-256{detail}.")
            continue

        prev = manifest.get(key) if had_manifest else None
        skip_unchanged = (
            extraction_ok
            and chunking_ok
            and prev is not None
            and prev.content_hash_sha256 == content_hash
        )
        if skip_unchanged:
            stats.files_skipped_unchanged += 1
            continue

        if prev:
            ids_to_delete.extend(prev.chunk_ids)

        file_rows, had_text, err = rows_for_file(
            sf,
            max_chars=max_chars,
            overlap=overlap,
            max_doc_chars=max_doc_chars,
        )
        if err:
            stats.errors.append(err)
        if not had_text:
            stats.files_empty += 1
            if prev:
                manifest.remove_file(key)
                stats.files_removed += 1
            continue

        stats.files_reindexed += 1
        stats.files_extracted += 1
        rows_to_upsert.extend(file_rows)
        file_updates.append((key, content_hash, [uid for uid, _ in file_rows]))

    if had_manifest:
        for key in list(manifest.files.keys()):
            if key not in current_keys:
                removed_ids = manifest.remove_file(key)
                if removed_ids:
                    ids_to_delete.extend(removed_ids)
                    stats.files_removed += 1

    if not ids_to_delete and not rows_to_upsert:
        return stats

    index_path = txtai_index_path()
    emb = Embeddings(embeddings_config())
    loaded_existing = False

    try:
        if index_ready():
            emb.load(str(index_path))
            loaded_existing = True
        elif not rows_to_upsert:
            stats.errors.append("Índice inexistente e nada a gravar.")
            return stats

        if ids_to_delete:
            if progress:
                progress(f"removendo {len(ids_to_delete)} chunk(s) obsoleto(s)…")
            stats.chunks_deleted = _delete_ids_batched(
                emb,
                ids_to_delete,
                batch_size=batch_size,
                errors=stats.errors,
            )

        if rows_to_upsert:
            _upsert_rows_batched(
                emb,
                rows_to_upsert,
                batch_size=batch_size,
                use_initial_index=not loaded_existing,
                progress=progress,
            )
            stats.chunks_written = len(rows_to_upsert)

        emb.save(str(index_path))
        for key, content_hash, chunk_ids in file_updates:
            manifest.set_file(key, content_hash=content_hash, chunk_ids=chunk_ids)
        save_manifest(manifest)
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
    emb = Embeddings(embeddings_config())
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
