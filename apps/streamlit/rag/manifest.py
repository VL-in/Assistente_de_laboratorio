"""Manifesto de arquivos indexados (hash SHA-256 + ids de chunks) para reindexação incremental."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from projects_loader import ScannedFile

from .paths import txtai_data_root

MANIFEST_VERSION = 1
MANIFEST_FILENAME = "index_manifest.json"
# Incrementar quando a lógica de extração mudar (força reindexação de .docx no modo incremental).
EXTRACTION_LOGIC_VERSION = 2


def file_index_key(sf: ScannedFile) -> str:
    """Chave estável ``project_id/relative_path`` (POSIX)."""
    return f"{sf.project_id}/{sf.relative_path}"


def manifest_path() -> Path:
    return txtai_data_root() / MANIFEST_FILENAME


@dataclass
class FileManifestEntry:
    content_hash_sha256: str
    chunk_ids: list[str] = field(default_factory=list)


def chunking_signature(max_chars: int, overlap: int, max_doc_chars: int) -> str:
    """Identifica parâmetros de chunking; mudança exige reindexar mesmo com mesmo hash do arquivo."""
    return f"{max_chars}:{overlap}:{max_doc_chars}"


@dataclass
class IndexManifest:
    version: int = MANIFEST_VERSION
    embedding_model: str = ""
    extraction_logic_version: int = EXTRACTION_LOGIC_VERSION
    chunking_signature: str = ""
    files: dict[str, FileManifestEntry] = field(default_factory=dict)

    def get(self, key: str) -> FileManifestEntry | None:
        return self.files.get(key)

    def set_file(self, key: str, *, content_hash: str, chunk_ids: list[str]) -> None:
        self.files[key] = FileManifestEntry(
            content_hash_sha256=content_hash,
            chunk_ids=list(chunk_ids),
        )

    def remove_file(self, key: str) -> list[str]:
        entry = self.files.pop(key, None)
        return list(entry.chunk_ids) if entry else []


def manifest_exists() -> bool:
    p = manifest_path()
    return p.is_file() and p.stat().st_size > 0


def load_manifest() -> IndexManifest:
    p = manifest_path()
    if not p.is_file():
        return IndexManifest()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return IndexManifest()

    if not isinstance(raw, dict):
        return IndexManifest()

    files: dict[str, FileManifestEntry] = {}
    for key, entry in (raw.get("files") or {}).items():
        if not isinstance(entry, dict):
            continue
        h = entry.get("content_hash_sha256") or entry.get("content_hash")
        if not h:
            continue
        ids = entry.get("chunk_ids") or []
        if not isinstance(ids, list):
            ids = []
        files[str(key)] = FileManifestEntry(
            content_hash_sha256=str(h),
            chunk_ids=[str(i) for i in ids],
        )

    return IndexManifest(
        version=int(raw.get("version") or MANIFEST_VERSION),
        embedding_model=str(raw.get("embedding_model") or ""),
        extraction_logic_version=int(
            raw.get("extraction_logic_version") or 0
        ),
        chunking_signature=str(raw.get("chunking_signature") or ""),
        files=files,
    )


def save_manifest(manifest: IndexManifest) -> None:
    txtai_data_root().mkdir(parents=True, exist_ok=True)
    payload = {
        "version": manifest.version,
        "embedding_model": manifest.embedding_model,
        "extraction_logic_version": manifest.extraction_logic_version,
        "chunking_signature": manifest.chunking_signature,
        "files": {
            key: {
                "content_hash_sha256": entry.content_hash_sha256,
                "chunk_ids": entry.chunk_ids,
            }
            for key, entry in sorted(manifest.files.items())
        },
    }
    manifest_path().write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def remove_manifest_file() -> None:
    p = manifest_path()
    if p.is_file():
        p.unlink(missing_ok=True)
