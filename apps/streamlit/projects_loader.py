"""
Descoberta de projetos de P&D e varredura de arquivos.

Convenção:
  - Cada subdiretório imediato da raiz configurada (ex.: .../Projetos) = um projeto.
  - Tudo abaixo desse subdiretório pertence ao mesmo projeto (planning, results, etc.).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Tipos alinhados ao MVP (planilhas e Word); amplie conforme parsers forem adicionados.
DEFAULT_DOCUMENT_EXTENSIONS: frozenset[str] = frozenset(
    {".docx", ".xlsx", ".xlsm", ".pdf", ".txt", ".md", ".csv"}
)

ENV_PROJETOS_ROOT = "ASSISTENTE_PROJETOS_DIR"


def running_inside_docker() -> bool:
    """Contêineres em geral criam este arquivo na raiz (Docker / podman compatível)."""
    return Path("/.dockerenv").exists()


def projetos_root_from_env() -> Path:
    """
    Ordem: variável de ambiente → (se Docker e sem env) /data/projetos → fallback dev Windows.
    No Compose, defina ASSISTENTE_PROJETOS_DIR=/data/projetos e monte Projetos nesse caminho.
    """
    raw = os.environ.get(ENV_PROJETOS_ROOT, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    if running_inside_docker():
        return Path("/data/projetos").resolve()
    return Path(r"D:\Vanessa\AI_project\Projetos").resolve()


@dataclass(frozen=True)
class ScannedFile:
    """Um arquivo dentro de um projeto (caminho relativo ao root do projeto)."""

    project_id: str
    project_root: Path
    absolute_path: Path
    relative_path: str  # POSIX-style, relativo ao projeto
    size_bytes: int
    modified_epoch: float
    content_hash_sha256: str | None = None


@dataclass
class ProjectScan:
    """Um projeto = um diretório filho direto da raiz de Projetos."""

    project_id: str
    root: Path
    files: list[ScannedFile] = field(default_factory=list)

    @property
    def file_count(self) -> int:
        return len(self.files)


def discover_project_directories(projetos_root: Path) -> list[Path]:
    """
    Lista apenas diretórios que estão um nível abaixo de `projetos_root`.
    Ignora arquivos soltos na raiz e não trata netos como projetos.
    """
    if not projetos_root.is_dir():
        return []
    out: list[Path] = []
    try:
        entries = sorted(projetos_root.iterdir(), key=lambda p: p.name.lower())
    except OSError:
        return []
    for entry in entries:
        if entry.is_dir():
            try:
                out.append(entry.resolve())
            except OSError:
                continue
    return out


def _walk_on_error(_err: OSError) -> None:
    """Ignora subpastas sem permissão de leitura (Windows/Linux)."""
    return


def _iter_files_recursive(
    project_root: Path,
    extensions: frozenset[str],
) -> Iterable[Path]:
    for dirpath, _dirnames, filenames in os.walk(project_root, onerror=_walk_on_error):
        for name in filenames:
            # Ignora temporários do Office (~$arquivo.docx) e ocultos.
            if name.startswith("~$") or name.startswith("."):
                continue
            p = Path(dirpath, name)
            if not p.is_file():
                continue
            suf = p.suffix.lower()
            if suf in extensions:
                yield p.resolve()


def scan_project(
    project_root: Path,
    *,
    extensions: frozenset[str] = DEFAULT_DOCUMENT_EXTENSIONS,
    compute_hashes: bool = False,
    chunk_size: int = 1024 * 1024,
) -> ProjectScan:
    project_id = project_root.name
    files: list[ScannedFile] = []

    for abs_path in _iter_files_recursive(project_root, extensions):
        try:
            rel = abs_path.relative_to(project_root)
        except ValueError:
            continue
        rel_str = rel.as_posix()
        try:
            meta = abs_path.stat()
        except OSError:
            continue
        h: str | None = None
        if compute_hashes:
            try:
                h = _file_sha256(abs_path, chunk_size=chunk_size)
            except OSError:
                h = None
        files.append(
            ScannedFile(
                project_id=project_id,
                project_root=project_root,
                absolute_path=abs_path,
                relative_path=rel_str,
                size_bytes=meta.st_size,
                modified_epoch=meta.st_mtime,
                content_hash_sha256=h,
            )
        )
    files.sort(key=lambda f: f.relative_path.lower())
    return ProjectScan(project_id=project_id, root=project_root, files=files)


def scan_all_projects(
    projetos_root: Path,
    *,
    extensions: frozenset[str] = DEFAULT_DOCUMENT_EXTENSIONS,
    compute_hashes: bool = False,
) -> list[ProjectScan]:
    """
    Carrega a árvore completa: um `ProjectScan` por subpasta direta da raiz.
    """
    scans: list[ProjectScan] = []
    for project_dir in discover_project_directories(projetos_root):
        scans.append(
            scan_project(
                project_dir,
                extensions=extensions,
                compute_hashes=compute_hashes,
            )
        )
    return scans


def documents_by_project(scans: list[ProjectScan]) -> dict[str, list[ScannedFile]]:
    """
    Mapa `project_id` → lista de arquivos desse projeto (inclui subpastas como planning/results).
    Útil para filtrar contexto do agente por projeto de P&D.
    """
    return {s.project_id: list(s.files) for s in scans}


def scans_to_flat_records(scans: list[ProjectScan]) -> list[dict]:
    """Útil para DataFrame / export; cada linha traz `project_id` para filtro do agente."""
    rows: list[dict] = []
    for scan in scans:
        for f in scan.files:
            rows.append(
                {
                    "project_id": f.project_id,
                    "project_root": str(f.project_root),
                    "relative_path": f.relative_path,
                    "absolute_path": str(f.absolute_path),
                    "size_bytes": f.size_bytes,
                    "modified_epoch": f.modified_epoch,
                    "content_hash_sha256": f.content_hash_sha256,
                }
            )
    return rows


def compute_file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash SHA-256 do conteúdo do arquivo (usado na reindexação incremental)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _file_sha256(path: Path, *, chunk_size: int) -> str:
    return compute_file_sha256(path, chunk_size=chunk_size)


def validate_projetos_root(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "Caminho não existe."
    if not path.is_dir():
        return False, "Caminho não é um diretório."
    return True, ""
