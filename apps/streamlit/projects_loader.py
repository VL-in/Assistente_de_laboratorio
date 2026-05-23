"""
Descoberta de projetos de P&D e varredura de arquivos para o inventário.

Convenção de projeto
--------------------
Cada subdiretório **imediato** da raiz configurada (ex.: ``D:/Projetos``) é
tratado como um projeto independente. O nome da pasta vira o ``project_id``.
Tudo dentro desse subdiretório — inclusive subpastas ``planning/``,
``results/``, etc. — pertence ao mesmo projeto.

Essa convenção espelha a estrutura de volumes Docker:

    /data/projetos/
        ELISA_2024/          ← project_id = "ELISA_2024"
            planning/
                Ensaio_01.docx
            results/
                Resultados_ELISA.xlsx
        Anticorpos/          ← project_id = "Anticorpos"
            Protocolo.docx

O ``project_id`` é gravado como metadado em cada chunk do índice txtai,
permitindo que o agente filtre respostas por projeto.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# ── Ambiente e configuração ──────────────────────────────────────────────────

# Extensões consideradas "documentos" no MVP; amplie aqui quando novos parsers
# forem adicionados em extract.py.
DEFAULT_DOCUMENT_EXTENSIONS: frozenset[str] = frozenset(
    {".docx", ".xlsx", ".xlsm", ".pdf", ".txt", ".md", ".csv"}
)

# Nome da variável de ambiente que sobrepõe o caminho padrão da raiz de projetos.
ENV_PROJETOS_ROOT = "ASSISTENTE_PROJETOS_DIR"


def running_inside_docker() -> bool:
    """
    Detecta se o processo está rodando dentro de um contêiner Docker.

    O Docker (e runtimes compatíveis como podman) criam ``/.dockerenv`` na raiz
    do contêiner. É uma heurística confiável para o MVP; não é adequada para
    ambientes de produção com hardening de segurança que removem esse arquivo.
    """
    return Path("/.dockerenv").exists()


def projetos_root_from_env() -> Path:
    """
    Resolve o caminho raiz dos projetos usando a cadeia de prioridade:

    1. ``ASSISTENTE_PROJETOS_DIR`` definido → usa esse valor (funciona em
       Docker e dev local com a variável exportada).
    2. Dentro de Docker sem a variável → ``/data/projetos`` (volume do Compose).
    3. Fora do Docker → caminho local de desenvolvimento no Windows.

    No Compose, recomenda-se sempre definir ``ASSISTENTE_PROJETOS_DIR``
    explicitamente para evitar depender do fallback hardcoded.
    """
    raw = os.environ.get(ENV_PROJETOS_ROOT, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    if running_inside_docker():
        return Path("/data/projetos").resolve()
    return Path(r"D:\Vanessa\AI_project\Projetos").resolve()


# ── Estruturas de dados ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class ScannedFile:
    """
    Representação imutável de um arquivo descoberto dentro de um projeto.

    Atributos
    ---------
    project_id:
        Nome da pasta raiz do projeto (filho direto da raiz de projetos).
    project_root:
        Caminho absoluto da pasta do projeto.
    absolute_path:
        Caminho absoluto do arquivo (resolvido, sem symlinks ambíguos).
    relative_path:
        Caminho POSIX relativo a ``project_root`` (ex.: ``planning/Ensaio.docx``).
        Normalizado para POSIX independentemente do SO para consistência nas
        chaves do manifesto.
    size_bytes:
        Tamanho em bytes no momento do escaneamento.
    modified_epoch:
        Timestamp de modificação (``st_mtime``) no momento do escaneamento.
    content_hash_sha256:
        Hash SHA-256 do conteúdo, preenchido apenas se ``compute_hashes=True``
        foi passado ao escanear. ``None`` quando não calculado — o pipeline de
        indexação calcula on-demand quando necessário.
    """

    project_id: str
    project_root: Path
    absolute_path: Path
    relative_path: str
    size_bytes: int
    modified_epoch: float
    content_hash_sha256: str | None = None


@dataclass
class ProjectScan:
    """
    Resultado do escaneamento de um projeto: lista de arquivos encontrados.

    Um ``ProjectScan`` corresponde a exatamente um diretório filho direto da
    raiz de projetos. A propriedade ``file_count`` é usada nas métricas da UI.
    """

    project_id: str
    root: Path
    files: list[ScannedFile] = field(default_factory=list)

    @property
    def file_count(self) -> int:
        return len(self.files)


def filter_scans_by_extensions(
    scans: list[ProjectScan],
    extensions: frozenset[str],
) -> list[ProjectScan]:
    """
    Deriva um sub-inventário filtrado por extensão a partir de um scan completo.

    Evita re-varrer o disco quando o caller já possui ``scans`` com hashes
    calculados — útil para sincronizar planilhas (OLAP) após escaneamento RAG.
    """
    normalized = frozenset(
        e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions
    )
    out: list[ProjectScan] = []
    for scan in scans:
        files = [
            f for f in scan.files if f.absolute_path.suffix.lower() in normalized
        ]
        if not files:
            continue
        out.append(
            ProjectScan(project_id=scan.project_id, root=scan.root, files=files)
        )
    return out


# ── Varredura de arquivos ────────────────────────────────────────────────────

def discover_project_directories(projetos_root: Path) -> list[Path]:
    """
    Lista os diretórios de primeiro nível dentro de ``projetos_root``.

    Retorna apenas subpastas diretas (um nível de profundidade) em ordem
    alfabética. Arquivos soltos na raiz e subpastas de nível mais profundo
    não são tratados como projetos.

    Erros de permissão ao listar a raiz retornam lista vazia em vez de levantar
    exceção, para que a UI exiba uma mensagem de erro em vez de travar.
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
    """Callback silencioso para ``os.walk``: ignora subpastas inacessíveis."""
    return


def _iter_files_recursive(
    project_root: Path,
    extensions: frozenset[str],
) -> Iterable[Path]:
    """
    Percorre recursivamente ``project_root`` e yield cada arquivo com extensão
    aceita.

    Filtros aplicados:
    - Temporários do Office (``~$arquivo.docx``): criados enquanto o arquivo
      está aberto; nunca devem ser indexados.
    - Arquivos ocultos (prefixo ``.``): metadados do sistema, não documentos.
    """
    for dirpath, _dirnames, filenames in os.walk(project_root, onerror=_walk_on_error):
        for name in filenames:
            if name.startswith("~$") or name.startswith("."):
                continue
            p = Path(dirpath, name)
            if not p.is_file():
                continue
            if p.suffix.lower() in extensions:
                yield p.resolve()


def scan_project(
    project_root: Path,
    *,
    extensions: frozenset[str] = DEFAULT_DOCUMENT_EXTENSIONS,
    compute_hashes: bool = False,
    chunk_size: int = 1024 * 1024,
) -> ProjectScan:
    """
    Varre recursivamente ``project_root`` e retorna o inventário de arquivos.

    Parâmetros
    ----------
    project_root:
        Raiz do projeto (subpasta direta da raiz de projetos).
    extensions:
        Conjunto de extensões a incluir no inventário.
    compute_hashes:
        Se ``True``, calcula SHA-256 de cada arquivo durante o escaneamento.
        Útil para detectar mudanças sem reler o arquivo na indexação, mas
        torna o escaneamento mais lento em pastas com muitos arquivos grandes.
        Quando ``False``, o hash é calculado on-demand pelo pipeline RAG.
    chunk_size:
        Tamanho do bloco de leitura para o cálculo de hash (bytes). O default
        de 1 MB mantém o consumo de memória constante independentemente do
        tamanho do arquivo.

    Retorna
    -------
    ProjectScan
        Arquivos ordenados alfabeticamente pelo caminho relativo.
    """
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
    Escaneia todos os projetos sob ``projetos_root`` e retorna a lista completa.

    Cada subpasta de primeiro nível vira um ``ProjectScan`` independente. A
    ordem da lista segue a ordem alfabética das pastas (definida em
    ``discover_project_directories``).
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
    Retorna um mapa ``project_id`` → lista de arquivos.

    Útil para filtrar o contexto do agente por projeto, garantindo que respostas
    sobre o projeto A não misturem evidências do projeto B.
    """
    return {s.project_id: list(s.files) for s in scans}


def scans_to_flat_records(scans: list[ProjectScan]) -> list[dict]:
    """
    Achata todos os ``ProjectScan`` em uma lista de dicionários.

    Formato compatível com ``pd.DataFrame.from_records`` para exibição na UI
    e exportação. Cada registro traz ``project_id`` para permitir filtragem
    posterior.
    """
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


# ── Utilitários de hash e validação ─────────────────────────────────────────

def compute_file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """
    Calcula o SHA-256 do conteúdo de um arquivo usando leitura em streaming.

    Lê o arquivo em blocos de ``chunk_size`` bytes para manter o consumo de
    memória constante (~1 MB por vez), independentemente do tamanho do arquivo.
    Usado pelo pipeline de indexação incremental para comparar com o hash
    registrado no manifesto.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _file_sha256(path: Path, *, chunk_size: int) -> str:
    """
    Wrapper interno de ``compute_file_sha256`` usado em ``scan_project``.

    Isola o parâmetro ``chunk_size`` do escaneamento da assinatura pública de
    ``compute_file_sha256``, permitindo que os dois evoluam independentemente.
    """
    return compute_file_sha256(path, chunk_size=chunk_size)


def validate_projetos_root(path: Path) -> tuple[bool, str]:
    """
    Verifica se ``path`` é um diretório existente e acessível.

    Retorna ``(True, "")`` se válido, ou ``(False, mensagem_de_erro)`` caso
    contrário. Usada pela UI antes de iniciar o escaneamento.
    """
    if not path.exists():
        return False, "Caminho não existe."
    if not path.is_dir():
        return False, "Caminho não é um diretório."
    return True, ""
