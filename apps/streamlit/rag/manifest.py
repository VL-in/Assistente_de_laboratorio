"""
Manifesto de indexação para reprocessamento incremental por hash.

O manifesto é um arquivo JSON (``index_manifest.json``) salvo junto ao índice
txtai. Ele guarda, para cada arquivo indexado:
  - O hash SHA-256 do conteúdo na última indexação.
  - Os IDs dos chunks gerados (usados para deletar entradas antigas do índice).

Na próxima execução incremental, o hash atual do arquivo é comparado com o
registrado no manifesto:
  - Igual → arquivo inalterado, chunks existentes são mantidos.
  - Diferente → chunks antigos são deletados e o arquivo é reprocessado.
  - Ausente no manifesto → arquivo novo, precisa ser indexado.
  - Presente no manifesto mas ausente no disco → arquivo removido, chunks são
    deletados do índice.

Além dos hashes de arquivo, o manifesto armazena três campos de "versão" que
invalidam **todos** os arquivos quando mudam, mesmo que os hashes sejam iguais:
  - ``embedding_model``: troca de modelo exige reindexação total.
  - ``extraction_logic_version``: mudança no código de extração (ex.: novo
    suporte a tabelas Word) exige reprocessamento para uniformidade.
  - ``chunking_signature``: mudança nos parâmetros de chunking (tamanho, overlap)
    altera os limites dos chunks, invalidando os IDs gravados.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from projects_loader import ScannedFile

from .paths import txtai_data_root


# ── Constantes e schema ─────────────────────────────────────────────────────

MANIFEST_VERSION = 1
MANIFEST_FILENAME = "index_manifest.json"

# Incrementar este valor sempre que a lógica de extração de texto mudar de
# forma que documentos já indexados precisem ser reprocessados para refletir
# o novo comportamento (ex.: v1 → v2 quando tabelas Word foram adicionadas).
EXTRACTION_LOGIC_VERSION = 3


# ── Chave de arquivo ────────────────────────────────────────────────────────

def file_index_key(sf: ScannedFile) -> str:
    """
    Retorna a chave estável ``project_id/relative_path`` (separador POSIX).

    Esta chave é usada como identificador primário no dicionário de arquivos do
    manifesto. O caminho relativo já é normalizado para POSIX em
    ``scan_project``, garantindo consistência entre Windows e Linux.
    """
    return f"{sf.project_id}/{sf.relative_path}"


# ── Estruturas de dados ─────────────────────────────────────────────────────

@dataclass
class FileManifestEntry:
    """Registro de um arquivo no manifesto.

    Atributos
    ---------
    content_hash_sha256:
        Hash SHA-256 do conteúdo do arquivo no momento da última indexação.
        Comparado a cada execução incremental para decidir se o arquivo mudou.
    chunk_ids:
        Lista de IDs determinísticos (SHA-256 de ``project_id + path + índice``)
        dos chunks gerados. Necessária para deletar entradas antigas do índice
        txtai antes de reindexar o arquivo com o novo conteúdo.
    """

    content_hash_sha256: str
    chunk_ids: list[str] = field(default_factory=list)


def chunking_signature(max_chars: int, overlap: int, max_doc_chars: int) -> str:
    """
    Gera uma string que identifica os parâmetros de chunking usados na indexação.

    Quando esta assinatura difere do valor gravado no manifesto, todos os
    arquivos são reprocessados na próxima execução incremental — mesmo que o
    conteúdo dos documentos não tenha mudado. Isso garante que os chunks
    refletam os parâmetros corretos de tamanho e sobreposição.
    """
    return f"{max_chars}:{overlap}:{max_doc_chars}"


@dataclass
class IndexManifest:
    """
    Estado completo do índice: versões de controle + entradas por arquivo.

    Campos de controle de versão
    ----------------------------
    embedding_model:
        ID do modelo de embedding usado. Troca de modelo exige reconstrução
        total porque os vetores são incompatíveis entre modelos.
    extraction_logic_version:
        Versão da lógica de extração de texto. Ver ``EXTRACTION_LOGIC_VERSION``.
    chunking_signature:
        Parâmetros de chunking serializados. Ver ``chunking_signature()``.

    Campo de dados
    --------------
    files:
        Mapa ``project_id/relative_path`` → ``FileManifestEntry``.
    """

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
        """Remove a entrada e retorna os IDs de chunks para deleção no índice."""
        entry = self.files.pop(key, None)
        return list(entry.chunk_ids) if entry else []


# ── I/O em disco ────────────────────────────────────────────────────────────

def manifest_path() -> Path:
    return txtai_data_root() / MANIFEST_FILENAME


def manifest_exists() -> bool:
    p = manifest_path()
    return p.is_file() and p.stat().st_size > 0


def load_manifest() -> IndexManifest:
    """
    Lê o manifesto do disco e retorna um ``IndexManifest`` populado.

    Retorna um manifesto vazio (sem entradas) se o arquivo não existir, estiver
    corrompido ou tiver formato inesperado — sem levantar exceção, para não
    interromper a indexação incremental.

    Suporta migração silenciosa do campo antigo ``content_hash`` para
    ``content_hash_sha256``, garantindo compatibilidade com manifestos gerados
    por versões anteriores do código.
    """
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
        # Migração: versões antigas gravavam "content_hash"; versões atuais usam
        # "content_hash_sha256". Aceita ambos para não perder o histórico de hashes.
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
        extraction_logic_version=int(raw.get("extraction_logic_version") or 0),
        chunking_signature=str(raw.get("chunking_signature") or ""),
        files=files,
    )


def save_manifest(manifest: IndexManifest) -> None:
    """
    Serializa o manifesto em JSON e grava no disco.

    As entradas de arquivo são ordenadas alfabeticamente pela chave para tornar
    o arquivo legível e o diff entre versões previsível.

    Deve ser chamada **após** ``Embeddings.save()``, pois o mtime do manifesto
    é usado como referência de invalidação do cache do Streamlit.
    """
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
    """Remove o arquivo de manifesto do disco (usado em reconstrução total)."""
    p = manifest_path()
    if p.is_file():
        p.unlink(missing_ok=True)
