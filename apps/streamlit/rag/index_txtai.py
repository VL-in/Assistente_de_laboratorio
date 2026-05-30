"""
Construção e consulta do índice de embeddings txtai.

Este módulo é o núcleo do pipeline RAG. Ele expõe dois grupos de funções:

Indexação (``build_index``)
    Ponto de entrada único para criar ou atualizar o índice. Decide
    automaticamente entre três fluxos:
    - **Reconstrução total** (``replace_existing=True``): apaga tudo e indexa
      do zero. Usado quando o modelo ou a lógica de extração muda.
    - **Indexação incremental** (índice existente + ``replace_existing=False``):
      compara SHA-256 de cada arquivo com o manifesto; processa apenas
      arquivos novos, alterados ou removidos.
    - **Primeira indexação** (sem índice em disco): equivalente à reconstrução
      total, mas sem necessidade de apagar nada.

Busca híbrida (``search_chunks``, ``search_with_backend``)
    Consulta o índice por similaridade semântica (E5) **e** correspondência
    lexical BM25 quando ``RAG_HYBRID_ENABLED=1`` (padrão). ``search_with_backend``
    recebe a instância já carregada (cache do Streamlit), evitando recarregar o
    modelo a cada pergunta do usuário.

Formatação de contexto (``format_context_for_llm``)
    Monta o bloco de contexto citável injetado no system prompt do chat.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from projects_loader import ProjectScan, ScannedFile, compute_file_sha256

from .chunking import DEFAULT_CHUNK_MAX_CHARS, DEFAULT_CHUNK_OVERLAP, chunk_text
from .extract import extract_from_scanned_file
from .hybrid import env_hybrid_enabled, hybrid_dense_weight
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


# ── Configuração do modelo ───────────────────────────────────────────────────

from .embedding_client import EMBEDDING_MODEL_ID, EMBEDDING_TRANSFORM_PATH

# Modelo multilingual E5 (~384 dimensões, até 512 tokens por entrada).
# Servido pelo contêiner TEI (``docker/embeddings``). Troca de modelo invalida
# o índice existente — use "Substituir índice" na UI.
#
# Prefixos E5: ``passage:`` na indexação, ``query:`` na busca (via ``instructions``).

# Alias de tipo: tupla (id_do_chunk, dicionário_de_metadados)
Row = tuple[str, dict]


def embeddings_config() -> dict:
    """
    Retorna a configuração passada ao construtor ``txtai.Embeddings``.

    ``content=True`` instrui o txtai a armazenar o texto e os metadados junto
    ao vetor no índice. Sem isso, ``search()`` retornaria apenas ``(id, score)``
    e não seria possível montar o contexto com trechos citáveis para o LLM.

    ``hybrid=True`` (padrão) cria um índice BM25 paralel ao vetorial. Na busca,
    o txtai funde os scores dos dois índices — útil para termos técnicos exatos
    (ex.: ``tampão de amostra``) que a busca só semântica costuma diluir.

    O vetor é calculado pelo serviço TEI (``EMBEDDING_SERVICE_URL``), não
    carregado dentro do processo Streamlit.
    """
    cfg: dict = {
        "path": "external",
        "method": "external",
        "transform": EMBEDDING_TRANSFORM_PATH,
        "content": True,
        "instructions": {
            "query": "query: ",
            "data": "passage: ",
        },
    }
    if env_hybrid_enabled():
        cfg["hybrid"] = True
    return cfg


# ── Tipos e estatísticas ─────────────────────────────────────────────────────

@dataclass
class BuildStats:
    """
    Contadores de resultado de uma execução de ``build_index``.

    Atributos
    ---------
    files_total:
        Total de arquivos no inventário filtrado.
    files_extracted:
        Arquivos dos quais texto foi extraído com sucesso.
    files_empty:
        Arquivos sem texto extraível (vazios, imagens, formato não suportado).
    files_skipped_unchanged:
        Arquivos cujo SHA-256 e parâmetros de chunking coincidem com o
        manifesto — nenhuma operação no índice foi necessária.
    files_reindexed:
        Arquivos processados no modo incremental por terem mudado (hash,
        versão de extração ou parâmetros de chunking diferentes).
    files_removed:
        Arquivos ausentes no disco mas presentes no manifesto (deletados pelo
        usuário) cujos chunks foram removidos do índice.
    chunks_written:
        Número de chunks gravados/atualizados no índice nesta execução.
    chunks_deleted:
        Número de chunks removidos do índice (modo incremental).
    incremental:
        ``True`` se a execução usou o fluxo incremental.
    errors:
        Mensagens de aviso e erro não fatais coletados durante a execução.
    """

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


# ── Helpers de preparação ────────────────────────────────────────────────────

def _flatten_files(
    scans: list[ProjectScan],
    *,
    project_ids: set[str] | None,
) -> list[ScannedFile]:
    """Agrega todos os ``ScannedFile`` dos scans, opcionalmente filtrando por projeto."""
    out: list[ScannedFile] = []
    for scan in scans:
        if project_ids is not None and scan.project_id not in project_ids:
            continue
        out.extend(scan.files)
    return out


def resolve_content_hash(sf: ScannedFile) -> tuple[str | None, str | None]:
    """
    Retorna ``(hash_sha256, mensagem_erro)``.

    Usa o hash já calculado no escaneamento quando disponível. Caso contrário,
    calcula on-demand lendo o arquivo. Retorna ``(None, erro)`` se o arquivo
    não puder ser lido.
    """
    if sf.content_hash_sha256:
        return sf.content_hash_sha256, None
    try:
        return compute_file_sha256(sf.absolute_path), None
    except OSError as exc:
        return None, str(exc)


def chunk_uid(sf: ScannedFile, chunk_index: int) -> str:
    """
    Gera um ID determinístico e único para um chunk específico de um arquivo.

    O ID é um SHA-256 de ``project_id + relative_path + chunk_index``,
    separados por bytes nulos para evitar colisões. Determinismo é essencial
    para o modo incremental: ao reindexar um arquivo, os IDs dos chunks
    novos coincidem com os antigos, permitindo substituição precisa via
    ``delete`` + ``upsert`` sem contaminar o índice.
    """
    raw = f"{sf.project_id}\x00{sf.relative_path}\x00{chunk_index}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def rows_for_file(
    sf: ScannedFile,
    *,
    max_chars: int,
    overlap: int,
    max_doc_chars: int,
) -> tuple[list[Row], bool, str | None]:
    """
    Extrai texto, faz chunking e monta as linhas prontas para inserção no txtai.

    Retorna
    -------
    (rows, teve_texto, mensagem_erro)
        ``rows``: lista de ``(id, {text, project_id, relative_path, ...})``
        ``teve_texto``: ``False`` se arquivo vazio ou falha de extração
        ``mensagem_erro``: aviso descritivo, ou ``None`` se tudo ok
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
        # O prefixo de citação fica no campo "cited" (exibição / contexto LLM).
        # O campo "text" contém apenas o conteúdo puro, sem o prefixo.
        # Separar os dois campos é essencial para a qualidade do embedding:
        # se o prefixo "[Projeto: X] [Arquivo: Y]" fosse incluído no "text",
        # todos os chunks teriam um vetor parcialmente idêntico (o prefixo),
        # o que reduz a discriminação semântica e produz resultados irrelevantes.
        cited = (
            f"[Projeto: {sf.project_id}] [Arquivo: {sf.relative_path}] [Chunk {idx}]\n"
            f"{part}"
        )
        row_dict = {
            "text": part,    # somente conteúdo → qualidade máxima do embedding
            "cited": cited,  # texto completo com citação → exibição e contexto LLM
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
    """
    Processa todos os arquivos dos scans e retorna as linhas para indexação.

    Usado no fluxo de reconstrução total (não-incremental). O fluxo incremental
    usa ``rows_for_file`` diretamente, arquivo a arquivo.
    """
    stats = BuildStats()
    rows: list[Row] = []
    files = _flatten_files(scans, project_ids=project_ids)
    stats.files_total = len(files)

    for sf in files:
        if progress:
            progress(f"{sf.project_id}/{sf.relative_path}")

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


# ── Operações no índice txtai ────────────────────────────────────────────────

def _delete_ids_batched(
    emb: object,
    ids: list[str],
    *,
    batch_size: int,
    errors: list[str] | None = None,
) -> int:
    """
    Remove chunks do índice em lotes, ignorando falhas por lote.

    Deduplica os IDs antes de enviar para evitar erros em operações de deleção
    que não são idempotentes em certas versões do txtai.
    """
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
    """
    Insere ou atualiza chunks no índice em lotes.

    O txtai exige que o **primeiro** lote de uma instância nova use
    ``Embeddings.index()`` para inicializar as estruturas internas. Lotes
    subsequentes — incluindo adições a um índice já carregado — devem usar
    ``Embeddings.upsert()``.

    ``use_initial_index=True`` indica que esta instância de ``Embeddings`` não
    tem dados ainda; ``False`` indica que o índice foi carregado via ``.load()``
    e já possui entradas.
    """
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


# ── Ponto de entrada: build_index ────────────────────────────────────────────

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
    Cria ou atualiza o índice de embeddings em ``txtai_index_path()``.

    Fluxo de decisão
    ----------------
    ``replace_existing=True``
        Apaga o índice e o manifesto e reconstrói tudo do zero. Necessário
        quando o modelo de embedding ou a versão de extração muda.

    ``replace_existing=False`` + índice existente em disco
        Ativa a **reindexação incremental por hash** (delega a
        ``_build_index_incremental``). Processa apenas arquivos alterados,
        novos ou removidos desde a última indexação.

    ``replace_existing=False`` + sem índice em disco
        Primeira indexação: cria um índice novo a partir de zero sem precisar
        apagar nada.

    O parâmetro ``incremental`` (raramente usado) permite sobrescrever a
    detecção automática quando chamado de código externo.
    """
    from txtai import Embeddings

    files = _flatten_files(scans, project_ids=project_ids)
    stats = BuildStats(files_total=len(files))

    if not files:
        stats.errors.append("Nenhum arquivo no inventário — escaneie as pastas antes de indexar.")
        return stats

    ensure_txtai_parent_exists()
    index_path = txtai_index_path()

    # Determina o modo: incremental automático quando existe índice e o usuário
    # não pediu substituição explícita.
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

    # --- Fluxo de reconstrução total (não-incremental) ---
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

    try:
        # Sempre constrói do zero: não carrega índice existente para evitar
        # duplicação de chunks em execuções repetidas sem replace_existing.
        _upsert_rows_batched(
            emb,
            rows,
            batch_size=batch_size,
            use_initial_index=True,
            progress=progress,
        )
        # Monta manifesto completo com hashes e IDs de todos os arquivos
        # processados, para que a próxima execução possa operar de forma incremental.
        chunk_sig = chunking_signature(max_chars, overlap, max_doc_chars)
        manifest = IndexManifest(
            embedding_model=EMBEDDING_MODEL_ID,
            extraction_logic_version=EXTRACTION_LOGIC_VERSION,
            chunking_signature=chunk_sig,
            hybrid_index=env_hybrid_enabled(),
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
        # O manifesto é salvo APÓS o índice; seu mtime serve de referência para
        # invalidação do cache do Streamlit (ver index_mtime()).
        save_manifest(manifest)
        stats.chunks_written = len(rows)
    finally:
        emb.close()

    return stats


# ── Fluxo incremental ────────────────────────────────────────────────────────

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
    """
    Atualiza o índice processando apenas os arquivos que mudaram.

    Algoritmo
    ---------
    1. Carrega o manifesto; verifica compatibilidade de modelo/versão/parâmetros.
    2. Para cada arquivo no inventário:
       a. Calcula o hash SHA-256 atual.
       b. Compara com o manifesto: se igual e versões ok → pula (``skipped``).
       c. Se diferente: coleta IDs antigos para deleção (antes de processar,
          garantindo remoção mesmo se o arquivo virou vazio) e extrai/chunka.
    3. Após o loop, verifica quais chaves do manifesto não existem mais no
       inventário (arquivos deletados do disco) e coleta seus IDs para deleção.
    4. Se não há nada para deletar nem para inserir → retorna sem gravar
       (early-return seguro: o manifesto já está correto para este estado).
    5. Carrega o índice existente, aplica deleções e inserções, salva.
    """
    from txtai import Embeddings

    stats.incremental = True
    manifest = load_manifest()
    had_manifest = manifest_exists() and bool(manifest.files)

    # Troca de modelo de embedding incompatibiliza os vetores existentes;
    # exige reconstrução total em vez de atualização.
    if manifest.embedding_model and manifest.embedding_model != EMBEDDING_MODEL_ID:
        stats.errors.append(
            f"Modelo do manifesto ({manifest.embedding_model}) difere do atual "
            f"({EMBEDDING_MODEL_ID}). Use **Substituir índice** para reconstruir."
        )
        return stats

    want_hybrid = env_hybrid_enabled()
    if had_manifest and want_hybrid and not manifest.hybrid_index:
        stats.errors.append(
            "Índice atual foi criado só com busca semântica (sem BM25). "
            "Use **Substituir índice** uma vez para habilitar busca híbrida."
        )
        return stats
    if had_manifest and not want_hybrid and manifest.hybrid_index:
        stats.errors.append(
            "Índice híbrido (BM25) no disco, mas RAG_HYBRID_ENABLED=0. "
            "Use **Substituir índice** ou reative a busca híbrida."
        )
        return stats

    if not had_manifest:
        stats.errors.append(
            "Manifesto ausente: todos os arquivos serão processados nesta execução. "
            "Arquivos já removidos do disco podem permanecer no índice até a próxima "
            "indexação incremental (ou use **Substituir índice** uma vez)."
        )

    # Atualiza os campos de versão no manifesto em memória; serão gravados no
    # final apenas se houver mudanças efetivas no índice.
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
    manifest.hybrid_index = want_hybrid

    current_keys: set[str] = set()
    rows_to_upsert: list[Row] = []
    ids_to_delete: list[str] = []
    # Acumulador de atualizações do manifesto; gravado no final para garantir
    # consistência entre índice e manifesto (ambos atualizados ou nenhum).
    file_updates: list[tuple[str, str, list[str]]] = []  # (key, hash, chunk_ids)

    for sf in files:
        key = file_index_key(sf)
        current_keys.add(key)
        if progress:
            progress(f"analisando {sf.project_id}/{sf.relative_path}")

        content_hash, hash_err = resolve_content_hash(sf)
        if not content_hash:
            detail = f" ({hash_err})" if hash_err else ""
            stats.errors.append(f"{key}: não foi possível calcular SHA-256{detail}.")
            continue

        prev = manifest.get(key) if had_manifest else None

        # Pula o arquivo apenas quando todas as condições de estabilidade são
        # atendidas: versões compatíveis, arquivo presente no manifesto e hash
        # idêntico ao registrado.
        skip_unchanged = (
            extraction_ok
            and chunking_ok
            and prev is not None
            and prev.content_hash_sha256 == content_hash
        )
        if skip_unchanged:
            stats.files_skipped_unchanged += 1
            continue

        # IDs antigos coletados ANTES de chamar rows_for_file: se o arquivo
        # virou vazio (sem texto), seus chunks antigos ainda precisam ser
        # removidos do índice, e o manifesto precisa ser atualizado.
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
                # Remove do manifesto agora; os IDs já foram adicionados a
                # ids_to_delete acima.
                manifest.remove_file(key)
                stats.files_removed += 1
            continue

        stats.files_reindexed += 1
        stats.files_extracted += 1
        rows_to_upsert.extend(file_rows)
        file_updates.append((key, content_hash, [uid for uid, _ in file_rows]))

    # Arquivos presentes no manifesto mas ausentes no inventário atual foram
    # deletados do disco pelo usuário. Seus chunks precisam sair do índice.
    if had_manifest:
        for key in list(manifest.files.keys()):
            if key not in current_keys:
                removed_ids = manifest.remove_file(key)
                if removed_ids:
                    ids_to_delete.extend(removed_ids)
                    stats.files_removed += 1

    # Early-return seguro: nenhuma operação no índice é necessária quando todos
    # os arquivos foram pulados. O manifesto em memória reflete o estado correto
    # (sem mudanças), então não precisamos gravá-lo novamente.
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
        # Atualiza o manifesto com os hashes e IDs dos arquivos processados;
        # entradas de arquivos removidos/vazios já foram limpas no loop acima.
        for key, content_hash, chunk_ids in file_updates:
            manifest.set_file(key, content_hash=content_hash, chunk_ids=chunk_ids)
        save_manifest(manifest)
    finally:
        emb.close()

    return stats


# ── Utilitários de estado do índice ─────────────────────────────────────────

def index_ready() -> bool:
    """Retorna ``True`` se o diretório do índice existe e contém arquivos."""
    p = txtai_index_path()
    if not p.exists() or not p.is_dir():
        return False
    return any(p.iterdir())


def index_mtime() -> float:
    """
    Retorna o timestamp de modificação do índice; ``0.0`` se inexistente.

    Usado como chave de invalidação do ``st.cache_resource`` do Streamlit: quando
    o valor muda (após um rebuild), o backend txtai em cache é descartado e
    recarregado com o índice atualizado.

    Usa o manifesto como referência primária porque, no Windows, o ``st_mtime``
    de um diretório nem sempre é atualizado quando arquivos internos mudam.
    O manifesto é sempre gravado **após** ``emb.save()``, tornando seu mtime
    o indicador mais confiável de quando o índice foi atualizado pela última vez.
    """
    from .manifest import manifest_path

    mp = manifest_path()
    if mp.is_file():
        try:
            return mp.stat().st_mtime
        except OSError:
            pass
    p = txtai_index_path()
    if not p.exists() or not p.is_dir():
        return 0.0
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


# ── Busca híbrida / semântica ────────────────────────────────────────────────

def _search_parameters(
    query: str,
    *,
    hybrid_weight: float | None = None,
) -> dict[str, object]:
    """Parâmetros nomeados para a query SQL do txtai."""
    params: dict[str, object] = {"query": query}
    if env_hybrid_enabled():
        params["weight"] = hybrid_dense_weight(hybrid_weight)
    return params


def _backend_supports_hybrid(backend: object) -> bool:
    """True quando o índice carregado possui componente BM25."""
    if not env_hybrid_enabled():
        return False
    issparse = getattr(backend, "issparse", None)
    if callable(issparse):
        return bool(issparse())
    return getattr(backend, "scoring", None) is not None


def _annotate_hits(
    hits: list[dict],
    *,
    backend: object,
    hybrid_weight: float | None,
) -> list[dict]:
    """Marca cada hit com metadados de modo de busca (observabilidade / UI dev)."""
    if not hits:
        return hits
    hybrid_active = _backend_supports_hybrid(backend)
    weight = hybrid_dense_weight(hybrid_weight) if hybrid_active else None
    for hit in hits:
        hit["search_mode"] = "hybrid" if hybrid_active else "semantic"
        if weight is not None:
            hit["hybrid_dense_weight"] = weight
    return hits


def format_search_results(raw: object) -> list[dict]:
    """
    Normaliza a saída de ``Embeddings.search`` para uma lista uniforme de dicts.

    O txtai retorna dicts quando ``content=True`` está ativo (MVP) e tuplas
    ``(id, score)`` quando não está. Esta função aceita ambos os formatos para
    compatibilidade com índices criados com configurações diferentes.
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


def search_chunks(query: str, limit: int, *, hybrid_weight: float | None = None) -> list[dict]:
    """
    Executa busca híbrida (ou só semântica) abrindo e índice a cada chamada.

    Adequado para scripts e testes isolados onde não há cache de instância.
    Para uso no Streamlit, prefira ``search_with_backend`` com a instância
    cacheada em ``st.cache_resource``.
    """
    from txtai import Embeddings

    if not index_ready():
        return []

    q = (query or "").strip()
    if not q:
        return []

    path = txtai_index_path()
    emb = Embeddings(embeddings_config())
    weight = hybrid_dense_weight(hybrid_weight) if env_hybrid_enabled() else None
    try:
        emb.load(str(path))
        raw = emb.search(q, limit, weights=weight) if weight is not None else emb.search(q, limit)
        hits = format_search_results(raw)
        return _annotate_hits(hits, backend=emb, hybrid_weight=hybrid_weight)
    finally:
        emb.close()


def filter_hits_by_project(
    hits: list[dict],
    project_ids: set[str] | None,
) -> list[dict]:
    """Restringe resultados RAG a um subconjunto de projetos (evita vazamento entre projetos)."""
    if not project_ids:
        return hits
    filtered: list[dict] = []
    for hit in hits:
        pid = hit.get("project_id")
        if pid is None and isinstance(hit.get("text"), str):
            # Compatibilidade com índices antigos: prefixo "[Projeto: X]" no texto.
            m = re.search(r"\[Projeto:\s*([^\]]+)\]", hit["text"])
            if m:
                pid = m.group(1).strip()
        if pid in project_ids:
            filtered.append(hit)
    return filtered


# Campos customizados que ``rows_for_file`` grava em cada chunk e que precisamos
# devolver junto com a busca para que o LLM possa citar projeto/arquivo. A busca
# semântica padrão do txtai (``emb.search(q, k)``) só retorna ``id``/``text``/``score``
# mesmo com ``content=True``; para trazer colunas customizadas é preciso usar a
# sintaxe SQL do txtai (``select … from txtai where similar(:q)``).
_RAG_SQL_COLUMNS: tuple[str, ...] = (
    "id",
    "text",
    "score",
    "project_id",
    "relative_path",
    "chunk_index",
    "cited",
)


def search_with_backend(
    backend: object,
    query: str,
    limit: int,
    *,
    project_ids: set[str] | None = None,
    retrieve_limit: int | None = None,
    hybrid_weight: float | None = None,
) -> list[dict]:
    """
    Executa busca híbrida (BM25 + semântica) ou só semântica no índice txtai.

    Receber o backend como parâmetro (em vez de criá-lo internamente) evita
    recarregar o índice txtai a cada pergunta do usuário. A instância deve
    ser gerenciada pelo chamador (ex.: ``st.cache_resource`` no Streamlit).
    A vetorização em si ocorre no serviço TEI externo.

    Busca híbrida
    -------------
    Com ``hybrid=True`` na indexação, o txtai mantém um índice BM25 sobre o
    texto bruto de cada chunk **e** o índice vetorial E5. Na consulta, funde
    os dois rankings — termos exatos (ex.: ``tampão de amostra``) sobem no
    top-K mesmo quando a busca só semântica privilegia palavras isoladas.

    O peso denso α (``RAG_HYBRID_WEIGHT``, padrão 0.4) equilibra significado
    vs. correspondência literal. Documentos **novos** indexados incrementalmente
    entram nos dois índices automaticamente; não há lista fixa de termos.

    Quando ``project_ids`` é informado, busca ``limit * 5`` candidatos e filtra
    por projeto antes de truncar — compensa hits de outros projetos no top-K bruto.

    ``retrieve_limit`` (opcional) expande o pool de candidatos antes de um rerank
    externo — ex.: buscar 24 trechos e depois reordenar para ``limit=6``.

    Implementação
    -------------
    Preferimos a forma SQL do txtai (``select <cols> from txtai where similar(:q)
    limit :k``) porque a busca semântica padrão (``backend.search(q, k)``)
    devolve apenas ``id``/``text``/``score`` mesmo com ``content=True`` — perdendo
    os campos customizados (``project_id``, ``relative_path``, ``chunk_index``,
    ``cited``) que são essenciais para o LLM citar projeto/arquivo na resposta.

    Com híbrida ativa, passamos ``similar(:query, :weight)`` para controlar α.

    Caso a query SQL falhe (índice antigo sem alguma coluna, versão do txtai
    incompatível, etc.), caímos no ``backend.search(q, k, weights=…)`` simples.
    """
    q = (query or "").strip()
    if not q:
        return []
    output_limit = int(retrieve_limit) if retrieve_limit is not None else int(limit)
    fetch_limit = output_limit * 5 if project_ids else output_limit
    params = _search_parameters(q, hybrid_weight=hybrid_weight)
    weight = params.get("weight")

    if weight is not None:
        sql = (
            f"select {', '.join(_RAG_SQL_COLUMNS)} "
            f"from txtai where similar(:query, :weight) limit {int(fetch_limit)}"
        )
    else:
        sql = (
            f"select {', '.join(_RAG_SQL_COLUMNS)} "
            f"from txtai where similar(:query) limit {int(fetch_limit)}"
        )
    try:
        raw = backend.search(sql, parameters=params)  # type: ignore[attr-defined]
        hits = format_search_results(raw)
    except Exception:
        if weight is not None:
            hits = format_search_results(
                backend.search(q, fetch_limit, weights=weight)  # type: ignore[attr-defined]
            )
        else:
            hits = format_search_results(backend.search(q, fetch_limit))  # type: ignore[attr-defined]

    hits = filter_hits_by_project(hits, project_ids)
    hits = hits[:output_limit]
    return _annotate_hits(hits, backend=backend, hybrid_weight=hybrid_weight)


# ── Formatação de contexto para o LLM ───────────────────────────────────────

# Regex de fallback para extrair Projeto/Arquivo do prefixo ``[Projeto: …]
# [Arquivo: …]`` quando o hit não vem com os campos estruturados (índice antigo
# ou backend que devolve apenas ``text``/``score``).
_PROJECT_RE = re.compile(r"\[Projeto:\s*([^\]]+)\]")
_FILE_RE = re.compile(r"\[Arquivo:\s*([^\]]+)\]")


def _hit_project_and_file(h: dict) -> tuple[str, str]:
    """
    Devolve ``(project_id, relative_path)`` de um hit, com fallback via regex.

    Preferimos os campos estruturados ``project_id``/``relative_path`` quando
    presentes. Caso venham vazios (cenário comum em ``backend.search(q, k)``
    sem SQL), tentamos extrair do prefixo embutido no ``cited``/``text``.
    """
    project_id = str(h.get("project_id") or "").strip()
    relative_path = str(h.get("relative_path") or "").strip()
    if project_id and relative_path:
        return project_id, relative_path
    body = h.get("cited") or h.get("text") or ""
    if not project_id:
        m = _PROJECT_RE.search(body)
        if m:
            project_id = m.group(1).strip()
    if not relative_path:
        m = _FILE_RE.search(body)
        if m:
            relative_path = m.group(1).strip()
    return project_id, relative_path


def format_context_for_llm(hits: list[dict], *, max_chars: int = 12000) -> str:
    """
    Monta o bloco de contexto citável a ser injetado no system prompt do chat.

    Cada hit vira uma seção cujo cabeçalho carrega explicitamente o ``project_id``
    e o ``relative_path`` do arquivo de origem, no formato:

        ### Evidência [N] — Projeto: <project_id> · Arquivo: <relative_path>

    Esse cabeçalho enriquecido é essencial para auditoria: incentiva o LLM a
    citar o nome do arquivo (e não apenas ``[N]``) na resposta final, já que o
    rótulo visível da evidência passa a conter o caminho legível. O corpo do
    bloco continua trazendo o prefixo ``[Projeto: ...] [Arquivo: ...] [Chunk N]``
    embutido em ``rows_for_file`` (redundância proposital — reforça a citação).

    Defensivo: se ``project_id``/``relative_path`` vierem ausentes no hit
    (ex.: índice antigo ou backend devolveu apenas ``text``/``score``),
    ``_hit_project_and_file`` extrai os valores via regex no próprio body.
    Assim, o cabeçalho da evidência *sempre* leva o nome do arquivo quando a
    informação estiver minimamente disponível.

    O limite de ``max_chars`` (padrão 12 000) garante que o contexto não
    ultrapasse uma fração razoável da janela de contexto do LLM. Quando
    atingido, um marcador ``(… contexto truncado …)`` é inserido para que o
    modelo saiba que há evidências omitidas e não presuma completude.
    """
    lines: list[str] = []
    used = 0
    for i, h in enumerate(hits, start=1):
        # Usa "cited" (texto completo com prefixo de projeto/arquivo) quando
        # disponível; cai em "text" para compatibilidade com índices antigos.
        body = (h.get("cited") or h.get("text") or "").strip()
        if not body:
            continue
        project_id, relative_path = _hit_project_and_file(h)
        header_parts = [f"### Evidência [{i}]"]
        meta_parts: list[str] = []
        if project_id:
            meta_parts.append(f"Projeto: {project_id}")
        if relative_path:
            meta_parts.append(f"Arquivo: {relative_path}")
        if meta_parts:
            header_parts.append("— " + " · ".join(meta_parts))
        header = " ".join(header_parts)
        block = f"{header}\n{body}\n\n"
        if used + len(block) > max_chars:
            lines.append(f"(… contexto truncado em {max_chars} caracteres …)")
            break
        lines.append(block)
        used += len(block)
    return "\n".join(lines).strip()
