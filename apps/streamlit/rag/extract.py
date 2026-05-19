"""
Extração de texto plano a partir dos documentos do inventário.

Funciona como um **dispatcher por extensão de arquivo**: a função pública
``extract_from_scanned_file`` identifica o tipo do arquivo e delega para o
extrator específico. Todos os erros de parsing são capturados internamente e
devolvidos em ``ExtractOutcome.ok = False``, para que uma falha em um arquivo
não interrompa o pipeline inteiro de indexação.

Formatos suportados
-------------------
- ``.docx`` — parágrafos + tabelas em ordem de documento (python-docx)
- ``.xlsx`` / ``.xlsm`` — todas as abas (openpyxl, modo read-only)
- ``.pdf`` — texto por página (pypdf)
- ``.txt`` / ``.md`` — texto plano com detecção de encoding
- ``.csv`` — células separadas por tabulação

Truncagem
---------
Todos os extratores respeitam ``max_chars_total`` para evitar carregar arquivos
muito grandes inteiramente na RAM. Arquivos truncados são indexados normalmente
(o texto disponível já cobre a maioria dos casos práticos).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from projects_loader import ScannedFile


# ── Resultado da extração ────────────────────────────────────────────────────

@dataclass
class ExtractOutcome:
    """
    Resultado de uma tentativa de extração de texto.

    Atributos
    ---------
    text:
        Texto bruto extraído. Vazio se a extração falhou ou o arquivo não
        contém texto legível.
    detail:
        Mensagem humana descrevendo o resultado: número de parágrafos/tabelas,
        abas processadas, aviso de truncagem ou mensagem de erro. Exibida na
        UI de indexação e gravada nos metadados do chunk.
    ok:
        ``True`` se o texto foi extraído com sucesso (mesmo que truncado).
        ``False`` indica falha de parsing ou arquivo sem conteúdo.
    """

    text: str
    detail: str
    ok: bool


# ── Roteador principal ───────────────────────────────────────────────────────

def extract_from_scanned_file(sf: ScannedFile, *, max_chars_total: int = 2_000_000) -> ExtractOutcome:
    """
    Seleciona o extrator adequado para ``sf`` e retorna o texto plano.

    Qualquer exceção de parsing é capturada e devolvida como
    ``ExtractOutcome(ok=False)``, para que o pipeline de indexação continue
    processando os demais arquivos mesmo quando um documento está corrompido
    ou tem formato inesperado.

    Parâmetros
    ----------
    sf:
        Arquivo escaneado com caminho absoluto e metadados.
    max_chars_total:
        Limite de caracteres do texto final. Evita alocar GBs de memória para
        PDFs ou planilhas muito grandes. O padrão de 2 M chars cobre a grande
        maioria dos documentos de laboratório.
    """
    path = sf.absolute_path
    suffix = path.suffix.lower()

    try:
        if suffix == ".docx":
            return _extract_docx(path, max_chars_total=max_chars_total)
        if suffix in (".xlsx", ".xlsm"):
            return _extract_excel(path, max_chars_total=max_chars_total)
        if suffix == ".pdf":
            return _extract_pdf(path, max_chars_total=max_chars_total)
        if suffix in (".txt", ".md"):
            return _extract_plain(path, max_chars_total=max_chars_total)
        if suffix == ".csv":
            return _extract_csv(path, max_chars_total=max_chars_total)
    except Exception as exc:  # noqa: BLE001 — falha isolada; pipeline não deve parar
        return ExtractOutcome(text="", detail=f"Erro ao ler arquivo: {exc}", ok=False)

    return ExtractOutcome(text="", detail=f"Extensão não suportada para extração: {suffix}", ok=False)


# ── Helper de truncagem ──────────────────────────────────────────────────────

def _truncate(s: str, limit: int) -> tuple[str, bool]:
    """Retorna ``(texto, foi_truncado)``."""
    if len(s) <= limit:
        return s, False
    return s[:limit], True


# ── Helpers para documentos Word (.docx) ────────────────────────────────────

def _walk_block_container(element: object, parent: object):
    """
    Percorre parágrafos e tabelas em ordem de documento dentro de um contêiner.

    Além de ``<w:p>`` e ``<w:tbl>`` diretos, desce em:
    - ``<w:sdt>`` (controles de conteúdo / formulários Word — comum em templates)
    - ``<w:ins>`` (revisões rastreadas)
    - ``<w:smartTag>`` (marcadores legados)

    Sem isso, tabelas de insumos embutidas em controles de conteúdo ficam invisíveis
    para a busca semântica mesmo com ``python-docx`` instalado.
    """
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in element.iterchildren():  # type: ignore[attr-defined]
        tag = child.tag
        if tag == qn("w:p"):
            yield Paragraph(child, parent)
        elif tag == qn("w:tbl"):
            yield Table(child, parent)
        elif tag == qn("w:sdt"):
            content = child.find(qn("w:sdtContent"))
            if content is not None:
                yield from _walk_block_container(content, parent)
        elif tag in (qn("w:ins"), qn("w:smartTag")):
            yield from _walk_block_container(child, parent)


def _iter_docx_blocks(document: object):
    """Blocos do corpo principal do documento (``<w:body>``)."""
    yield from _walk_block_container(document.element.body, document)  # type: ignore[attr-defined]


def _iter_hdr_ftr_blocks(header_footer: object):
    """Blocos de cabeçalho ou rodapé (quando não vinculado ao anterior)."""
    if getattr(header_footer, "is_linked_to_previous", False):
        return
    yield from _walk_block_container(header_footer._element, header_footer)  # type: ignore[attr-defined]


def _cell_to_text(cell: object) -> str:
    """
    Texto de uma célula, incluindo parágrafos e tabelas aninhadas.

    ``cell.text`` do python-docx omite tabelas aninhadas; percorrer os blocos
    da célula garante que sub-tabelas de insumos entrem na indexação.
    """
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    pieces: list[str] = []
    for block in _walk_block_container(cell._tc, cell):  # type: ignore[attr-defined]
        if isinstance(block, Paragraph):
            t = (block.text or "").strip()
            if t:
                pieces.append(t)
        elif isinstance(block, Table):
            for line in _table_to_lines(block):
                pieces.append(line.replace("\t", " | "))
    if not pieces:
        t = (cell.text or "").strip().replace("\n", " ")  # type: ignore[attr-defined]
        if t:
            pieces.append(t)
    return " | ".join(pieces)


def _table_row_cell_texts(row: object) -> list[str]:
    """
    Retorna os textos das células de uma linha, sem duplicar células mescladas.

    Em tabelas Word com células mescladas (merge), ``row.cells`` pode retornar
    o mesmo objeto ``_tc`` (nó XML da célula) mais de uma vez. A deduplicação
    por ``id(cell._tc)`` garante que cada célula física apareça apenas uma vez
    na linha extraída.
    """
    seen_tc: set[int] = set()
    texts: list[str] = []
    for cell in row.cells:  # type: ignore[attr-defined]
        tc_id = id(cell._tc)
        if tc_id in seen_tc:
            continue
        seen_tc.add(tc_id)
        texts.append(_cell_to_text(cell))
    return texts


def _table_to_lines(table: object) -> list[str]:
    """Serializa uma tabela Word em linhas de texto separadas por tabulação."""
    lines: list[str] = []
    for row in table.rows:  # type: ignore[attr-defined]
        cells = _table_row_cell_texts(row)
        line = "\t".join(cells).strip()
        if line:
            lines.append(line)
    return lines


def _append_docx_blocks(
    blocks: object,
    parts: list[str],
    *,
    n_paras: list[int],
    n_tables: list[int],
) -> None:
    """Acumula parágrafos e tabelas serializadas em ``parts``."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for block in blocks:
        if isinstance(block, Paragraph):
            t = (block.text or "").strip()
            if t:
                parts.append(t)
                n_paras[0] += 1
        elif isinstance(block, Table):
            table_lines = _table_to_lines(block)
            if table_lines:
                parts.append("### Tabela de insumos / materiais")
                parts.extend(table_lines)
                n_tables[0] += 1


# ── Extratores por formato ───────────────────────────────────────────────────

def _extract_docx(path: Path, *, max_chars_total: int) -> ExtractOutcome:
    """
    Extrai texto de um arquivo Word, preservando parágrafos e tabelas em ordem.

    Cada tabela recebe um cabeçalho ``### Tabela de insumos / materiais`` para
    sinalizar ao modelo de linguagem que o bloco seguinte é tabular — útil para
    perguntas sobre lotes, validades e fabricantes.
    """
    from docx import Document

    doc = Document(str(path))
    parts: list[str] = []
    n_tables = [0]
    n_paras = [0]

    _append_docx_blocks(_iter_docx_blocks(doc), parts, n_paras=n_paras, n_tables=n_tables)

    for section in doc.sections:
        for hf in (section.header, section.footer):
            _append_docx_blocks(
                _iter_hdr_ftr_blocks(hf),
                parts,
                n_paras=n_paras,
                n_tables=n_tables,
            )

    text = "\n".join(parts)
    n_paras_val, n_tables_val = n_paras[0], n_tables[0]
    text, cut = _truncate(text, max_chars_total)
    detail = f"docx: {n_paras_val} parágrafo(s), {n_tables_val} tabela(s)"
    if cut:
        detail += f" (truncado a {max_chars_total} caracteres)"
    return ExtractOutcome(text=text, detail=detail, ok=bool(text))


def _extract_excel(path: Path, *, max_chars_total: int) -> ExtractOutcome:
    """
    Extrai todas as abas de uma planilha Excel como texto tabular.

    Usa dois mecanismos de truncagem:
    1. Por linha: cada aba tem limite de 50 000 linhas para evitar loop muito
       longo em planilhas com muitas linhas em branco.
    2. Global: contador ``remaining`` para o total de caracteres. Quando
       esgotado, retorna imediatamente sem processar o restante das abas.

    Células individuais com mais de 2 000 chars são truncadas com ``…`` para
    evitar que fórmulas ou textos longos dominem o espaço de um único chunk.
    """
    from openpyxl import load_workbook

    # read_only=True e data_only=True: mais rápido e evita calcular fórmulas.
    wb = load_workbook(filename=str(path), read_only=True, data_only=True)
    lines: list[str] = []
    remaining = max_chars_total

    try:
        for sheet in wb.worksheets:
            lines.append(f"### Planilha: {sheet.title}")
            row_limit = 50_000
            for i, row in enumerate(sheet.iter_rows(values_only=True)):
                if i > row_limit:
                    lines.append(f"(… planilha {sheet.title} truncada após {row_limit} linhas …)")
                    break
                cells = []
                for cell in row:
                    if cell is None:
                        cells.append("")
                    else:
                        s = str(cell).strip()
                        if len(s) > 2000:
                            s = s[:2000] + "…"
                        cells.append(s)
                line = "\t".join(cells).strip()
                if line:
                    lines.append(line)
                    remaining -= len(line) + 1
                    if remaining <= 0:
                        lines.append("(… truncado por limite global de caracteres …)")
                        text = "\n".join(lines)
                        return ExtractOutcome(
                            text=text,
                            detail="xlsx: múltiplas abas (truncado)",
                            ok=True,
                        )
    finally:
        wb.close()

    text = "\n".join(lines).strip()
    text, extra_cut = _truncate(text, max_chars_total)
    detail = "xlsx: abas e linhas"
    if extra_cut:
        detail += f" (truncado a {max_chars_total} caracteres)"
    if not text:
        return ExtractOutcome(text="", detail="xlsx: sem células com conteúdo", ok=False)
    return ExtractOutcome(text=text, detail=detail, ok=True)


def _extract_pdf(path: Path, *, max_chars_total: int) -> ExtractOutcome:
    """
    Extrai texto de um PDF página a página.

    Erros em páginas individuais (PDF corrompido, página de imagem sem OCR) são
    registrados como marcadores de texto ``[página N: erro ...]`` e o
    processamento continua nas demais páginas.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    parts: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            parts.append(f"[página {i + 1}: erro {exc}]")
            continue
        if t.strip():
            parts.append(t)
    text = "\n\n".join(parts)
    text, cut = _truncate(text, max_chars_total)
    detail = "pdf: texto extraído"
    if cut:
        detail += f" (truncado a {max_chars_total} caracteres)"
    return ExtractOutcome(text=text, detail=detail, ok=bool(text))


def _extract_plain(path: Path, *, max_chars_total: int) -> ExtractOutcome:
    """
    Lê um arquivo de texto simples (.txt ou .md) com detecção de encoding.

    A ordem de tentativa é: UTF-8 → UTF-8 com BOM → latin-1. Latin-1 é o
    fallback final porque mapeia todos os 256 valores de byte sem lançar
    ``UnicodeDecodeError``, garantindo que o loop sempre termine com ``break``
    e que ``text`` sempre tenha um valor atribuído.
    """
    raw = path.read_bytes()
    text = ""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    text = text.strip()
    text, cut = _truncate(text, max_chars_total)
    detail = "texto plano"
    if cut:
        detail += f" (truncado a {max_chars_total} caracteres)"
    return ExtractOutcome(text=text, detail=detail, ok=bool(text))


def _extract_csv(path: Path, *, max_chars_total: int) -> ExtractOutcome:
    """
    Lê um CSV e serializa cada linha como campos separados por tabulação.

    Limite de 100 000 linhas por arquivo para evitar bloqueio prolongado em
    dumps de banco exportados acidentalmente.
    """
    import csv

    lines: list[str] = []
    remaining = max_chars_total
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        for i, row in enumerate(reader):
            if i > 100_000:
                lines.append("(… CSV truncado após 100k linhas …)")
                break
            line = "\t".join(str(c) for c in row)
            lines.append(line)
            remaining -= len(line) + 1
            if remaining <= 0:
                lines.append("(… truncado por limite global …)")
                break
    text = "\n".join(lines).strip()
    text, cut = _truncate(text, max_chars_total)
    detail = "csv"
    if cut:
        detail += f" (truncado a {max_chars_total} caracteres)"
    return ExtractOutcome(text=text, detail=detail, ok=bool(text))
