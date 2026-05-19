"""Extração de texto bruto de documentos do inventário (docx, planilhas, pdf, texto)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from projects_loader import ScannedFile


@dataclass
class ExtractOutcome:
    """Resultado da extração de um arquivo."""

    text: str
    detail: str  # contexto humano (aba, aviso de truncamento)
    ok: bool


def extract_from_scanned_file(sf: ScannedFile, *, max_chars_total: int = 2_000_000) -> ExtractOutcome:
    """
    Lê o arquivo apontado por ``sf`` e devolve texto plano.

    ``max_chars_total`` evita carregar PDF/planilhas gigantes na RAM de uma vez.
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
    except Exception as exc:  # noqa: BLE001 — queremos continuar o pipeline
        return ExtractOutcome(text="", detail=f"Erro ao ler arquivo: {exc}", ok=False)

    return ExtractOutcome(text="", detail=f"Extensão não suportada para extração: {suffix}", ok=False)


def _truncate(s: str, limit: int) -> tuple[str, bool]:
    if len(s) <= limit:
        return s, False
    return s[:limit], True


def _iter_docx_blocks(document: object):
    """
    Percorre parágrafos e tabelas na ordem do documento Word.

    ``python-docx`` expõe ``paragraphs`` e ``tables`` separados; insumos e validades
  costumam estar em **tabelas**, que ficavam de fora na extração antiga.
    """
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    body = document.element.body  # type: ignore[attr-defined]
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def _table_row_cell_texts(row: object) -> list[str]:
    """Textos das células de uma linha (deduplica células repetidas por merge no Word)."""
    seen_tc: set[int] = set()
    texts: list[str] = []
    for cell in row.cells:  # type: ignore[attr-defined]
        tc_id = id(cell._tc)
        if tc_id in seen_tc:
            continue
        seen_tc.add(tc_id)
        t = (cell.text or "").strip().replace("\n", " ")
        texts.append(t)
    return texts


def _table_to_lines(table: object) -> list[str]:
    lines: list[str] = []
    for row in table.rows:  # type: ignore[attr-defined]
        cells = _table_row_cell_texts(row)
        line = "\t".join(cells).strip()
        if line:
            lines.append(line)
    return lines


def _extract_docx(path: Path, *, max_chars_total: int) -> ExtractOutcome:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    parts: list[str] = []
    n_tables = 0
    n_paras = 0

    for block in _iter_docx_blocks(doc):
        if isinstance(block, Paragraph):
            t = (block.text or "").strip()
            if t:
                parts.append(t)
                n_paras += 1
        elif isinstance(block, Table):
            table_lines = _table_to_lines(block)
            if table_lines:
                parts.append("### Tabela de insumos / materiais")
                parts.extend(table_lines)
                n_tables += 1

    text = "\n".join(parts)
    text, cut = _truncate(text, max_chars_total)
    detail = f"docx: {n_paras} parágrafo(s), {n_tables} tabela(s)"
    if cut:
        detail += f" (truncado a {max_chars_total} caracteres)"
    return ExtractOutcome(text=text, detail=detail, ok=bool(text))


def _extract_excel(path: Path, *, max_chars_total: int) -> ExtractOutcome:
    from openpyxl import load_workbook

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
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            text = ""
    else:
        text = raw.decode("utf-8", errors="replace")
    text = text.strip()
    text, cut = _truncate(text, max_chars_total)
    detail = "texto plano"
    if cut:
        detail += f" (truncado a {max_chars_total} caracteres)"
    return ExtractOutcome(text=text, detail=detail, ok=bool(text))


def _extract_csv(path: Path, *, max_chars_total: int) -> ExtractOutcome:
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
