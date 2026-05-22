"""
Consultas em linguagem natural → SQL read-only → execução DuckDB.

O LLM (LM Studio) gera apenas ``SELECT``; o resultado entra no contexto do chat.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from openai import OpenAI

from qwen35_inference import (
    DEFAULT_SQL_MAX_TOKENS,
    PROFILE_OLAP_SQL,
    create_chat_completion,
    strip_thinking_blocks_with_flag,
)

from .connection import open_duckdb
from .ingest import has_ingested_tables
from .schema_catalog import build_schema_catalog_text

_MAX_RESULT_ROWS = 50
_SQL_MAX_TOKENS = DEFAULT_SQL_MAX_TOKENS

# Palavras proibidas em SELECTs de leitura. Inclui DDL/DML e administracao
# (PRAGMA/SET/ATTACH) que poderiam alterar estado do DuckDB ou expor dados.
_FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|REPLACE|UPSERT|"
    r"DROP|CREATE|ALTER|TRUNCATE|"
    r"ATTACH|DETACH|COPY|EXPORT|IMPORT|"
    r"LOAD|INSTALL|SET|RESET|PRAGMA|"
    r"CALL|EXEC|EXECUTE|"
    r"VACUUM|CHECKPOINT|ANALYZE|"
    r"GRANT|REVOKE)\b",
    re.IGNORECASE,
)

# Strings literais ('...' e "..."), comentarios de linha (-- ...) e de bloco
# (/* ... */). Removidos antes da checagem de palavras proibidas para evitar
# falsos positivos (ex.: WHERE status = 'DROP-OUT').
_SQL_LITERAL_PATTERN = re.compile(
    r"'(?:[^']|'')*'"        # string com aspas simples (DuckDB escapa com '')
    r"|\"(?:[^\"]|\"\")*\""  # identificador entre aspas duplas
    r"|--[^\n]*"             # comentario de linha
    r"|/\*.*?\*/",           # comentario de bloco
    re.DOTALL,
)


def _strip_sql_literals_and_comments(sql: str) -> str:
    """
    Retorna o SQL sem strings literais, identificadores entre aspas e
    comentarios. Usado **apenas** para validacao; o SQL executado e o original.
    """
    return _SQL_LITERAL_PATTERN.sub(" ", sql)

# Qwen3.5: desligar thinking via API (``enable_thinking=False``), não via ``/no_think``
# no prompt — ver ``qwen35_inference.PROFILE_OLAP_SQL``.
_TEXT_TO_SQL_SYSTEM = """Você traduz perguntas em português para SQL DuckDB.

Regras obrigatórias:
- Retorne APENAS uma consulta SQL, sem explicação antes ou depois.
- Use somente SELECT ou WITH ... SELECT (leitura).
- Identifique tabelas e colunas EXCLUSIVAMENTE pelo catálogo fornecido.
- Use aspas duplas nos nomes de tabelas quando contiverem caracteres especiais: "nome_tabela".
- Inclua LIMIT 50 no final se a consulta puder retornar muitas linhas, exceto quando o usuário pedir mais linhas.
- Não invente tabelas nem colunas que não estejam no catálogo.
- Para filtrar por projeto, use a coluna _project_id.
- Para filtrar por arquivo, use _source_file ou _sheet_name.
"""


@dataclass
class OlapQueryResult:
    """Resultado de uma pergunta OLAP via linguagem natural."""

    ok: bool
    sql: str | None = None
    dataframe: pd.DataFrame | None = None
    error: str | None = None
    context_for_llm: str = ""
    raw_llm_response: str = ""


def _strip_think_blocks(text: str) -> tuple[str, bool]:
    """Remove blocos de raciocínio; delega a ``qwen35_inference``."""
    return strip_thinking_blocks_with_flag(text)


def _extract_sql(text: str) -> tuple[str, bool]:
    """
    Extrai SQL da resposta do LLM.

    Retorna ``(sql, truncated)``. ``truncated`` é ``True`` quando há indício de
    que a resposta foi cortada no meio do bloco ``<think>`` (modelo gastou
    todos os tokens raciocinando).

    Trata casos comuns:
    - Bloco ```sql ... ```
    - Tags <think>...</think> (modelos que pensam antes), inclusive sem fechamento
    - Texto explicativo antes do SELECT
    """
    cleaned, truncated = _strip_think_blocks(text)

    block = re.search(r"```(?:sql)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    if block:
        return block.group(1).strip(), truncated

    match = re.search(r"(SELECT\b.*|WITH\b.*)", cleaned, re.DOTALL | re.IGNORECASE)
    if match:
        sql = match.group(1).strip()
        end_patterns = [
            r"\n\n[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ]",
            r"\nEssa consulta",
            r"\nEsta consulta",
            r"\nIsso retorna",
            r"\nO resultado",
        ]
        for pattern in end_patterns:
            end_match = re.search(pattern, sql, re.IGNORECASE)
            if end_match:
                sql = sql[: end_match.start()].strip()
                break
        return sql, truncated

    lines = [ln for ln in cleaned.strip().splitlines() if ln.strip()]
    return "\n".join(lines).strip(), truncated


def validate_readonly_sql(sql: str) -> tuple[bool, str]:
    """
    Valida que a consulta é somente leitura.

    Comparacoes ignoram strings literais e comentarios — uma query como
    ``SELECT * FROM x WHERE status = 'DROP-OUT'`` NAO e bloqueada apenas
    porque a palavra ``DROP`` aparece dentro de aspas.

    Regras:
    1. SQL nao pode ser vazio.
    2. Apos remover whitespace e ``;`` final unico, deve comecar com
       ``SELECT`` ou ``WITH``.
    3. Apos remover strings/identificadores entre aspas e comentarios, nao
       pode conter palavras proibidas (ver ``_FORBIDDEN_SQL``).
    4. Apos remover literais e comentarios, nao pode conter ``;`` (apenas
       uma instrucao por chamada).
    """
    s = sql.strip().rstrip(";").strip()
    if not s:
        return False, "SQL vazio."
    upper = s.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return False, "Apenas SELECT ou WITH ... SELECT são permitidos."
    sanitized = _strip_sql_literals_and_comments(s)
    if _FORBIDDEN_SQL.search(sanitized):
        return False, "Comando SQL não permitido (somente leitura)."
    if ";" in sanitized:
        return False, "Uma única instrução SQL por vez."
    return True, ""


def _ensure_limit(sql: str, limit: int = _MAX_RESULT_ROWS) -> str:
    if re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
        return sql
    return f"{sql.rstrip()}\nLIMIT {limit}"


def generate_sql(
    question: str,
    *,
    client: OpenAI,
    model: str,
    max_tokens: int = _SQL_MAX_TOKENS,
) -> tuple[str | None, str | None, str]:
    """
    Pede ao LLM o SQL; retorna ``(sql, erro, raw_llm_response)``.

    O terceiro elemento sempre traz a resposta bruta do modelo (vazio quando a
    chamada falha antes de receber resposta) — útil para o painel de
    diagnóstico mostrar exatamente o que o LLM devolveu quando o SQL não pôde
    ser extraído.
    """
    catalog = build_schema_catalog_text()

    if catalog.startswith("("):
        return None, f"Catálogo vazio: {catalog}", ""

    user = f"Catálogo do banco:\n\n{catalog}\n\nPergunta do usuário:\n{question.strip()}"
    try:
        completion = create_chat_completion(
            client,
            messages=[
                {"role": "system", "content": _TEXT_TO_SQL_SYSTEM},
                {"role": "user", "content": user},
            ],
            model=model,
            profile=PROFILE_OLAP_SQL,
            max_tokens=max_tokens,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}", ""

    if not raw:
        return None, (
            "O LLM retornou resposta vazia. Verifique se o modelo está carregado no "
            "LM Studio e se a aba **Diagnóstico → Testar GET /v1/models** lista o modelo."
        ), raw

    sql, truncated = _extract_sql(raw)
    if truncated and not sql:
        return None, (
            "O modelo gastou todos os tokens no bloco de raciocínio (<think>) e não "
            f"chegou a gerar SQL. Aumente `max_tokens` (atual {max_tokens}) ou use um "
            "modelo sem raciocínio embutido."
        ), raw

    ok, err = validate_readonly_sql(sql)
    if not ok:
        return None, err, raw
    return _ensure_limit(sql), None, raw


def execute_sql(sql: str) -> tuple[pd.DataFrame | None, str | None]:
    ok, err = validate_readonly_sql(sql)
    if not ok:
        return None, err
    sql = _ensure_limit(sql)
    from .ingest import database_exists_quick

    if not database_exists_quick():
        return None, "Banco DuckDB não existe. Escaneie as pastas primeiro."
    conn = open_duckdb(read_only=True)
    try:
        df = conn.execute(sql).fetchdf()
        return df, None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        conn.close()


def _format_dataframe_for_prompt(df: pd.DataFrame) -> str:
    if df.empty:
        return "(consulta executada; zero linhas)"
    preview = df.head(_MAX_RESULT_ROWS)
    return preview.to_string(index=False)


def run_nl_olap_query(
    question: str,
    *,
    client: OpenAI,
    model: str,
) -> OlapQueryResult:
    """
    Fluxo completo: pergunta → SQL (LLM) → execução → texto para o chat.
    """
    if not has_ingested_tables():
        return OlapQueryResult(
            ok=False,
            error="Nenhuma planilha no DuckDB. Escaneie as pastas na barra lateral.",
            context_for_llm="",
        )

    sql, gen_err, raw = generate_sql(question, client=client, model=model)
    if gen_err or not sql:
        return OlapQueryResult(
            ok=False,
            sql=sql,
            error=gen_err or "SQL não gerado.",
            raw_llm_response=raw,
        )

    df, run_err = execute_sql(sql)
    if run_err:
        return OlapQueryResult(
            ok=False, sql=sql, error=run_err, raw_llm_response=raw
        )

    table_text = _format_dataframe_for_prompt(df)  # type: ignore[arg-type]
    ctx = (
        f"### Consulta OLAP (DuckDB)\n"
        f"Pergunta analítica: {question.strip()}\n"
        f"SQL executado:\n```sql\n{sql}\n```\n"
        f"Resultado ({len(df)} linha(s)):\n```\n{table_text}\n```\n"
        "Use estes dados tabulares para responder. Se o resultado estiver vazio, diga que não encontrou."
    )
    return OlapQueryResult(
        ok=True,
        sql=sql,
        dataframe=df,
        context_for_llm=ctx,
        raw_llm_response=raw,
    )
