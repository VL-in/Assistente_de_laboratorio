"""
Consultas em linguagem natural → SQL read-only → execução DuckDB.

O LLM remoto (OpenRouter) gera apenas ``SELECT``; o resultado entra no contexto
do chat.
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

# Máximo de linhas exibidas no texto OLAP injetado no chat (preview).
# Não altera o SQL executado — alinhado ao system prompt: LIMIT só quando
# o usuário pede quantidade específica (o LLM inclui LIMIT na query).
_MAX_PROMPT_PREVIEW_ROWS = 50
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
- Não inclua LIMIT no final da consulta a menos que o usuário peça por quantidade específica de linhas.
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


_SQL_STATEMENT_START = re.compile(
    r"^[ \t]*(SELECT|WITH)\b",
    re.IGNORECASE | re.MULTILINE,
)

# Texto que tipicamente aparece DEPOIS do SQL na resposta do LLM (pt-BR e
# en-US). Sao stop-patterns para cortar narrativa que vier no fim.
_SQL_TRAILING_NARRATIVE = re.compile(
    r"\n\s*(?:"
    r"essa\s+consulta|esta\s+consulta|isso\s+retorna|"
    r"o\s+resultado|a\s+query|nota[: ]|observa[çc][ãa]o|explica[çc][ãa]o|"
    r"this\s+query|note\s*:|explanation\s*:|the\s+result|that\s+will"
    r")",
    re.IGNORECASE,
)


def _truncate_at_first_top_level_semicolon(sql: str) -> str:
    """
    Retorna ``sql`` ate o primeiro ``;`` fora de string literal/comentario.

    DuckDB so executa uma instrucao por chamada; manter o ``;`` faria o
    validator rejeitar com mensagem confusa. Como literais sao mais raros
    que a forma direta, o parser e linear e simples (sem AST).
    """
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        if ch == "'":
            # avanca ate o proximo ' que nao seja '' (escape)
            i += 1
            while i < n:
                if sql[i] == "'":
                    if i + 1 < n and sql[i + 1] == "'":
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        if ch == '"':
            i += 1
            while i < n and sql[i] != '"':
                i += 1
            i += 1
            continue
        if ch == "-" and i + 1 < n and sql[i + 1] == "-":
            nl = sql.find("\n", i)
            if nl == -1:
                return sql[:i].rstrip()
            i = nl + 1
            continue
        if ch == "/" and i + 1 < n and sql[i + 1] == "*":
            end = sql.find("*/", i + 2)
            if end == -1:
                return sql[:i].rstrip()
            i = end + 2
            continue
        if ch == ";":
            return sql[:i].rstrip()
        i += 1
    return sql.rstrip()


def _cut_trailing_narrative(sql: str) -> str:
    """Corta texto explicativo apos a SQL (pt-BR e en-US)."""
    match = _SQL_TRAILING_NARRATIVE.search(sql)
    if match:
        return sql[: match.start()].rstrip()
    return sql


def _extract_sql(text: str) -> tuple[str, bool]:
    """
    Extrai SQL da resposta do LLM.

    Retorna ``(sql, truncated)``. ``truncated`` e ``True`` quando ha indicio
    de que a resposta foi cortada no meio do bloco ``<think>`` (modelo
    gastou todos os tokens raciocinando).

    Estrategia de extracao (em ordem):
    1. Remove blocos ``<think>``/``<reasoning>``.
    2. Se ha ```` ```sql ... ``` ```` (ou ```` ``` ... ``` `````), retorna o
       conteudo do primeiro bloco.
    3. Procura ``SELECT``/``WITH`` em **inicio de linha** (re.MULTILINE),
       captura ate o fim do texto e:
       a. corta no primeiro ``;`` fora de string;
       b. corta no primeiro stop-pattern de narrativa (pt-BR ou en-US).
    4. Sem nenhuma das opcoes acima, retorna o texto compacto como esta
       (deixa o validator decidir).

    Exigir que ``SELECT``/``WITH`` esteja em inicio de linha evita confundir
    a palavra ``SELECT`` mencionada no meio de uma frase em portugues
    (ex.: "Vou usar SELECT mas precisa pensar...") com o inicio da query.
    """
    cleaned, truncated = _strip_think_blocks(text)

    block = re.search(r"```(?:sql)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    if block:
        sql = block.group(1).strip()
        sql = _truncate_at_first_top_level_semicolon(sql)
        return sql, truncated

    match = _SQL_STATEMENT_START.search(cleaned)
    if match:
        sql = cleaned[match.start():].strip()
        sql = _cut_trailing_narrative(sql)
        sql = _truncate_at_first_top_level_semicolon(sql)
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
    catalog = build_schema_catalog_text(sample_rows=0)

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
            generation_name="olap-nl-to-sql",
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}", ""

    if not raw:
        return None, (
            "O LLM retornou resposta vazia. Verifique a chave `OPENROUTER_API_KEY` "
            "no `.env` e teste em **Desenvolvimento → Diagnóstico → Testar GET /v1/models**."
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
    return sql, None, raw


def execute_sql(sql: str) -> tuple[pd.DataFrame | None, str | None]:
    ok, err = validate_readonly_sql(sql)
    if not ok:
        return None, err
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
    preview = df.head(_MAX_PROMPT_PREVIEW_ROWS)
    text = preview.to_string(index=False)
    if len(df) > _MAX_PROMPT_PREVIEW_ROWS:
        text += (
            f"\n(… exibindo {_MAX_PROMPT_PREVIEW_ROWS} de {len(df)} linha(s) "
            "no contexto do chat; consulta completa na UI OLAP …)"
        )
    return text


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
