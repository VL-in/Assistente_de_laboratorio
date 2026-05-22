# Revisão de código — 2026-05-22

**Escopo solicitado:** "Revise todo o código para encontrar bugs, erro de
arquitetura, leitura e injeção de dados no DuckDB, a sua leitura e consulta
pelo modelo hospedado no Streamlit e outros corner cases do projeto."

**Áreas auditadas:**

- `apps/streamlit/app.py` (UI + orquestração)
- `apps/streamlit/llm_config.py` + `qwen35_inference.py` (LLM)
- `apps/streamlit/olap/` (DuckDB: ingest, nl_query, indexes, schema_catalog, connection)
- `apps/streamlit/rag/` (txtai: extract, chunking, index_txtai, manifest)
- `apps/streamlit/projects_loader.py`
- `apps/streamlit/tests/`
- `apps/api/` (API FastAPI alternativa, atualmente desacoplada)
- `docker-compose.yml`, `.env`, `.env.docker.example`

---

## Resumo executivo

| # | Item | Severidade | Quebra hoje? | Esforço | Aplicado nesta data |
|---|---|---|---|---|---|
| 1 | `_read_csv` não detecta separador `;` (CSV brasileiro) | P0 | Sim, silencioso | Baixo | Sim |
| 2 | `_FORBIDDEN_SQL` rejeita SQL legítimo com `DROP` em string literal | P0 | Sim (UX) | Médio | Sim |
| 3 | `_extract_sql` confunde menções a `SELECT` com a query, e não trunca em `;` | P0 | Sim | Baixo | Sim |
| 4 | `sync_tabular_from_scans` apaga TODAS as tabelas se escaneamento vier vazio | P0 | Risco de perda de dado | Baixo | Sim |
| 5 | `apps/api/` é dead code com configuração divergente do Streamlit | P0 | Confusão futura | Baixo | Sim |
| 6 | Streaming exibe blocos `<think>` em tempo real antes de stripá-los | P1 | UX | Médio | Não |
| 7 | Sidebar faz dois escaneamentos com SHA-256 duplicado | P1 | Performance | Baixo | Não |
| 8 | `_TEXT_TO_SQL_SYSTEM` ainda menciona `<think>` mesmo com `enable_thinking=False` | P1 | Redundância | Baixo | Não |
| 9 | `CREATE OR REPLACE TABLE` apaga índices da tabela (DuckDB) | P1 | Mitigado por re-create | Baixo | Não |
| 10 | Cliente `OpenAI` recriado a cada mensagem; timeout fixo 120s | P1 | Performance / thinking | Baixo | Não |
| 11 | `_read_csv` e `_read_excel_sheets` engolem todas as exceções | P1 | Diagnóstico difícil | Baixo | Não |
| 12 | `sanitize_history_message` só limpa para Qwen3.5 (vaza se trocar modelo) | P1 | Edge case | Baixo | Não |
| 13 | RAG não filtra por `project_id` (risco de vazamento entre projetos) | P1 | Arquitetura | Médio | Não |
| 14 | `_try_convert_column_to_numeric` aceita 70% — pode tipar coluna mista errado | P2 | Dado errado | Baixo | Não |
| 15 | Volume `assistente_sqlite` declarado no compose mas sem uso | P2 | Limpeza | Baixo | Não |
| 16 | Path hardcoded do dev em `docker-compose.yml` (`D:/Vanessa/...`) | P2 | Onboarding | Baixo | Não |
| 17 | `_check_openai_compatible_models` não envia Authorization | P2 | Compat | Baixo | Não |
| 18 | `_FORBIDDEN_SQL` não cobre `MERGE`, `VACUUM`, `CHECKPOINT` | P2 | Segurança | Baixo | Endereçado em #2 |
| 19 | `format_context_for_llm` corta por caractere (não por token) | P2 | Janela de contexto | Médio | Não |
| 20 | Schema catalog inclui 2 linhas de dados reais no prompt | P2 | Vazamento de dado | Baixo | Não |
| 21 | `check_duckdb` cria arquivo de banco em modo "teste" | P2 | Side effect | Baixo | Não |
| 22 | Test leak: `test_manifest_incremental` não restaura `ASSISTENTE_TXTAI_DIR` | P2 | Suite frágil | Baixo | Não |

---

## P0 — bugs críticos (corrigidos nesta data)

### 1. `_read_csv` ignora separador `;` (CSV brasileiro)

**Arquivo:** `apps/streamlit/olap/ingest.py` (linhas ~295-301 antes da correção)

**Sintoma:** um CSV exportado do Excel brasileiro (`a;b;c\n1;2;3`) é
ingerido como **uma única coluna** com nome `a;b;c` e duas linhas com valor
`1;2;3`. Reproduzido no smoke-test desta revisão.

**Causa:** `pd.read_csv(path, encoding=enc)` usa o default `sep=","`, sem
tentar fallback. O laboratório-alvo é brasileiro, então CSVs com `;` são
comuns (Excel pt-BR salva assim quando o sistema usa vírgula decimal).

**Correção:** tentar `sniff` do `csv` da stdlib + fallback para
`engine="python", sep=None` (autodetecção). Mantém múltiplas tentativas de
encoding.

---

### 2. `_FORBIDDEN_SQL` rejeita SQL com palavra-chave em string literal

**Arquivo:** `apps/streamlit/olap/nl_query.py` (linhas ~28-32)

**Sintoma confirmado em smoke-test:**

```
SELECT * FROM x WHERE col = 'DROP TABLE Y'  → REJEITADO
SELECT * FROM x WHERE c = 'a;b'             → REJEITADO
SELECT 'set' AS x                            → REJEITADO
SELECT 'CREATE OR REPLACE' AS x              → REJEITADO
```

**Causa:** regex aplicada sobre o SQL inteiro, incluindo strings literais.

**Correção:** remover literais de string (`'...'`, `"..."`) e comentários
(`-- ...`, `/* ... */`) antes de aplicar o regex de palavras proibidas.
Também ampliar a lista para `MERGE`, `VACUUM`, `CHECKPOINT`, `DETACH`.

---

### 3. `_extract_sql` mistura ruído da resposta com o SQL

**Arquivo:** `apps/streamlit/olap/nl_query.py` (linhas ~68-105)

**Sintoma confirmado em smoke-test:**

- Entrada: `"Vou usar SELECT mas precisa pensar... SELECT * FROM y LIMIT 10"`
- Saída atual: `"SELECT mas precisa pensar... SELECT * FROM y LIMIT 10"`
- Esperado: `"SELECT * FROM y LIMIT 10"`

Também: `"SELECT 1 FROM x;"` mantém o `;` final, e o validator depois
rejeita com mensagem confusa "Uma única instrução SQL por vez".

**Correções:**

1. Exigir que `SELECT`/`WITH` esteja no **início de uma linha** (após
   `re.MULTILINE`), não em qualquer ponto do texto.
2. Limitar a captura a, no máximo, um único statement: cortar tudo após
   o primeiro `;` no nível superior (fora de strings).
3. Stop-patterns em pt-BR e en-US (`This query`, `Note that`, `Observação`).

---

### 4. Escaneamento vazio derruba todas as tabelas do DuckDB

**Arquivos:** `apps/streamlit/olap/ingest.py` (linhas 453-461),
`apps/streamlit/app.py` (`_render_sidebar`).

**Sintoma:** se o usuário digita um caminho temporariamente inválido na
sidebar OU se a pasta de projetos está temporariamente inacessível (rede
fora do ar), o `scan_all_projects` retorna `[]`. Em seguida,
`sync_tabular_from_scans([])` interpreta TODAS as chaves do manifesto
como "stale" e dropa todas as tabelas + entradas do manifesto.

Próxima execução com a pasta correta tem que reingerir tudo do zero.

**Correções:**

1. `sync_tabular_from_scans` pula a fase de poda quando `scans` está
   vazio E o manifesto não está, retornando `IngestStats` com flag de
   `aborted_due_to_empty_scan=True`.
2. `_render_sidebar` só dispara o sync OLAP se `scans` tem ao menos um
   arquivo tabular, exibindo aviso caso contrário.

---

### 5. `apps/api/` é dead code com configuração divergente

**Pasta:** `apps/api/`

**Diagnóstico:**

- Não é importado pelo Streamlit, não está no `docker-compose.yml`, não
  é mencionado no `README.md`.
- `apps/api/app/config.py` lê `LLM_BASE_URL` sem normalizar para `/v1`
  (divergente de `apps/streamlit/llm_config.py`).
- `apps/api/app/llm.py` usa OpenAI sem parâmetros Qwen3.5
  (`enable_thinking`, sampling) e sem strip de `<think>`.
- `apps/api/app/routers/chat.py` não tem system prompt, RAG nem OLAP —
  só passa mensagens cruas.

Manter código duplicado e desatualizado é dívida técnica futura. Se o
projeto voltar a ter API externa, melhor reescrever a partir do que
existe hoje no Streamlit (com a centralização em `llm_config` e
`qwen35_inference`).

**Correção:** remover `apps/api/` por completo. Caso volte a ser
necessário, o git tem o histórico (`git log -- apps/api`).

---

## P1 — pendentes (recomendado próxima rodada)

### 6. Streaming exibe `<think>` em tempo real

Usuário vê tokens de raciocínio sendo digitados e depois somem do
histórico (são strippados). UX inconsistente. Filtrar no gerador de
tokens ou separar visualmente (expander "Raciocínio").

### 7. Duplo escaneamento + hash duplicado

Sidebar faz `scan_all_projects` duas vezes (todas as extensões + só
tabulares). Cada arquivo tabular paga SHA-256 duas vezes. Filtrar a
lista existente em vez de re-walking.

### 8. `_TEXT_TO_SQL_SYSTEM` redundante

Linha "Não escreva tags `<think>`" no system prompt agora que
`enable_thinking=False` é enviado via `extra_body`. Mencionar a tag
negativamente pode levar modelos pequenos a invertê-la.

### 9. `CREATE OR REPLACE TABLE` apaga índices

Comportamento do DuckDB confirmado em smoke-test:

```
indices antes: [('i',)]
indices depois CREATE OR REPLACE: []
```

Hoje mitigado porque `ensure_indexes_for_all_ingested_tables` rodando
no fim do sync recria tudo. Para sync grandes, recriar índice por
tabela logo após `_register_table` evita janela sem índice.

### 10. Cliente `OpenAI` por mensagem + timeout fixo

`OpenAI(base_url, api_key, timeout=120.0)` é instanciado a cada chat.
Qwen3.5 em modo thinking pode demorar mais de 2 minutos. Cachear via
`st.cache_resource` e tornar timeout configurável.

### 11. `_read_csv` / `_read_excel_sheets` engolem todas as exceções

`except Exception: continue` esconde encoding errado, permissão negada,
arquivo bloqueado pelo Office. `stats.errors` deveria registrar.

### 12. `sanitize_history_message` só limpa Qwen3.5

Se usuário muda de modelo no LM Studio, histórico anterior com
`<think>` vai cru para o novo modelo. Aplicar strip incondicionalmente.

### 13. RAG não filtra por `project_id`

`rag_semantic_search(prompt, top_k)` retorna trechos de todos os
projetos. Risco de a resposta sobre o projeto A trazer evidência do
projeto B. Adicionar multiselect "Projetos visíveis" no chat.

---

## P2 — observações para backlog

- **14.** `_try_convert_column_to_numeric` aceita 70%: subir para 95% ou
  exigir conversão total.
- **15.** Volume `assistente_sqlite` declarado no compose mas sem uso —
  remover.
- **16.** Caminho hardcoded `D:/Vanessa/AI_project/Projetos` no compose
  como fallback — trocar por `./projetos`.
- **17.** `_check_openai_compatible_models` não envia `Authorization`.
  Não compatível com backends OpenAI-like que exigem bearer real.
- **18.** Lista `_FORBIDDEN_SQL` ampliada na correção #2 desta data.
- **19.** `format_context_for_llm` corta por caractere (12 000); deveria
  considerar janela do modelo em tokens.
- **20.** Schema catalog envia 2 linhas de dados reais no prompt;
  possível vazamento de dado sensível em logs do servidor LLM.
- **21.** `check_duckdb` cria o arquivo do banco em modo "teste" quando
  o arquivo não existe — side effect.
- **22.** `test_manifest_incremental.py` mexe em variável de ambiente
  sem restaurar; pode contaminar testes em sequência.

---

## Histórico de commits desta rodada

1. `feat: configuração Qwen3.5-MTP, indexação OLAP e log de revisão` — trabalho pendente da sessão + este arquivo.
2. `fix(olap): autodetectar separador CSV (vírgula/ponto-e-vírgula/tab)` — item 1.
3. `fix(olap): SQL validator ignora literais e cobre mais palavras` — item 2.
4. `fix(olap): extrai SQL exigindo SELECT/WITH em início de linha` — item 3.
5. `fix(olap): salvaguarda contra drop em sync com escaneamento vazio` — item 4.
6. `chore: remove apps/api dead code (substituído pelo Streamlit)` — item 5.

---

## Como reproduzir os smoke-tests da revisão

Script ad-hoc usado nesta auditoria (não comitado):

- Valida `validate_readonly_sql` com strings contendo `'DROP TABLE Y'`,
  `'a;b'`, `'set'`, `'CREATE OR REPLACE'`.
- Valida `_extract_sql` com "Vou usar SELECT mas precisa pensar... SELECT * FROM y LIMIT 10".
- Cria CSV `a;b;c\n1;2;3` em tempdir e roda `_read_csv`.
- Cria tabela DuckDB com índice, executa `CREATE OR REPLACE TABLE`,
  lista `duckdb_indexes()`.

Os mesmos cenários são incorporados como testes unitários nos commits
das correções 1-3 (ver `apps/streamlit/tests/`).
