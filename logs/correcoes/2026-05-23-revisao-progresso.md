# Revisão de progresso e corner cases — 2026-05-23

**Escopo:** sincronizar playbook/README com o estado real do código; corrigir
itens P1/P2 pendentes da revisão de 2026-05-22.

---

## Progresso atualizado (resumo)

| Área | Antes (playbook 2026-05-18) | Agora |
|------|------------------------------|-------|
| DuckDB / OLAP | Pendente | Entregue (`olap/`, NL→SQL, ingestão no scan) |
| Chat + RAG | Em validação | Entregue (roteador, Qwen3.5, streaming) |
| UI | 7 abas planas | 3 abas principais + Desenvolvimento |
| Testes | Não registrados | 99 testes unitários (`apps/streamlit/tests/`) |
| Fase 1 | ~65–70% | ~80% |
| Fase 2 | ~70% | ~85% |
| Fase 3 | ~75–80% | ~85% |

Detalhes completos: `.cursor/plans/20260429playbook_mvp_biotech.plan.md`.

---

## Corner cases corrigidos nesta rodada

| # | Item | Arquivo(s) | Status |
|---|------|------------|--------|
| 6 | Streaming exibia `<think>` antes do strip | `qwen35_inference.py`, `app.py` | Corrigido (`iter_stream_answer_text`) |
| 7 | Duplo escaneamento + hash duplicado para OLAP | `projects_loader.py`, `app.py` | Corrigido (`filter_scans_by_extensions`) |
| 12 | `sanitize_history_message` só limpava Qwen3.5 | `qwen35_inference.py` | Corrigido (strip em todo assistant) |
| 13 | RAG não filtrava por `project_id` | `rag/index_txtai.py`, `app.py` | Parcial (filtro Documentos + dev) |
| 17 | Diagnóstico LLM sem `Authorization` | `app.py` | Corrigido |
| 20 | Schema catalog enviava linhas reais ao LLM | `schema_catalog.py`, `nl_query.py` | Corrigido (`sample_rows=0` no prompt) |
| 21 | `check_duckdb` criava arquivo ao testar | `olap/connection.py` | Corrigido (`:memory:` se sem arquivo) |
| 22 | Teste não restaurava `ASSISTENTE_TXTAI_DIR` | `test_manifest_incremental.py` | Corrigido |
| — | Cliente OpenAI recriado a cada mensagem | `app.py` | Corrigido (`st.cache_resource`) |
| — | Fallback hardcoded no Compose | `docker-compose.yml` | Corrigido (`./projetos`) |

---

## Ainda pendente (backlog)

- Guardrail explícito “sem fonte → não inventar” (além do system prompt).
- Suíte curada >90% citações válidas.
- Autenticação web + SQLite/auditoria.
- Truncagem de contexto RAG por tokens (hoje ~12k caracteres).
- `LLM_TIMEOUT_S` documentado no README (variável já lida em `app.py`).
- Erros de leitura CSV/XLSX registrados em `stats.errors` (item 11 da revisão anterior).

---

## Testes

```powershell
cd apps/streamlit
python -m unittest discover -s tests -p "test_*.py" -v
```

Esperado: todos OK (107 testes após ML/AbRank e revisão geral).

---

## Revisão geral + limpeza (mesmo dia, segunda rodada)

| # | Item | Severidade | Status |
|---|------|------------|--------|
| 23 | `kaggle_sources` engolia erros com `except Exception` | Alta | Corrigido (auth/rede propagam) |
| 24 | `_load_catalog_dataset` chamava `_catalog_picker()` de novo (widget duplicado) | Média | Corrigido |
| 25 | `prepare_feature_matrix` não descartava `log_Aff` NaN em regressão | Média | Corrigido |
| 26 | Features 100% NaN (ex. `escape` em amostra) no treino | Baixa | Excluídas em `default_feature_columns` |
| 27 | `_msg.txt`, imports mortos, `load_bundle_from_path` | Baixa | Removidos |
| 28 | README/playbook desatualizados (OLAP, demo ML) | Média | Atualizados |
| 29 | Diagnóstico sem volume `/data/ml` / status FLAML | Baixa | Adicionado em `app.py` |
