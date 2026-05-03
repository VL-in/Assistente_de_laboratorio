# Assistente de lab

MVP offline-first para análise documental com RAG local (ver playbook em `.cursor/plans/` e blueprint em `Histórico/`).

## LM Studio (IA local)

O modelo de linguagem roda no [LM Studio](https://lmstudio.ai/), que expõe uma API compatível com OpenAI no seu PC.

1. Instale o LM Studio e baixe um modelo de chat no catálogo do aplicativo.
2. Abra a aba do **servidor local** e clique para **iniciar o servidor**. Anote a **porta** (padrão comum: `1234`) e o **nome do modelo** exibido na interface — esse nome é o valor de `LLM_MODEL`.
3. Copie `apps/api/.env.example` para `apps/api/.env` e ajuste `LLM_BASE_URL` (se mudou a porta) e `LLM_MODEL` (nome exato do modelo).
4. `LLM_API_KEY` pode ficar como no exemplo; o LM Studio em geral não valida a chave, mas o cliente HTTP precisa de algum valor.

Fluxo recomendado: **interface ou app desktop → API FastAPI → LM Studio**. Evita chamar o LM Studio direto do navegador (CORS e segurança).

## API FastAPI (`apps/api`)

```bash
cd apps/api
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# Edite .env: defina LLM_MODEL com o id do modelo no LM Studio
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- `GET /health` — API no ar.
- `GET /health/llm` — tenta falar com o LM Studio (use para saber se o servidor local está ligado).
- `POST /chat` — corpo JSON: `{"messages":[{"role":"user","content":"Olá"}]}` — resposta: `{"message":"..."}`.
- `POST /chat/stream` — mesmo corpo, resposta em SSE (`text/event-stream`).

Documentação interativa: `http://127.0.0.1:8000/docs` (com o servidor uvicorn rodando).
