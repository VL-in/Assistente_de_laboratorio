# Assistente de lab

Aplicação web **MVP offline-first** para apoiar P&D com documentos locais, **RAG** (txtai + embeddings multilíngues), **OLAP** (DuckDB), **ML tradicional** (FLAML + Kaggle AbRank) e geração via **OpenRouter** (API compatível com OpenAI). A UI e o pipeline Python rodam **no Docker**; dados sensíveis ficam na sua máquina e o LLM é consultado por API remota — o que tira a carga de hardware do PC local.

| Documento | Função |
|-----------|--------|
| [`.cursor/plans/20260429playbook_mvp_biotech.plan.md`](.cursor/plans/20260429playbook_mvp_biotech.plan.md) | Playbook: fases, escopo, riscos e **tabela de progresso** |
| [`.env.docker.example`](.env.docker.example) | Modelo de `.env` para Compose (copiar para `.env` na raiz) |

---

## Visão rápida do repositório

```
Assistente_de_lab/
├── docker-compose.yml          # Orquestração: Streamlit + volumes
├── docker/streamlit/Dockerfile # Imagem da app (usuário não-root, HEALTHCHECK HTTP)
├── .env.docker.example         # Modelo de `.env` (segredos ficam só no `.env`)
├── apps/streamlit/
│   ├── app.py                  # UI: Conversa, Documentos, ML tradicional, Desenvolvimento
│   ├── projects_loader.py      # Inventário: um subdiretório de 1º nível = projeto
│   ├── llm_config.py           # Defaults do LLM (OpenRouter) + resolver de chave
│   ├── observability/          # Langfuse: traces e sessões do chat
│   ├── qwen35_inference.py     # Perfis de sampling (Qwen3.5 ↔ outros) + strip de thinking
│   ├── agents/                 # Chat multiagente (Triage → Tools → Synthesizer)
│   │   └── AGENTS.md           # Arquitetura do crew (documentação técnica)
│   ├── olap/                   # DuckDB: ingestão, NL→SQL, catálogo
│   ├── ml/                     # ML tradicional: FLAML, chat_infer, .pkl
│   ├── rag/                    # Extração, chunking, índice txtai (upsert em lotes)
│   ├── tests/                  # ~155 testes unitários
│   └── requirements.txt
```

**Convenção de projeto:** na pasta configurada como raiz (ex.: `Projetos`), cada **subdiretório imediato** é um **projeto** (`project_id` = nome da pasta). Pastas como `planning/` ou `results/` pertencem ao mesmo projeto.

---

## Guia rápido (primeira execução)

Siga esta ordem para subir a aplicação sem erros de chave de API, rede ou RAG.

### 1. Pré-requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) ou Docker Engine + Compose v2 (Linux)
- Conta no [OpenRouter](https://openrouter.ai/) com **chave de API** ativa (https://openrouter.ai/settings/keys)
- Uma pasta de projetos no host, com **um subdiretório por projeto** e documentos (`.docx`, `.xlsx`, `.pdf`, `.txt`, `.md`, `.csv`)

> **Por que OpenRouter?** Ele expõe centenas de modelos (incluindo os **gratuitos** via `openrouter/free`) através de uma única API compatível com OpenAI. Isso elimina a necessidade de hospedar o LLM localmente e libera o hardware da sua máquina. O custo dos modelos pagos aparece no seu painel da conta; os modelos `:free` são gratuitos com limites de requisição.

### 2. Configurar `.env`

Na **raiz** do repositório:

```bash
cp .env.docker.example .env
```

Edite o `.env` (nunca commite este arquivo):

| Variável | Obrigatório | O que preencher |
|----------|-------------|-----------------|
| `PROJETOS_HOST_DIR` | Sim | Caminho **absoluto** no host da pasta que contém os projetos (ex.: `D:/Vanessa/AI_project/Projetos`) |
| `OPENROUTER_API_KEY` | **Sim** | Chave gerada em https://openrouter.ai/settings/keys (começa com `sk-or-v1-...`) |
| `LLM_MODEL` | Não | Slug do modelo (padrão `openrouter/auto`). Veja [modelos](#escolha-do-modelo) |
| `LLM_BASE_URL` | Não | Padrão `https://openrouter.ai/api/v1` |
| `OPENROUTER_APP_TITLE` | Não | Nome que aparece nos rankings do OpenRouter |
| `STREAMLIT_PORT` | Não | Padrão `8502` |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Não | Observabilidade LLM (Langfuse); ver [seção abaixo](#observabilidade-langfuse) |

### 3. Subir a aplicação

```bash
docker compose up --build
```

Para rodar em segundo plano:

```bash
docker compose up -d --build
```

Abra no navegador: **http://127.0.0.1:8502**

Após alterar o `.env`, recrie o contêiner para aplicar as variáveis:

```bash
docker compose up -d --build
```

### 4. Validar na interface

1. **Desenvolvimento → Diagnóstico** — confirme que a chave da API aparece como **configurada**, teste **GET /v1/models** e (opcional) verifique **Observabilidade (Langfuse)**.
2. **Documentos** — informe a raiz (ou use a do `.env`) e **Escanear pastas**.
3. **Documentos** — **Atualizar base agora** (primeira vez ou após mudanças nos arquivos).
4. **Desenvolvimento → Busca híbrida** — valide recuperação de trechos (BM25 + semântica).
5. **Conversa** — envie uma mensagem; o crew roteia automaticamente para documentos, planilhas e/ou ML quando disponíveis.
6. **ML tradicional** — carregue **AbRank (Kaggle)**, revise o dicionário, treine FLAML (`log_Aff`), salve `.pkl` e teste predição em lote ou via chat (com `.pkl` em `ASSISTENTE_ML_CHAT_MODEL`).

### ML tradicional (aba dedicada)

Dataset padrão: **[AbRank no Kaggle](https://www.kaggle.com/datasets/aurlienplissier/abrank)** (`aurlienplissier/abrank`), split `Benchmarks/train_regression.csv`, alvo **`log_Aff`** (regressão de afinidade anticorpo–antígeno).

| Variável | Descrição |
|----------|-----------|
| `ASSISTENTE_ML_DIR` | Onde salvar modelos `.pkl` (Docker: `/data/ml`) |
| `ASSISTENTE_ML_CHAT_MODEL` | Caminho do `.pkl` usado no chat (padrão: `/data/ml/modelo_20260524_224734_04768.pkl`) |
| `KAGGLE_API_TOKEN` | Token da API Kaggle (obrigatório fora do Kaggle Notebooks) |
| `KAGGLEHUB_CACHE` | Cache dos downloads (Docker: `/data/ml/kagglehub`) |

Catálogo YAML em `apps/streamlit/ml/catalogs/abrank_kaggle.yaml`.

---

## Executar com Docker (detalhes)

O serviço usa `restart: unless-stopped` e `PYTHONUNBUFFERED=1` para logs mais imediatos.

### Variáveis de ambiente (Compose / `.env`)

| Variável | Onde | Descrição |
|----------|------|-----------|
| `PROJETOS_HOST_DIR` | Host | Pasta montada em `/data/projetos` no contêiner (somente leitura). |
| `STREAMLIT_PORT` | Host | Porta publicada (padrão **8502**). Mapeamento: `HOST:8502→8502`. |
| `ASSISTENTE_PROJETOS_DIR` | Contêiner | Definido no Compose como `/data/projetos` (não precisa alterar). |
| `ASSISTENTE_TXTAI_DIR` | Contêiner | Volume persistente do índice txtai (`/data/txtai`). |
| `EMBEDDING_SERVICE_URL` | Contêiner | URL do TEI de embeddings (padrão `http://embeddings:80`). |
| `EMBEDDING_TIMEOUT_S` | Contêiner | Timeout HTTP para vetorização (padrão 120 s). |
| `LLM_BASE_URL` | Contêiner | Endpoint OpenAI-compatível. Padrão: `https://openrouter.ai/api/v1`. |
| `LLM_MODEL` | Contêiner | Slug do modelo no catálogo OpenRouter. |
| `OPENROUTER_API_KEY` | Contêiner | Chave de API real do OpenRouter. |
| `OPENROUTER_APP_TITLE` | Contêiner | Título exibido nos rankings (opcional). |
| `OPENROUTER_HTTP_REFERER` | Contêiner | URL exibida nos rankings (opcional). |
| `LANGFUSE_PUBLIC_KEY` | Contêiner | Chave pública do projeto Langfuse (observabilidade LLM). |
| `LANGFUSE_SECRET_KEY` | Contêiner | Chave secreta do projeto Langfuse. |
| `LANGFUSE_BASE_URL` | Contêiner | API Langfuse (padrão `https://cloud.langfuse.com`; EUA: `https://us.cloud.langfuse.com`). |

### Observabilidade (Langfuse)

**Skill do Cursor (opcional):** para o agente seguir as práticas oficiais ao evoluir traces:

```bash
npx skills add langfuse/skills --skill "langfuse"
```

Ou copie de [github.com/langfuse/skills](https://github.com/langfuse/skills) para `~/.cursor/skills/langfuse`.

O [Langfuse](https://langfuse.com/docs) registra cada chamada ao LLM (Triage, SQL NL→SQL, Synthesizer, etc.): prompt, modelo, tokens, latência e erros. A integração usa o [SDK Python com wrapper OpenAI](https://langfuse.com/integrations/model-providers/openai-py) — o código do app continua com `openai.OpenAI`; o patch é aplicado automaticamente quando as chaves estão no `.env`.

**Como ativar:**

1. Crie conta em https://cloud.langfuse.com (ou self-host).
2. Em **Settings → API Keys**, gere `LANGFUSE_PUBLIC_KEY` e `LANGFUSE_SECRET_KEY`.
3. Adicione ao `.env` (veja também `.env.docker.example`).
4. Recrie o contêiner: `docker compose up -d --build`.

Cada sessão do navegador no Streamlit vira um **session** no Langfuse (várias mensagens agrupadas). O status aparece em **Desenvolvimento → Diagnóstico → Observabilidade (Langfuse)**.

**Hierarquia de traces (best practices):**

| Nome no Langfuse | O que é |
|------------------|---------|
| `chat-turn` | Um turno completo (mensagem do usuário → resposta) |
| `crew-pipeline` | Orquestração multiagente |
| `crew-triage` / `crew-synthesizer` | Chamadas LLM nomeadas |
| `olap-nl-to-sql` / `ml-feature-extract` | LLM interno das Tools |
| `RAG Tool` / `OLAP Tool` / `ML Tool` | Spans tipo *tool* (sem LLM no RAG) |

Tags automáticas: `feature:chat`, `route:rag`, `route:olap`, `route:ml` conforme a rota escolhida.

| Variável | Efeito |
|----------|--------|
| `LANGFUSE_ENABLED=0` | Desliga o envio de traces sem remover as chaves. |
| `LANGFUSE_TRACING_ENVIRONMENT` | Rótulo do ambiente (`development`, `production`, …). |
| `LANGFUSE_RELEASE` | Versão do deploy (útil para comparar releases). |
| `LANGFUSE_TAGS` | Tags separadas por vírgula (ex.: `streamlit,openrouter`). |

### OpenRouter

O OpenRouter é uma API agregadora compatível com OpenAI: você usa **uma chave** e tem acesso a centenas de modelos diferentes (OpenAI, Anthropic, Meta, Mistral, Google, Qwen, etc.). O endpoint padrão é `https://openrouter.ai/api/v1`.

**Como obter a chave:**

1. Crie conta gratuita em https://openrouter.ai.
2. Vá em **Settings → Keys** (https://openrouter.ai/settings/keys).
3. Clique em **Create Key**, copie o valor e cole em `OPENROUTER_API_KEY` no `.env`.

**Testar a chave de dentro do contêiner:**

```bash
docker exec assistente-lab-streamlit python -c "
import json, os, urllib.request
key = os.environ['OPENROUTER_API_KEY']
req = urllib.request.Request(
    'https://openrouter.ai/api/v1/models',
    headers={'Authorization': f'Bearer {key}'},
)
with urllib.request.urlopen(req, timeout=10) as r:
    data = json.loads(r.read().decode())
    print('Modelos disponíveis:', len(data.get('data', [])))
"
```

Se a chamada listar modelos, o chat na UI deve funcionar. Se aparecer **HTTP 401**, a chave está errada ou ausente; em caso de **timeout**, verifique a conexão do contêiner com a internet.

### Escolha do modelo

O `LLM_MODEL` aceita qualquer slug listado em https://openrouter.ai/models. Para começar, recomendamos:

| Slug | Quando usar | Custo |
|------|-------------|-------|
| `openrouter/auto` | Padrão — o roteador escolhe o melhor modelo para cada requisição. | Mesmo do modelo selecionado |
| `openrouter/free` | Sorteia entre modelos **gratuitos** disponíveis. | **Grátis** (limites por minuto/dia) |
| `meta-llama/llama-3.3-70b-instruct:free` | Modelo grande e multilíngue, bom para PT-BR. | **Grátis** |
| `qwen/qwen3.5-7b-instruct:free` | Qwen3.5 — a app ainda usa os perfis Qwen quando esse slug é detectado. | **Grátis** |
| `anthropic/claude-3.5-sonnet` | Alta qualidade para síntese complexa. | Pago |

> Os perfis de sampling (`PROFILE_CHAT_INSTRUCT`, `PROFILE_OLAP_SQL` etc.) em `apps/streamlit/qwen35_inference.py` continuam aplicando os parâmetros validados do Qwen3.5 apenas quando o slug contém `qwen3.5`. Outros modelos recebem um perfil neutro (sem `extra_body.chat_template_kwargs`).

Variáveis opcionais no `.env`:

| Variável | Efeito |
|----------|--------|
| `LLM_ENABLE_THINKING=1` | Liga o toggle **Modo raciocínio** por padrão (só faz efeito com Qwen3.5). |
| `LLM_TIMEOUT_S=120` | Timeout das chamadas ao OpenRouter (segundos). Aumente em redes lentas. |
| `CHAT_MAX_TOKENS=2048` | Limite de tokens de saída no chat. |

### Volumes dentro do contêiner

| Caminho | Uso |
|---------|-----|
| `/data/projetos` | Bind do host: documentos dos projetos (**somente leitura**) |
| `/data/txtai` | Volume nomeado: índice vetorial RAG (persiste entre reinícios) |
| `/data` (contêiner `embeddings`) | Volume nomeado: cache do modelo TEI (`intfloat/multilingual-e5-small`) |
| `/data/duckdb` | Volume nomeado: OLAP |
| `/data/ml` | Volume nomeado: modelos ML (`.pkl`) |
| `/data/sqlite` | Volume nomeado: metadados (fases futuras) |

### Saúde do contêiner (HEALTHCHECK)

A imagem verifica **`http://127.0.0.1:8502/_stcore/health`** (endpoint interno do Streamlit).

```bash
docker inspect --format='{{json .State.Health}}' assistente-lab-streamlit
```

No Windows (PowerShell), use `curl.exe` — o alias `curl` do PowerShell não é o mesmo comando:

```powershell
curl.exe -s -o NUL -w "HTTP %{http_code}`n" http://127.0.0.1:8502/_stcore/health
```

Esperado: **HTTP 200**. Logs: `docker compose logs streamlit --tail 50`.

---

## Sistema multiagentes (chat)

O **único caminho** da aba **Conversa** — orquestração custom (não usa `crewai.Crew.kickoff` em runtime). O Triage decide quais Tools acionar; o Synthesizer gera a resposta final com streaming. Provedor LLM: **OpenRouter**.

```
Mensagem → Greeter (rule-based) → Triage → Tools (RAG, OLAP, ML em paralelo) → Synthesizer → Resposta
```

| Agente / Tool | Papel | Chamadas LLM |
|---------------|-------|--------------|
| **Greeter** | Curto-circuita saudações ("oi", "obrigado") | 0 |
| **Triage** | Classifica intenção e decide rotas (JSON) | 1 |
| **RAG Tool** | Busca híbrida no índice txtai (BM25 + E5) | 0 (só embedding) |
| **OLAP Tool** | NL → SQL read-only no DuckDB | 1 (gera SQL) |
| **ML Tool** | Extrai features e roda predição AbRank | 1 (extrai features) |
| **Synthesizer** | Resposta final cordial + citações | 1 (com streaming) |

Detalhes completos: [`apps/streamlit/agents/AGENTS.md`](apps/streamlit/agents/AGENTS.md).

### Modo aprendizado (trilha visível)

Ative o toggle **"Mostrar trilha do crew"** em **Desenvolvimento → Parâmetros do chat** para ver, em um expander abaixo da resposta, cada etapa executada com:

- duração em ms,
- entrada/saída resumida,
- metadados (rota escolhida, evidências, SQL gerado, predições).

Ideal para inspecionar o handoff entre agentes durante a curva de aprendizado. Com **Langfuse** ativo, a mesma execução também aparece no painel web (traces + sessions).

### Chamadas LLM por cenário

| Cenário | Chamadas |
|---------|----------|
| Saudação | 0 |
| Pergunta só docs | 2 (triage + synth) |
| Pergunta só planilhas | 3 (triage + SQL + synth) |
| Pergunta só ML | 3 (triage + extract + synth) |
| Pergunta combinada | 3–4 |

Variáveis úteis: `CREW_TRACE_HANDOFF=1`, `CREW_PARALLEL_TOOLS=1` (ver `.env.docker.example` e `AGENTS.md`). A flag `USE_CREWAI` foi removida — não é mais necessária.

---

## Pipeline RAG (txtai)

Fluxo implementado na UI:

```
Pasta de projetos → inventário (scan) → extração de texto → chunking → índice txtai → busca / chat com contexto
```

| Etapa | Onde na UI | Detalhe técnico |
|-------|------------|-----------------|
| Inventário | Fontes e inventário | `projects_loader.py` — extensões filtráveis na barra lateral |
| Extração | Indexação RAG | `rag/extract.py` — docx (**parágrafos + tabelas**), xlsx, pdf, txt, md, csv |
| Chunking | Indexação RAG | `rag/chunking.py` — padrão **520** caracteres (~100–130 tokens), sobreposição **120**, lote 64 |
| Índice | Indexação RAG | `rag/index_txtai.py` + TEI — modelo [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) (até 512 tokens) + **BM25** (busca híbrida) |
| Busca | Teste RAG / Chat | Híbrida: semântica (E5) + lexical (BM25); trechos citam projeto e arquivo |

**Busca híbrida:** por padrão o txtai mantém dois índices em paralelo — vetorial (significado) e BM25 (termos exatos). Isso ajuda a recuperar nomes compostos de laboratório (ex.: *tampão de amostra*) que a busca só semântica costuma diluir em palavras isoladas. Peso denso α padrão: `0.4` (`RAG_HYBRID_WEIGHT`). Desligue com `RAG_HYBRID_ENABLED=0`.

**Primeira indexação / migração:** na aba **Indexação RAG**, use **Substituir índice existente** (recomendado). Índices criados antes da busca híbrida **não** têm BM25 — é obrigatório reconstruir uma vez. Na primeira subida do Compose, o contêiner `embeddings` baixa o modelo TEI (~470 MB); aguarde o healthcheck ficar saudável antes de indexar.

**Reindexação incremental:** desmarque *Substituir índice existente* para processar só arquivos **novos, alterados ou removidos** (comparação por SHA-256 e manifesto em `/data/txtai/index_manifest.json`). Arquivos inalterados são pulados. Documentos **novos** entram automaticamente nos índices semântico **e** BM25 — não há lista fixa de termos técnicos; qualquer palavra presente no texto indexado pode ser recuperada lexicalmente.

**Chat com RAG/OLAP/ML:** na aba **Conversa**, basta enviar a mensagem — o Triage escolhe as rotas quando o índice, as planilhas ou o modelo `.pkl` estão disponíveis. Em **Desenvolvimento → Parâmetros do chat**, o modo *override* permite forçar ou desligar RAG, OLAP e ML manualmente.

---

## Desenvolvimento local (sem Docker)

Útil para depurar `app.py` e `projects_loader.py`:

```powershell
cd apps/streamlit
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
$env:ASSISTENTE_PROJETOS_DIR="D:\caminho\Projetos"
$env:OPENROUTER_API_KEY="sk-or-v1-sua-chave-aqui"
# (opcional) Langfuse local:
# $env:LANGFUSE_PUBLIC_KEY="pk-lf-..."
# $env:LANGFUSE_SECRET_KEY="sk-lf-..."
# (opcional) escolher um modelo específico:
# $env:LLM_MODEL="openrouter/free"
streamlit run app.py --server.port 8502
```

Fora do Docker, se `ASSISTENTE_PROJETOS_DIR` não estiver definida, o loader usa um fallback Windows em `projects_loader.py` (em outros SO, defina a variável).

---

## Problemas comuns

| Sintoma | Causa provável | O que fazer |
|---------|----------------|-------------|
| Pasta vazia ou “Caminho não existe” | `PROJETOS_HOST_DIR` incorreto | Caminho absoluto real no host; um subdiretório por projeto |
| **HTTP 401 no chat** | `OPENROUTER_API_KEY` ausente ou inválida | Verifique o `.env`, recrie a chave em https://openrouter.ai/settings/keys e rode `docker compose up -d --build` |
| **HTTP 402 / saldo insuficiente** | Modelo pago sem créditos na conta | Troque para `openrouter/free` no `LLM_MODEL` ou adicione créditos no painel do OpenRouter |
| **HTTP 429 (rate limit)** | Limite de requisições do plano gratuito atingido | Aguarde alguns minutos; reduza paralelismo (`CREW_PARALLEL_TOOLS=0`) |
| Langfuse inativo no diagnóstico | Chaves ausentes ou `LANGFUSE_ENABLED=0` | Preencha `LANGFUSE_*` no `.env` e recrie o contêiner |
| Chat sem resposta / timeout | Conexão do contêiner com a internet | Teste `docker exec assistente-lab-streamlit curl -sI https://openrouter.ai/api/v1/models`; aumente `LLM_TIMEOUT_S` |
| Inventário vazio | Sem subpastas na raiz ou extensões filtradas | Ajuste filtro na barra lateral; confira extensões dos arquivos |
| **RAG não retorna documentos esperados** | Índice não construído, desatualizado ou só semântico (sem BM25) | Escaneie → **Indexação RAG** → *Substituir índice* → reconstrua |
| Termo composto não aparece (ex.: *tampão de amostra*) | Busca só semântica ou α muito alto | Reconstrua índice híbrido; teste em **Desenvolvimento → Busca**; ajuste `RAG_HYBRID_WEIGHT` (ex.: `0.35`) |
| Busca só mostra planilhas grandes | Projeto com muitos chunks de xlsx domina o top‑K | Normal em MVP; refine a pergunta ou use **Teste RAG** com mais trechos |
| `unhealthy` no Docker | Streamlit não subiu | `docker compose logs streamlit` |
| Permissão negada em arquivos | Bind somente leitura | Usuário da imagem (`uid 10001`) precisa ler os arquivos no host |

---

## Rastreio código ↔ produto

| Necessidade | Começar em |
|-------------|------------|
| UI, inventário, abas | `apps/streamlit/app.py` |
| Chat multiagente | `apps/streamlit/agents/runner.py`, `agents/crew.py` |
| Roteamento de intenção | `apps/streamlit/agents/triage.py`, `agents/intent_rules.py` |
| Configuração do LLM (defaults, chave) | `apps/streamlit/llm_config.py` |
| Observabilidade Langfuse | `apps/streamlit/observability/langfuse_client.py` |
| Extração, chunking, índice, busca | `apps/streamlit/rag/` |
| OLAP / NL→SQL | `apps/streamlit/olap/` |
| ML tradicional + inferência no chat | `apps/streamlit/ml/` |
| Scan de projetos | `apps/streamlit/projects_loader.py` |
| Testes unitários (~155) | `apps/streamlit/tests/` |
| Imagem e healthcheck | `docker/streamlit/Dockerfile` |
| Portas, volumes, env | `docker-compose.yml` |

Próximas entregas (autenticação, auditoria SQLite, guardrails, E2E) estão no [**playbook**](.cursor/plans/20260429playbook_mvp_biotech.plan.md).
