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
│   ├── chat_router.py          # Roteador: documentos vs planilhas por mensagem
│   ├── projects_loader.py      # Inventário: um subdiretório de 1º nível = projeto
│   ├── llm_config.py           # Defaults do LLM (OpenRouter) + resolver de chave
│   ├── qwen35_inference.py     # Perfis de sampling (Qwen3.5 ↔ outros) + strip de thinking
│   ├── olap/                   # DuckDB: ingestão, NL→SQL, catálogo
│   ├── ml/                     # ML tradicional: FLAML, catálogo de colunas, .pkl
│   ├── rag/                    # Extração, chunking, índice txtai (upsert em lotes)
│   ├── agents/                 # Sistema multiagentes (CrewAI local)
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

1. **Desenvolvimento → Diagnóstico** — confirme que a chave da API aparece como **configurada** e clique em **Testar GET /v1/models** (o OpenRouter deve responder).
2. **Documentos** — informe a raiz (ou use a do `.env`) e **Escanear pastas**.
3. **Documentos** — **Atualizar base agora** (primeira vez ou após mudanças nos arquivos).
4. **Desenvolvimento → Busca semântica** — valide recuperação de trechos.
5. **Conversa** — envie uma mensagem; documentos e planilhas entram automaticamente quando disponíveis.
6. **ML tradicional** — carregue **AbRank (Kaggle)**, revise o dicionário, treine FLAML (`log_Aff`), salve `.pkl` e teste predição em lote novo.

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
| `LLM_BASE_URL` | Contêiner | Endpoint OpenAI-compatível. Padrão: `https://openrouter.ai/api/v1`. |
| `LLM_MODEL` | Contêiner | Slug do modelo no catálogo OpenRouter. |
| `OPENROUTER_API_KEY` | Contêiner | Chave de API real do OpenRouter. |
| `OPENROUTER_APP_TITLE` | Contêiner | Título exibido nos rankings (opcional). |
| `OPENROUTER_HTTP_REFERER` | Contêiner | URL exibida nos rankings (opcional). |

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

## Sistema multiagentes (CrewAI)

Pipeline opcional para orquestrar o chat em agentes especializados. Ativado por `USE_CREWAI=1` no `.env`. Continua usando o OpenRouter como provedor do LLM.

```
Mensagem → Greeter (rule-based) → Triage Agent → Tools (RAG, OLAP, ML em paralelo) → Synthesizer Agent → Resposta
```

| Agente / Tool | Papel | LLM calls |
|---------------|-------|-----------|
| **Greeter** | Curto-circuita saudações ("oi", "obrigado") | 0 |
| **Triage Agent** | Classifica intenção e decide rotas (JSON) | 1 |
| **RAG Tool** | Busca semântica no índice txtai | 0 (só embedding) |
| **DuckDB Tool** | NL → SQL read-only no DuckDB | 1 (gera SQL) |
| **ML Tool** | Extrai features e roda predição AbRank | 1 (extrai features) |
| **Synthesizer Agent** | Resposta final cordial + citações | 1 (com streaming) |

Detalhes completos: [`apps/streamlit/agents/AGENTS.md`](apps/streamlit/agents/AGENTS.md).

### Modo aprendizado (trilha visível)

Ative o toggle **"Mostrar trilha do crew"** em **Desenvolvimento → Parâmetros do chat** para ver, em um expander abaixo da resposta, cada etapa executada com:

- duração em ms,
- entrada/saída resumida,
- metadados (rota escolhida, evidências, SQL gerado, predições).

Ideal para inspecionar o handoff entre agentes durante a curva de aprendizado.

### Comparativo de chamadas LLM (atual × Crew)

| Cenário | Roteador legado | Crew |
|---------|-----------------|------|
| Saudação | 0 | 0 |
| Pergunta só docs | 2 (router + chat) | 2 (triage + synth) |
| Pergunta só planilhas | 3 (router + SQL + chat) | 3 (triage + SQL + synth) |
| Pergunta só ML | 3 (router + extract + chat) | 3 (triage + extract + synth) |
| Pergunta combinada | 3–4 | 3–4 |

Ou seja, o Crew **não aumenta o custo em tokens** — o ganho é em rastreabilidade, manutenção e paralelismo das Tools.

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
| Chunking | Indexação RAG | `rag/chunking.py` — padrão ~520 caracteres, sobreposição 80 |
| Índice | Indexação RAG | `rag/index_txtai.py` — modelo `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` |
| Busca | Teste RAG / Chat | Similaridade semântica; trechos citam projeto e arquivo |

**Primeira indexação:** na aba **Indexação RAG**, use **Substituir índice existente** (recomendado). A primeira execução pode demorar vários minutos (download do modelo de embeddings).

**Reindexação incremental:** desmarque *Substituir índice existente* para processar só arquivos **novos, alterados ou removidos** (comparação por SHA-256 e manifesto em `/data/txtai/index_manifest.json`). Arquivos inalterados são pulados. Na primeira vez após atualizar o app, use *Substituir* uma vez ou aceite um ciclo incremental para alinhar o manifesto.

**Chat com RAG:** na aba **Chat**, ative **Usar RAG (txtai)** depois que o índice estiver pronto (aba Diagnóstico mostra *Índice pronto: sim*).

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
| **HTTP 429 (rate limit)** | Limite de requisições do plano gratuito atingido | Aguarde alguns minutos; reduza o número de chamadas paralelas (`CREW_PARALLEL_TOOLS=0`) |
| Chat sem resposta / timeout | Conexão do contêiner com a internet | Teste `docker exec assistente-lab-streamlit curl -sI https://openrouter.ai/api/v1/models`; aumente `LLM_TIMEOUT_S` |
| Inventário vazio | Sem subpastas na raiz ou extensões filtradas | Ajuste filtro na barra lateral; confira extensões dos arquivos |
| **RAG não retorna documentos esperados** | Índice não construído ou desatualizado | Escaneie → **Indexação RAG** → *Substituir índice* → reconstrua |
| Busca só mostra planilhas grandes | Projeto com muitos chunks de xlsx domina o top‑K | Normal em MVP; refine a pergunta ou use **Teste RAG** com mais trechos |
| `unhealthy` no Docker | Streamlit não subiu | `docker compose logs streamlit` |
| Permissão negada em arquivos | Bind somente leitura | Usuário da imagem (`uid 10001`) precisa ler os arquivos no host |

---

## Rastreio código ↔ produto

| Necessidade | Começar em |
|-------------|------------|
| UI, inventário, abas | `apps/streamlit/app.py` |
| Chat + RAG no prompt | `apps/streamlit/app.py` (`rag_semantic_search`, `format_context_for_llm`) |
| Configuração do LLM (defaults, chave) | `apps/streamlit/llm_config.py` |
| Extração, chunking, índice, busca | `apps/streamlit/rag/` |
| Scan de projetos | `apps/streamlit/projects_loader.py` |
| Imagem e healthcheck | `docker/streamlit/Dockerfile` |
| Portas, volumes, env | `docker-compose.yml` |

Próximas entregas (DuckDB na UI, autenticação, etc.) estão no **playbook**.
