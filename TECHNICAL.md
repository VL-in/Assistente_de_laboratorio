# Documentação Técnica — R&D Lab Manager

Referência técnica para desenvolvedores e mantenedores. Para instalar e usar a aplicação, consulte o [README.md](README.md).

| Documento | Função |
|-----------|--------|
| [`.env.example`](.env.example) | Modelo de variáveis de ambiente — copiar para `.env` na raiz |
| [`apps/streamlit/agents/AGENTS.md`](apps/streamlit/agents/AGENTS.md) | Arquitetura detalhada do sistema multiagentes |

---

## Estrutura do repositório

```
Assistente_de_lab/
├── docker-compose.yml              # Orquestração: Streamlit + serviço de embeddings (TEI)
├── docker/
│   ├── streamlit/Dockerfile        # Imagem da app (usuário não-root, healthcheck HTTP)
│   └── embeddings/Dockerfile       # Serviço TEI (multilingual-e5-small, ~470 MB)
├── .env.example                    # Modelo de variáveis (segredos ficam só no .env)
├── scripts/
│   └── run_evals_docker.ps1        # Script PowerShell para rodar avaliações no Docker
└── apps/streamlit/
    ├── app.py                      # UI: abas Conversa, Documentos, ML, Desenvolvimento
    ├── projects_loader.py          # Inventário de projetos: cada subdiretório = um projeto
    ├── llm_config.py               # Configuração do LLM (OpenRouter, retry, timeout)
    ├── qwen35_inference.py         # Perfis de sampling (Qwen3.5 e outros modelos)
    ├── agents/                     # Sistema multiagentes do chat
    │   ├── AGENTS.md               # Documentação técnica da arquitetura
    │   ├── runner.py               # Orquestrador principal
    │   ├── crew.py                 # Pipeline: Triage + Dispatcher
    │   ├── greeter.py              # Saudações determinísticas (zero LLM)
    │   ├── triage.py               # Classificador de intenção (1 LLM call)
    │   ├── tools.py                # Definições das ferramentas RAG, OLAP e ML
    │   ├── synthesizer.py          # Geração da resposta final (streaming)
    │   ├── handoff.py              # Rastreamento de etapas (HandoffTrace)
    │   ├── intent_rules.py         # Regras determinísticas compartilhadas (regex)
    │   └── security.py             # Guardrails: PII (Presidio) + segredos (detect-secrets)
    ├── rag/                        # Pipeline RAG (extração, chunking, índice, busca)
    ├── olap/                       # DuckDB: ingestão de planilhas, NL→SQL, catálogo
    ├── ml/                         # ML tradicional: FLAML, inferência via chat, .pkl
    ├── observability/              # Langfuse: traces e sessões do chat
    ├── evals/                      # Avaliação end-to-end com DeepEval
    │   ├── run_assistente_eval.py  # CLI principal (18 goldens)
    │   ├── harness.py              # Runtime: EvalRuntime, TurnResult
    │   ├── goldens_projetos_252_253.py  # 18 casos reais (Chikungunya + Dengue)
    │   ├── judge_model.py          # LLM-as-judge (OpenRouter / Anthropic)
    │   ├── eval_bootstrap.py       # Throttle/retry para o tier gratuito
    │   ├── golden_dataset_template.py   # Schema do dataset (ChatGolden, categorias)
    │   ├── datasets/               # Goldens exportados (JSON/JSONL)
    │   └── results/                # Saídas: test_cases_*.json e métricas
    ├── tests/                      # ~195 testes unitários
    ├── requirements.txt            # Dependências da app (inclui CrewAI)
    ├── requirements-base.txt       # Dependências base (sem CrewAI)
    ├── requirements-evals.txt      # Dependências para evals (sem CrewAI — evita conflito posthog)
    └── requirements-security.txt   # Camada de segurança (Presidio, detect-secrets)
```

---

## Sistema multiagentes (chat)

O chat segue um único pipeline — orquestração customizada que **não usa `crewai.Crew.kickoff`** em runtime. O Triage decide quais ferramentas acionar e o Synthesizer gera a resposta final com streaming.

### Arquitetura do pipeline

```
Entrada do usuário
    │
    ▼
scan_user_input()          ← Guardrail de entrada (security.py)
    │  bloqueada → recusa explícita (0 LLM)
    │  ok ↓
Greeter                    ← Saudações curtas (rule-based, 0 LLM)
    │  saudação → resposta direta
    │  pergunta ↓
Triage Agent               ← Classifica intenção: JSON {use_rag, use_olap, use_ml} (1 LLM)
    │
    ▼
Dispatcher                 ← Executa ferramentas em paralelo (ThreadPoolExecutor)
    ├── RAG Tool            ← Busca híbrida no txtai (0 LLM, só embedding)
    ├── OLAP Tool           ← NL→SQL no DuckDB (1 LLM para gerar SQL)
    └── ML Tool             ← Extrai features e prediz com .pkl (1 LLM)
    │
    ▼
anonymize_messages_for_external()  ← Presidio redige PII antes de enviar ao OpenRouter/Langfuse
    │
    ▼
Synthesizer Agent          ← Gera resposta final com citações (1 LLM, streaming)
    │
    ▼
sanitize_model_output()    ← Sanitização anti-exfiltração markdown (security.py)
    │
    ▼
Resposta ao usuário
```

### Agentes e ferramentas

| Componente | Papel | Chamadas LLM |
|------------|-------|:---:|
| **Greeter** | Responde saudações ("oi", "obrigado") sem acionar o LLM | 0 |
| **Triage Agent** | Classifica a intenção e decide rotas (saída JSON) | 1 |
| **RAG Tool** | Busca híbrida no índice txtai (BM25 + E5 + rerank) | 0 |
| **OLAP Tool** | Converte linguagem natural em SQL e consulta o DuckDB | 1 |
| **ML Tool** | Extrai features via LLM e roda predição com modelo `.pkl` | 1 |
| **Synthesizer** | Gera a resposta final cordial com citações (streaming) | 1 |

### Chamadas LLM por cenário

| Cenário | Total de chamadas |
|---------|:-----------------:|
| Saudação | 0 |
| Pergunta sobre documentos | 2 (triage + synth) |
| Pergunta sobre planilhas | 3 (triage + SQL + synth) |
| Predição ML | 3 (triage + features + synth) |
| Pergunta combinada | 3–4 |

### Trilha de execução (modo depuração)

Ative o toggle **"Mostrar trilha do crew"** em **Desenvolvimento → Parâmetros do chat** para ver um expander abaixo de cada resposta com:

- duração em ms de cada etapa
- entrada/saída resumida
- metadados (rota escolhida, SQL gerado, predições)

Com Langfuse ativo, a mesma execução aparece no painel web (traces + sessions).

Variáveis relevantes: `CREW_TRACE_HANDOFF=1` (ativa trilha), `CREW_PARALLEL_TOOLS=1` (ferramentas em paralelo).

---

## Camada de segurança

Implementada em [`agents/security.py`](apps/streamlit/agents/security.py). Três componentes independentes, em camadas distintas do pipeline:

### Componentes

| Componente | Biblioteca | Onde atua | O que faz |
|------------|-----------|-----------|-----------|
| **BanCode** | regex | Antes do Greeter | Bloqueia mensagens com blocos de código ou scripts (Python, bash, SQL, JS…) |
| **Toxicity (entrada)** | regex + `transformers` opt-in | Antes do Greeter | Bloqueia linguagem ofensiva/inadequada do usuário antes do Triage |
| **Guardrail de entrada** | regex + heurística | Antes do Greeter | Bloqueia prompt injection, jailbreak e mensagens acima de 4.000 chars |
| **Detecção de segredos (entrada)** | detect-secrets | Antes do Triage | Bloqueia mensagens com chaves de API, tokens, JWT |
| **Detecção de segredos (saída)** | detect-secrets | Após o LLM | Redige credenciais que possam ter sido ecoadas de documentos indexados |
| **Anonimização de PII** | Presidio (PT + EN) | Borda externa | Redige e-mail, telefone, CPF, CNPJ antes de enviar ao OpenRouter/Langfuse |
| **Sanitização de saída** | regex | Antes do `st.markdown` | Neutraliza links/imagens markdown para domínios não autorizados |
| **Toxicity (saída)** | regex + `transformers` opt-in | Antes do `st.markdown` | Registra aviso de auditoria se resposta do LLM contiver linguagem inadequada (não bloqueia) |

### Política de fronteira de PII

```
Usuário (autenticado)  ──[PII íntegra]────► vê a resposta completa
OpenRouter             ◄─[PII anonimizada]── prompt + contexto RAG
Langfuse (se ativo)    ◄─[PII anonimizada]── traces
```

A anonimização acontece **apenas na borda externa**: o usuário autenticado vê a resposta completa; o que sai para APIs externas já está redijido. Entidades como `DATE_TIME`, `ORGANIZATION` e `LOCATION` são preservadas por serem relevantes para o contexto de laboratório (validade, fabricante, local de ensaio).

### Configuração via ambiente

| Variável | Padrão | Efeito |
|----------|:------:|--------|
| `SECURITY_BAN_CODE_ENABLED` | `1` | BanCode na entrada (bloqueia código/scripts) |
| `SECURITY_TOXICITY_ENABLED` | `1` | Detecção de toxicidade (entrada bloqueia; saída registra aviso) |
| `SECURITY_TOXICITY_MODEL` | `unitary/toxic-bert` | Modelo HuggingFace para toxicidade (requer `transformers` instalado) |
| `SECURITY_TOXICITY_THRESHOLD` | `0.7` | Score mínimo para classificar como tóxico (0.0–1.0) |
| `SECURITY_INPUT_GUARD_ENABLED` | `1` | Guardrail de entrada |
| `SECURITY_OUTPUT_GUARD_ENABLED` | `1` | Sanitização de saída |
| `SECURITY_PII_REDACTION_ENABLED` | `1` | Anonimização na borda externa |
| `SECURITY_SECRETS_GUARD_ENABLED` | `1` | Detecção de segredos (entrada bloqueia, saída redige) |
| `SECURITY_MAX_INPUT_CHARS` | `4000` | Limite de caracteres por mensagem |
| `SECURITY_PII_LANGUAGES` | `pt,en` | Idiomas do Presidio |
| `SECURITY_ALLOWED_LINK_DOMAINS` | _(vazio)_ | Domínios autorizados em links markdown da saída |

> **Por que a toxicidade não bloqueia a saída?** Modelos alinhados raramente produzem toxicidade espontânea no domínio de laboratório. Bloquear a resposta inteira por um termo isolado causaria mais dano (resposta perdida) do que o risco. O log de auditoria e o campo `neutralized` permitem que o operador decida escalar para bloqueio sem mudar o código.

> **Por que detect-secrets e não LLM Guard para segredos?** O LLM Guard usa os plugins de entropia do detect-secrets por baixo, mas os plugins genéricos de alta entropia geram muitos falsos positivos em texto técnico de laboratório em português. A integração direta com detect-secrets permite configurar uma allowlist precisa de plugins (excluindo `HexHighEntropyString` e `Base64HighEntropyString`) sem instalar toda a cadeia de dependências do LLM Guard.

---

## Pipeline RAG (txtai)

Implementado em `apps/streamlit/rag/`.

### Fluxo de indexação

```
Pasta de projetos → scan (SHA-256) → extração de texto
    → chunking (520 chars, sobreposição 120)
    → índice semântico E5 (TEI) + BM25 em paralelo
    → manifesto de índice (/data/txtai/index_manifest.json)
```

### Fluxo de recuperação

```
Pergunta → vetorização E5 (TEI)
    → busca semântica + busca BM25 em paralelo
    → fusão: score = α × semântico + (1-α) × BM25  (α padrão: 0.4)
    → reranking cross-encoder (mmarco-mMiniLMv2-L12)
    → top-K trechos com citação [Projeto: X, Arquivo: Y]
```

### Módulos

| Módulo | O que faz |
|--------|-----------|
| `rag/extract.py` | Extrai texto de DOCX (parágrafos + tabelas), XLSX, PDF, TXT, MD, CSV |
| `rag/chunking.py` | Divide em chunks de 520 chars, sobreposição 120, lotes de 64; repete o cabeçalho do ensaio (data/título) em cada chunk |
| `rag/index_txtai.py` | Cria e mantém o índice semântico (E5 via TEI) + BM25 |
| `rag/embedding_client.py` | Cliente HTTP para o serviço TEI (vetorização em lote) |
| `rag/hybrid.py` | Fusão semântica + BM25 com peso configurável |
| `rag/rerank.py` | Cross-encoder que reordena os candidatos por relevância par-a-par |
| `rag/manifest.py` | Rastreio SHA-256 dos arquivos indexados (indexação incremental) |

### Busca híbrida

Por padrão o txtai mantém dois índices em paralelo — vetorial (significado) e BM25 (termos exatos). Isso ajuda a recuperar nomes compostos de laboratório (ex.: *tampão de amostra*) que a busca puramente semântica tende a diluir. Peso semântico padrão: `α = 0.4` (`RAG_HYBRID_WEIGHT`). Desative com `RAG_HYBRID_ENABLED=0`.

### Cabeçalho de ensaio nos chunks

O módulo de chunking repete o cabeçalho do documento (data e título do ensaio) em cada chunk gerado. Isso permite que perguntas como *"qual era o protocolo no ensaio de 12/05?"* encontrem os trechos corretos mesmo que a data apareça apenas no início do documento.

### Reranker

Após a recuperação híbrida, o cross-encoder multilíngue (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) reclassifica os chunks pela relevância par-a-par (pergunta ↔ trecho). Desative com `RAG_RERANK_ENABLED=0` ou troque o modelo via `RAG_RERANK_MODEL_ID`.

### Indexação incremental

- **Primeira indexação / migração:** use **Substituir índice existente** (recomendado). Índices criados antes da busca híbrida não têm BM25 — é obrigatório reconstruir uma vez.
- **Reindexações seguintes:** desmarque *Substituir índice existente* — apenas arquivos novos, alterados ou removidos são processados (comparação por SHA-256 e manifesto).

---

## OLAP (DuckDB)

Implementado em `apps/streamlit/olap/`. Permite consultas em linguagem natural sobre planilhas XLSX ingeridas.

| Módulo | O que faz |
|--------|-----------|
| `olap/ingest.py` | Lê XLSX dos projetos e cria tabelas no DuckDB (rastreio SHA-256) |
| `olap/nl_query.py` | Converte linguagem natural em SQL (1 LLM call) e executa |
| `olap/schema_catalog.py` | Gera resumo do esquema para contexto do LLM |
| `olap/connection.py` | Gerencia o ciclo de vida da conexão DuckDB |

**Restrição de segurança:** apenas `SELECT` e `WITH` são permitidos — o DuckDB opera em modo somente leitura.

---

## ML tradicional

Implementado em `apps/streamlit/ml/`. AutoML com FLAML e inferência via chat.

**Dataset padrão:** [AbRank no Kaggle](https://www.kaggle.com/datasets/aurlienplissier/abrank) (`aurlienplissier/abrank`), split `Benchmarks/train_regression.csv`, alvo `log_Aff` (regressão de afinidade anticorpo–antígeno).

| Módulo | O que faz |
|--------|-----------|
| `ml/training.py` | Treina modelo com FLAML (AutoML) |
| `ml/chat_infer.py` | Extrai features via LLM e prediz com o `.pkl` salvo |
| `ml/sequence_embeddings.py` | Embeddings de sequências proteicas (ESM-2) |
| `ml/predict.py` | Interface de predição em lote |
| `ml/catalogs/abrank_kaggle.yaml` | Manifesto do dataset AbRank |

| Variável | Descrição |
|----------|-----------|
| `ASSISTENTE_ML_DIR` | Onde salvar modelos `.pkl` (Docker: `/data/ml`) |
| `ASSISTENTE_ML_CHAT_MODEL` | Caminho do `.pkl` usado no chat |
| `KAGGLE_API_TOKEN` | Token Kaggle (obrigatório fora do Kaggle Notebooks) |
| `KAGGLEHUB_CACHE` | Cache dos downloads (Docker: `/data/ml/kagglehub`) |

---

## Observabilidade (Langfuse)

O [Langfuse](https://langfuse.com/docs) registra cada chamada ao LLM (Triage, NL→SQL, Synthesizer etc.) com prompt, modelo, tokens, latência e erros. A integração usa o [wrapper OpenAI do SDK Python](https://langfuse.com/integrations/model-providers/openai-py) — o código continua usando `openai.OpenAI`; o patch é aplicado automaticamente quando as chaves estão no `.env`.

**Como ativar:**

1. Crie conta em https://cloud.langfuse.com (ou self-host).
2. Em **Settings → API Keys**, gere `LANGFUSE_PUBLIC_KEY` e `LANGFUSE_SECRET_KEY`.
3. Adicione as chaves ao `.env`.
4. Recrie o contêiner: `docker compose up -d --build`.

**Hierarquia de traces:**

| Nome no Langfuse | O que é |
|------------------|---------|
| `chat-turn` | Um turno completo (mensagem → resposta) |
| `crew-pipeline` | Orquestração multiagente |
| `crew-triage` / `crew-synthesizer` | Chamadas LLM nomeadas |
| `olap-nl-to-sql` / `ml-feature-extract` | LLM interno das ferramentas |
| `RAG Tool` / `OLAP Tool` / `ML Tool` | Spans tipo *tool* |

Tags automáticas: `feature:chat`, `route:rag`, `route:olap`, `route:ml` conforme a rota.

| Variável | Efeito |
|----------|--------|
| `LANGFUSE_ENABLED=0` | Desliga o envio de traces sem remover as chaves |
| `LANGFUSE_TRACING_ENVIRONMENT` | Rótulo do ambiente (`development`, `production` etc.) |
| `LANGFUSE_RELEASE` | Versão do deploy (útil para comparar releases) |
| `LANGFUSE_TAGS` | Tags separadas por vírgula (ex.: `streamlit,openrouter`) |

---

## Variáveis de ambiente — referência completa

| Variável | Onde | Descrição |
|----------|------|-----------|
| `PROJETOS_HOST_DIR` | Host | Pasta montada em `/data/projetos` no contêiner (somente leitura) |
| `STREAMLIT_PORT` | Host | Porta publicada (padrão `8502`) |
| `ASSISTENTE_PROJETOS_DIR` | Contêiner | Definido automaticamente pelo Compose como `/data/projetos` |
| `ASSISTENTE_TXTAI_DIR` | Contêiner | Volume persistente do índice RAG (`/data/txtai`) |
| `EMBEDDING_SERVICE_URL` | Contêiner | URL do TEI (padrão `http://embeddings:80`) |
| `EMBEDDING_TIMEOUT_S` | Contêiner | Timeout HTTP para vetorização (padrão `120` s) |
| `LLM_BASE_URL` | Contêiner | Endpoint OpenAI-compatível (padrão `https://openrouter.ai/api/v1`) |
| `LLM_MODEL` | Contêiner | Slug do modelo no catálogo OpenRouter |
| `OPENROUTER_API_KEY` | Contêiner | Chave de API do OpenRouter |
| `OPENROUTER_APP_TITLE` | Contêiner | Título nos rankings do OpenRouter (opcional) |
| `OPENROUTER_HTTP_REFERER` | Contêiner | URL nos rankings (opcional) |
| `LLM_MIN_REQUEST_INTERVAL_S` | Contêiner | Pausa mínima entre chamadas ao LLM (padrão `0`; evals free tier: `12`) |
| `LLM_RETRY_MAX_ATTEMPTS` | Contêiner | Tentativas em erros 429/503 e respostas vazias (padrão `6`; evals: `10`) |
| `LLM_RETRY_BASE_DELAY_S` | Contêiner | Base do backoff exponencial em segundos (padrão `15`; evals: `25`) |
| `LLM_ENABLE_THINKING` | Contêiner | Liga o toggle **Modo raciocínio** por padrão (só faz efeito com Qwen3.5) |
| `LLM_TIMEOUT_S` | Contêiner | Timeout das chamadas ao OpenRouter em segundos (padrão `120`) |
| `CHAT_MAX_TOKENS` | Contêiner | Limite de tokens de saída no chat (padrão `2048`) |
| `RAG_HYBRID_ENABLED` | Contêiner | Ativa busca híbrida BM25 + semântica (padrão `1`) |
| `RAG_HYBRID_WEIGHT` | Contêiner | Peso α do índice semântico na fusão (padrão `0.4`) |
| `RAG_RERANK_ENABLED` | Contêiner | Ativa reranker cross-encoder (padrão `1`) |
| `RAG_RERANK_MODEL_ID` | Contêiner | Modelo cross-encoder (padrão `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) |
| `CREW_TRACE_HANDOFF` | Contêiner | Mostra trilha dos agentes no expander (padrão `1`) |
| `CREW_PARALLEL_TOOLS` | Contêiner | Executa RAG/OLAP/ML em paralelo (padrão `1`) |
| `LANGFUSE_PUBLIC_KEY` | Contêiner | Chave pública do projeto Langfuse |
| `LANGFUSE_SECRET_KEY` | Contêiner | Chave secreta do projeto Langfuse |
| `LANGFUSE_BASE_URL` | Contêiner | API Langfuse (padrão `https://cloud.langfuse.com`) |
| `SECURITY_BAN_CODE_ENABLED` | Contêiner | BanCode na entrada — bloqueia código/scripts (padrão `1`) |
| `SECURITY_TOXICITY_ENABLED` | Contêiner | Toxicidade — entrada bloqueia, saída registra aviso (padrão `1`) |
| `SECURITY_TOXICITY_MODEL` | Contêiner | Modelo HuggingFace de toxicidade (padrão `unitary/toxic-bert`) |
| `SECURITY_TOXICITY_THRESHOLD` | Contêiner | Score mínimo para toxicidade pelo modelo (padrão `0.7`) |
| `SECURITY_INPUT_GUARD_ENABLED` | Contêiner | Guardrail de entrada (padrão `1`) |
| `SECURITY_OUTPUT_GUARD_ENABLED` | Contêiner | Sanitização de saída (padrão `1`) |
| `SECURITY_PII_REDACTION_ENABLED` | Contêiner | Anonimização PII na borda externa (padrão `1`) |
| `SECURITY_SECRETS_GUARD_ENABLED` | Contêiner | Detecção de segredos técnicos (padrão `1`) |

### Volumes dentro do contêiner

| Caminho | Uso |
|---------|-----|
| `/data/projetos` | Bind do host — documentos dos projetos (somente leitura) |
| `/data/txtai` | Volume persistente — índice vetorial RAG |
| `/data/duckdb` | Volume persistente — banco OLAP |
| `/data/ml` | Volume persistente — modelos `.pkl` e cache HuggingFace (ESM-2) |
| `/data` (contêiner `embeddings`) | Volume persistente — cache do modelo TEI |
| `/data/sqlite` | Volume persistente — metadados (fases futuras) |

---

## Desenvolvimento local (sem Docker)

Útil para depurar `app.py` e módulos individuais:

```powershell
cd apps/streamlit
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

$env:ASSISTENTE_PROJETOS_DIR = "D:\caminho\Projetos"
$env:OPENROUTER_API_KEY = "sk-or-v1-sua-chave-aqui"

streamlit run app.py --server.port 8502
```

Fora do Docker, se `ASSISTENTE_PROJETOS_DIR` não estiver definida, o loader usa um fallback para Windows (`projects_loader.py`). Em outros sistemas operacionais, defina a variável.

### Healthcheck do contêiner

```bash
docker inspect --format='{{json .State.Health}}' assistente-lab-streamlit
```

No Windows (PowerShell), use `curl.exe` — o alias `curl` do PowerShell não é o mesmo:

```powershell
curl.exe -s -o NUL -w "HTTP %{http_code}`n" http://127.0.0.1:8502/_stcore/health
```

Esperado: **HTTP 200**. Logs: `docker compose logs streamlit --tail 50`.

### Testar a chave OpenRouter de dentro do contêiner

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

---

## Avaliação end-to-end (DeepEval)

Pipeline de avaliação automatizada usando **[DeepEval](https://docs.confident-ai.com/)** como framework de métricas. Roda dentro do contêiner Docker e avalia as mesmas rotas do chat real (RAG, OLAP, ML, combined).

> **Conflito de dependências:** CrewAI exige `posthog<6`; DeepEval exige `posthog>=7`. Por isso existe um `requirements-evals.txt` separado (sem CrewAI). No Docker, a imagem já inclui os dois conjuntos e o conflito é contornado na ordem de instalação.

### Dataset de goldens — 18 casos reais

Casos derivados dos projetos **252 (Chikungunya ELISA)** e **253 (Dengue ELISA)**:

| Categoria | Qtd | O que avalia |
|-----------|:---:|--------------|
| `rag` | 5 | Perguntas sobre protocolos, validade de antígenos e reagentes (DOCX/PDF) |
| `olap` | 5 | Consultas analíticas sobre amostras e resultados (XLSX no DuckDB) |
| `ml` | 3 | Predição `log_Aff` para pares anticorpo–antígeno de literatura |
| `combined` | 5 | RAG + OLAP combinados na mesma pergunta |

Arquivo fonte: [`apps/streamlit/evals/goldens_projetos_252_253.py`](apps/streamlit/evals/goldens_projetos_252_253.py)

### Fluxo em 3 fases

```
Fase 1 — Bootstrap       Fase 2 — Geração          Fase 3 — Métricas
eval_bootstrap.py     →  harness.py + run_crew_chat  →  judge_model.py + DeepEval
(throttle, retry)        18 perguntas → JSON              scores por caso → JSON
```

**Fase 1 — Bootstrap (`eval_bootstrap.py`):** desliga Langfuse, configura throttle conservador para o tier gratuito (`LLM_MIN_REQUEST_INTERVAL_S=12`, `LLM_RETRY_MAX_ATTEMPTS=10`, `LLM_RETRY_BASE_DELAY_S=25`).

**Fase 2 — Geração:** chama o mesmo `run_crew_chat()` do chat real para cada golden. Resultado salvo em `evals/results/eval_test_cases_<stamp>.json`.

**Fase 3 — Métricas:** um juiz LLM avalia cada par (pergunta, resposta, contexto) com as métricas do DeepEval.

### Executar no Docker (recomendado)

```powershell
# Todos os 18 casos (fases 2 + 3)
.\scripts\run_evals_docker.ps1

# Limitar casos e filtrar categoria
.\scripts\run_evals_docker.ps1 --limit 5 --category rag
```

**Modo manual (mais controle):**

```bash
# Fase 2 apenas — gera respostas sem LLM-as-judge
docker compose exec -e LANGFUSE_ENABLED=0 streamlit \
    python evals/run_assistente_eval.py --require-ready --skip-metrics

# Fase 3 apenas — retoma métricas de um JSON já gerado
docker compose exec -e LANGFUSE_ENABLED=0 streamlit \
    python evals/run_assistente_eval.py \
    --resume-metrics evals/results/eval_test_cases_20260531_120000.json
```

### Argumentos do CLI

| Argumento | Padrão | Descrição |
|-----------|--------|-----------|
| `--dataset PATH` | auto-detect | JSON/JSONL com os goldens |
| `--limit N` | todos | Executar apenas os N primeiros casos |
| `--category CAT` | todos | Filtrar por `rag`, `olap`, `ml`, `combined`, `out_of_scope` |
| `--skip-metrics` | — | Pular fase 3 (só gera respostas, salva JSON) |
| `--skip-unavailable` | — | Ignorar casos que exigem infra ausente |
| `--require-ready` | — | Abortar se RAG/OLAP/ML não estiverem prontos |
| `--resume-metrics PATH` | — | Retomar fase 3 a partir de JSON da fase 2 |
| `--judge-provider` | auto | `openrouter` ou `openai` |
| `--judge-model` | auto | Slug do modelo juiz |
| `--request-interval S` | env | Sobrescreve `LLM_MIN_REQUEST_INTERVAL_S` |

### Executar fora do Docker (venv separado)

```powershell
python -m venv .venv-evals
.\.venv-evals\Scripts\Activate.ps1
pip install -r apps/streamlit/requirements-evals.txt

$env:OPENROUTER_API_KEY = "sk-or-v1-..."
$env:ASSISTENTE_PROJETOS_DIR = "D:\caminho\Projetos"
$env:LANGFUSE_ENABLED = "0"

cd apps/streamlit
python evals/run_assistente_eval.py --skip-unavailable --skip-metrics
```

> Fora do Docker, o serviço TEI não está disponível — casos RAG são pulados automaticamente com `--skip-unavailable`.

### Executar via pytest / DeepEval CLI

```bash
# Smoke test (3 casos, sem métricas)
set EVAL_LIMIT=3
set EVAL_SKIP_METRICS=1
deepeval test run apps/streamlit/evals/test_assistente_e2e.py
```

| Variável | Padrão | Efeito |
|----------|:------:|--------|
| `EVAL_LIMIT` | `3` | Número de casos no smoke test |
| `EVAL_CATEGORY` | todos | Filtro de categoria |
| `EVAL_SKIP_METRICS` | `0` | `1` = pular LLM-as-judge |
| `EVAL_REQUIRE_READY` | `0` | `1` = abortar se infra incompleta |

### Throttle para o tier gratuito

O tier gratuito do OpenRouter tem limite de ~20 req/min. Para evitar erros `choices ausente` em avaliações longas:

| Variável | Free tier | O que faz |
|----------|:---------:|-----------|
| `LLM_MIN_REQUEST_INTERVAL_S` | `12` | Pausa mínima entre chamadas |
| `LLM_RETRY_MAX_ATTEMPTS` | `10` | Tentativas em 429/503 e respostas vazias |
| `LLM_RETRY_BASE_DELAY_S` | `25` | Base do backoff exponencial |
| `EVAL_METRICS_THROTTLE_S` | `15` | Pausa entre avaliações do juiz (fase 3) |

### Estrutura dos resultados

```
evals/
├── datasets/
│   ├── assistente_lab_goldens_<stamp>.json
│   └── assistente_lab_goldens_<stamp>.jsonl
└── results/
    ├── eval_test_cases_<stamp>.json    ← fase 2: input + actual_output + retrieval_context
    └── eval_results_<stamp>.json       ← fase 3: test_cases + scores por métrica
```

### Adicionar novos goldens

Edite `goldens_projetos_252_253.py` adicionando um `ChatGolden` na função da categoria correspondente (`_rag_goldens`, `_olap_goldens`, `_ml_goldens` ou `_combined_goldens`):

```python
ChatGolden(
    golden_id="rag-elisa-concentracao-01",
    input="Qual a concentração do anticorpo primário no protocolo ELISA?",
    expected_output="A concentração recomendada é 1:1000 em PBS-T...",
    context=["trecho do documento que contém a resposta"],
    category=EvalCategory.RAG,
    expected_routes=ExpectedRoutes(documents=True),
    requires_index=True,
    requires_olap=False,
    requires_ml_model=False,
    project_ids=["252"],
    tags=["elisa", "anticorpo", "concentração"],
    comments="Fonte: protocolo_elisa_v3.docx",
)
```

Após adicionar/remover casos, ajuste a asserção `assert len(items) == 18` em `build_projetos_goldens()` para o novo total. Depois exporte o dataset atualizado:

```bash
docker compose exec streamlit python evals/golden_dataset_template.py
```

---

## Rastreio código ↔ funcionalidade

| Necessidade | Começar em |
|-------------|------------|
| UI, abas, inventário | [`apps/streamlit/app.py`](apps/streamlit/app.py) |
| Pipeline de chat (orquestrador) | [`apps/streamlit/agents/runner.py`](apps/streamlit/agents/runner.py) |
| Pipeline de chat (Triage + Dispatcher) | [`apps/streamlit/agents/crew.py`](apps/streamlit/agents/crew.py) |
| Roteamento de intenção (regras) | [`apps/streamlit/agents/intent_rules.py`](apps/streamlit/agents/intent_rules.py) |
| Guardrails, PII, segredos | [`apps/streamlit/agents/security.py`](apps/streamlit/agents/security.py) |
| Configuração do LLM (defaults, retry) | [`apps/streamlit/llm_config.py`](apps/streamlit/llm_config.py) |
| Perfis de sampling (Qwen3.5 e outros) | [`apps/streamlit/qwen35_inference.py`](apps/streamlit/qwen35_inference.py) |
| Observabilidade Langfuse | [`apps/streamlit/observability/langfuse_client.py`](apps/streamlit/observability/langfuse_client.py) |
| Extração, chunking, índice, busca (RAG) | [`apps/streamlit/rag/`](apps/streamlit/rag/) |
| Consultas analíticas NL→SQL (OLAP) | [`apps/streamlit/olap/`](apps/streamlit/olap/) |
| AutoML e inferência no chat (ML) | [`apps/streamlit/ml/`](apps/streamlit/ml/) |
| Scan de projetos | [`apps/streamlit/projects_loader.py`](apps/streamlit/projects_loader.py) |
| Avaliação DeepEval (CLI) | [`apps/streamlit/evals/run_assistente_eval.py`](apps/streamlit/evals/run_assistente_eval.py) |
| Runtime de avaliação | [`apps/streamlit/evals/harness.py`](apps/streamlit/evals/harness.py) |
| Dataset de goldens | [`apps/streamlit/evals/goldens_projetos_252_253.py`](apps/streamlit/evals/goldens_projetos_252_253.py) |
| Throttle e retry de evals | [`apps/streamlit/evals/eval_bootstrap.py`](apps/streamlit/evals/eval_bootstrap.py) |
| Testes unitários (~195) | [`apps/streamlit/tests/`](apps/streamlit/tests/) |
| Imagem Docker e healthcheck | [`docker/streamlit/Dockerfile`](docker/streamlit/Dockerfile) |
| Portas, volumes, variáveis de ambiente | [`docker-compose.yml`](docker-compose.yml) |

---

## Problemas comuns — referência completa

| Sintoma | Causa provável | O que fazer |
|---------|----------------|-------------|
| Pasta vazia ou "Caminho não existe" | `PROJETOS_HOST_DIR` incorreto | Use o caminho absoluto real no host; um subdiretório por projeto |
| **HTTP 401 no chat** | `OPENROUTER_API_KEY` ausente ou inválida | Verifique o `.env`, recrie a chave e rode `docker compose up -d --build` |
| **HTTP 402 / saldo insuficiente** | Modelo pago sem créditos | Troque para `openrouter/free` em `LLM_MODEL` ou adicione créditos no painel |
| **HTTP 429 (rate limit)** | Limite do plano gratuito atingido | Aguarde alguns minutos; reduza com `CREW_PARALLEL_TOOLS=0` |
| Langfuse inativo no diagnóstico | Chaves ausentes ou `LANGFUSE_ENABLED=0` | Preencha `LANGFUSE_*` no `.env` e recrie o contêiner |
| Chat sem resposta / timeout | Sem conexão do contêiner com a internet | Teste `docker exec assistente-lab-streamlit curl -sI https://openrouter.ai/api/v1/models`; aumente `LLM_TIMEOUT_S` |
| Inventário vazio | Sem subpastas na raiz ou extensões filtradas | Ajuste o filtro na barra lateral; confira as extensões dos arquivos |
| **RAG não retorna documentos esperados** | Índice desatualizado ou sem BM25 | Escaneie → Indexação RAG → **Substituir índice** → reconstrua |
| Termo composto não aparece (ex.: *tampão de amostra*) | α muito alto ou índice sem BM25 | Reconstrua o índice híbrido; ajuste `RAG_HYBRID_WEIGHT` (ex.: `0.35`) |
| Busca só mostra planilhas grandes | Projeto com muitos chunks XLSX domina o top-K | Refine a pergunta ou use **Desenvolvimento → Busca** com mais trechos |
| `unhealthy` no Docker | Streamlit não subiu | `docker compose logs streamlit --tail 50` |
| Permissão negada em arquivos | Bind somente leitura | O usuário da imagem (`uid 10001`) precisa ter permissão de leitura nos arquivos do host |
| Mensagem bloqueada pelo guardrail | Padrão suspeito detectado (injection, segredo) | Reescreva a mensagem sem colar credenciais ou comandos de sistema |
| **Eval abortada** (`RuntimeError: choices ausente`) | OpenRouter free retornou HTTP 200 sem `choices` | O retry automático já cobre isso; se persistir, aumente `LLM_MIN_REQUEST_INTERVAL_S` para `15–20` |
| Eval muito lenta (>1 h para 18 casos) | Backoff exponencial longo | Reduza `LLM_RETRY_BASE_DELAY_S` para `15` e `LLM_MIN_REQUEST_INTERVAL_S` para `10` |
| Conflito `posthog` no `pip install` | CrewAI e DeepEval no mesmo venv | Use `requirements-evals.txt` (sem CrewAI); no Docker o conflito já está resolvido |
| Casos `ml` pulados com `--skip-unavailable` | Modelo `.pkl` não treinado ou caminho inválido | Treine na aba ML e configure `ASSISTENTE_ML_CHAT_MODEL`; ou rode apenas `--category rag` |
