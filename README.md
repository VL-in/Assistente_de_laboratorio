# Assistente de lab

Aplicação web **MVP offline-first** para apoiar P&D com documentos locais, **RAG** (txtai + embeddings multilíngues), **OLAP** (planejado: DuckDB) e geração via **LM Studio** no host. A UI e o pipeline Python rodam **no Docker**; dados sensíveis e o modelo de linguagem ficam na sua máquina.

| Documento | Função |
|-----------|--------|
| [`.cursor/plans/20260429playbook_mvp_biotech.plan.md`](.cursor/plans/20260429playbook_mvp_biotech.plan.md) | Playbook: fases, escopo, riscos e **tabela de progresso** |
| [`.env.docker.example`](.env.docker.example) | Modelo de `.env` para Compose (copiar para `.env` na raiz) |

---

## Visão rápida do repositório

```
Assistente_de_lab/
├── docker-compose.yml          # Orquestração: Streamlit + volumes + rede ao host
├── docker/streamlit/Dockerfile # Imagem da app (usuário não-root, HEALTHCHECK HTTP)
├── .env.docker.example         # Modelo de `.env` (segredos ficam só no `.env`)
├── apps/streamlit/
│   ├── app.py                  # UI: Conversa, Documentos, Desenvolvimento (RAG, OLAP, diagnóstico)
│   ├── chat_router.py          # Roteador: documentos vs planilhas por mensagem
│   ├── projects_loader.py      # Inventário: um subdiretório de 1º nível = projeto
│   ├── qwen35_inference.py     # Parâmetros Qwen3.5 / strip de thinking
│   ├── olap/                   # DuckDB: ingestão, NL→SQL, catálogo
│   ├── rag/                    # Extração, chunking, índice txtai (upsert em lotes)
│   └── requirements.txt
```

**Convenção de projeto:** na pasta configurada como raiz (ex.: `Projetos`), cada **subdiretório imediato** é um **projeto** (`project_id` = nome da pasta). Pastas como `planning/` ou `results/` pertencem ao mesmo projeto.

---

## Guia rápido (primeira execução)

Siga esta ordem para subir a aplicação sem erros de rede, LLM ou RAG.

### 1. Pré-requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) ou Docker Engine + Compose v2 (Linux)
- [LM Studio](https://lmstudio.ai/) com um modelo de chat carregado e o **servidor local** ativo (porta padrão **1234**)
- Uma pasta de projetos no host, com **um subdiretório por projeto** e documentos (`.docx`, `.xlsx`, `.pdf`, `.txt`, `.md`, `.csv`)

### 2. Configurar `.env`

Na **raiz** do repositório:

```bash
cp .env.docker.example .env
```

Edite o `.env` (nunca commite este arquivo):

| Variável | Obrigatório | O que preencher |
|----------|-------------|-----------------|
| `PROJETOS_HOST_DIR` | Sim | Caminho **absoluto** no host da pasta que contém os projetos (ex.: `D:/Vanessa/AI_project/Projetos`) |
| `LLM_BASE_URL` | Sim | URL do LM Studio **acessível de dentro do contêiner** (veja [LM Studio](#lm-studio-no-docker)) |
| `LLM_MODEL` | Sim | ID **exato** do modelo na aba Server do LM Studio (ex.: `qwen3.5-9b-mtp`) |
| `OPENAI_API_KEY` | Recomendado | Valor dummy, ex.: `lm-studio` |
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

1. **Desenvolvimento → Diagnóstico** — confirme pasta de projetos e clique em **Testar GET /v1/models** (LM Studio deve responder).
2. **Documentos** — informe a raiz (ou use a do `.env`) e **Escanear pastas**.
3. **Documentos** — **Atualizar base agora** (primeira vez ou após mudanças nos arquivos).
4. **Desenvolvimento → Busca semântica** — valide recuperação de trechos.
5. **Conversa** — envie uma mensagem; documentos e planilhas entram automaticamente quando disponíveis.

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
| `LLM_BASE_URL` | Contêiner | Base OpenAI-compatível do LM Studio. Pode omitir o sufixo `/v1`; o app adiciona automaticamente. |
| `LLM_MODEL` | Contêiner | ID exato do modelo carregado no LM Studio. |
| `OPENAI_API_KEY` | Contêiner | Valor dummy aceito pelo LM Studio na maioria dos setups. |

### LM Studio no Docker

O contêiner **não** roda o LM Studio; ele só se conecta ao servidor no **host** ou na **rede local**. A URL em `LLM_BASE_URL` precisa funcionar **de dentro do contêiner**, não apenas no navegador do PC.

| Cenário | Exemplo de `LLM_BASE_URL` | Observação |
|---------|---------------------------|------------|
| Docker Desktop (Windows/macOS), LM Studio no mesmo PC | `http://host.docker.internal:1234` | Opção mais simples no Desktop |
| LM Studio em outro PC na rede (IP fixo) | `http://192.168.15.7:1234` | Use o IP real da máquina que hospeda o LM Studio |
| IP do host visto pelo Docker (WSL2 / rede virtual) | `http://172.x.x.x:1234` | Só use se `curl` **de dentro do contêiner** responder |

**Como testar se a URL está correta** (substitua pela sua URL):

```bash
docker exec assistente-lab-streamlit python -c "
import json, os, urllib.request
base = os.environ['LLM_BASE_URL'].rstrip('/')
if not base.endswith('/v1'): base += '/v1'
with urllib.request.urlopen(base + '/models', timeout=10) as r:
    print(json.loads(r.read().decode())['data'][0]['id'])
"
```

Se aparecer o nome do modelo, o chat na UI deve funcionar. Se der timeout ou *No route to host*, troque `LLM_BASE_URL`, salve o `.env` e rode `docker compose up -d --build` de novo.

No LM Studio: aba **Server** → ative o servidor local → carregue o modelo → copie o **ID** para `LLM_MODEL`.

### Qwen3.5-9B-MTP (Unsloth GGUF)

Modelo padrão do MVP: [`unsloth/Qwen3.5-9B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF). O Streamlit aplica os parâmetros recomendados no model card:

| Modo | Uso no app | Parâmetros (resumo) |
|------|------------|---------------------|
| **Instruct** (padrão) | Chat + RAG + documentos | `enable_thinking=false`, `temperature=0.7`, `top_p=0.8` |
| **Thinking** | Toggle na aba Chat | `enable_thinking=true`, `temperature=1.0`, `top_p=0.95` |
| **SQL OLAP** | Geração de `SELECT` | `enable_thinking=false`, `temperature=0.6` (tarefa precisa) |

Variáveis opcionais no `.env`:

| Variável | Efeito |
|----------|--------|
| `LLM_ENABLE_THINKING=1` | Liga o toggle **Modo raciocínio** por padrão |

**Desempenho MTP no LM Studio / llama.cpp:** use build com suporte a *draft MTP* e carregue o GGUF MTP; a documentação Unsloth cita `--spec-type draft-mtp --spec-draft-n-max 6` no `llama-server` para ~1,5–2× mais velocidade na geração (configuração do servidor, não do app).

Implementação no código: `apps/streamlit/qwen35_inference.py`.

### Volumes dentro do contêiner

| Caminho | Uso |
|---------|-----|
| `/data/projetos` | Bind do host: documentos dos projetos (**somente leitura**) |
| `/data/txtai` | Volume nomeado: índice vetorial RAG (persiste entre reinícios) |
| `/data/duckdb` | Volume nomeado: OLAP (fases futuras) |
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
$env:LLM_BASE_URL="http://192.168.15.7:1234"
$env:LLM_MODEL="qwen3.5-9b-mtp"
$env:OPENAI_API_KEY="lm-studio"
streamlit run app.py --server.port 8502
```

Fora do Docker, se `ASSISTENTE_PROJETOS_DIR` não estiver definida, o loader usa um fallback Windows em `projects_loader.py` (em outros SO, defina a variável).

LM Studio em `127.0.0.1:1234` funciona no modo local; no Docker use `host.docker.internal` ou o IP da rede conforme a tabela acima.

---

## Problemas comuns

| Sintoma | Causa provável | O que fazer |
|---------|----------------|-------------|
| Pasta vazia ou “Caminho não existe” | `PROJETOS_HOST_DIR` incorreto | Caminho absoluto real no host; um subdiretório por projeto |
| **LM Studio inalcançável** no Diagnóstico | Servidor desligado, firewall ou URL que não funciona **no contêiner** | Teste com `docker exec` (seção [LM Studio](#lm-studio-no-docker)); prefira `host.docker.internal` ou IP LAN (`192.168.x.x`) |
| Chat sem resposta / erro de conexão | `LLM_MODEL` diferente do ID no LM Studio | Copie o ID exato da aba Server |
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
| Extração, chunking, índice, busca | `apps/streamlit/rag/` |
| Scan de projetos | `apps/streamlit/projects_loader.py` |
| Imagem e healthcheck | `docker/streamlit/Dockerfile` |
| Portas, volumes, env | `docker-compose.yml` |

Próximas entregas (DuckDB na UI, autenticação, etc.) estão no **playbook**.
