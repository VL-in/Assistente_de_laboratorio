# Assistente de lab

Aplicação web **MVP offline-first** para apoiar P&D com documentos locais, **RAG** (planejado: txtai), **OLAP** (planejado: DuckDB) e geração via **LM Studio** no host. A UI e o pipeline Python rodam **no Docker**; dados sensíveis e modelo ficam na sua máquina.

| Documento | Função |
|-----------|--------|
| [`.cursor/plans/20260429playbook_mvp_biotech.plan.md`](.cursor/plans/20260429playbook_mvp_biotech.plan.md) | Playbook: fases, escopo, riscos e **tabela de progresso** atualizada |
| [`.env.docker.example`](.env.docker.example) | Modelo de `.env` para Compose: caminhos, LM Studio e **placeholders** de chaves de API (copiar para `.env`) |

---

## Visão rápida do repositório

```
Assistente_de_lab/
├── docker-compose.yml          # Orquestração: Streamlit + volumes + rede ao host
├── docker/streamlit/Dockerfile # Imagem da app (usuário não-root, HEALTHCHECK HTTP)
├── .env.docker.example         # Modelo de `.env` na raiz (segredos ficam só no `.env`)
├── apps/streamlit/
│   ├── app.py                  # UI: abas (Início, Fontes, RAG, Chat, OLAP, Diagnóstico)
│   ├── projects_loader.py      # Inventário: um subdir. de 1º nível = projeto; walk recursivo
│   └── requirements.txt
└── apps/api/                   # FastAPI opcional (fora do caminho crítico do MVP)
```

**Convenção de projeto:** dentro da pasta configurada como raiz (ex.: `Projetos`), cada **subdiretório imediato** é um **projeto** (`project_id` = nome da pasta). Pastas como `planning/` ou `results/` pertencem ao mesmo projeto.

---

## Executar com Docker (recomendado)

**Pré-requisitos:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) ou Docker Engine + Compose v2 (Linux). Para testar LLM a partir do contêiner: [LM Studio](https://lmstudio.ai/) com servidor local ativo no host.

1. Na **raiz** do repositório, copie [`.env.docker.example`](.env.docker.example) para **`.env`** e edite pelo menos `PROJETOS_HOST_DIR` (caminho absoluto no **host** da pasta que contém um subdiretório por projeto). Chaves de API reais ficam **apenas** no `.env` (arquivo ignorado pelo Git).

2. Suba o stack:

   ```bash
   docker compose up --build
   ```

3. Abra no navegador: **`http://127.0.0.1:8502`** (ou `http://127.0.0.1:${STREAMLIT_PORT}` se alterou o `.env`).

O serviço usa `restart: unless-stopped` e `PYTHONUNBUFFERED=1` para logs mais imediatos.

### Variáveis de ambiente (Compose / `.env`)

| Variável | Onde | Descrição |
|----------|------|-----------|
| `PROJETOS_HOST_DIR` | Host | Pasta montada em `/data/projetos` no contêiner (somente leitura). |
| `STREAMLIT_PORT` | Host | Porta publicada (padrão **8502**). Deve coincidir com a porta interna mapeada `HOST:8502→8502`. |
| `ASSISTENTE_PROJETOS_DIR` | Contêiner | Definido no Compose como `/data/projetos` (não precisa mudar no uso normal). |
| `LLM_BASE_URL` | Contêiner | Base OpenAI-compatível (ex.: `http://172.21.64.1:1234` ou `http://host.docker.internal:1234/v1`). Pode omitir o sufixo `/v1`; o app completa automaticamente. |
| `LLM_MODEL` | Contêiner | ID exato do modelo no LM Studio. |
| `OPENAI_API_KEY` | Contêiner | Valor dummy aceito pelo LM Studio na maioria dos setups. |

### Volumes dentro do contêiner

| Caminho | Uso |
|---------|-----|
| `/data/projetos` | Bind do host: documentos dos projetos (**RO**). |
| `/data/txtai` | Volume nomeado: índice/embeddings (fases futuras). |
| `/data/duckdb` | Volume nomeado: OLAP. |
| `/data/sqlite` | Volume nomeado: metadados/auditoria (fases futuras). |

### Saúde do contêiner (HEALTHCHECK)

A imagem define verificação HTTP em **`http://127.0.0.1:8502/_stcore/health`** (endpoint interno do Streamlit). Para inspecionar no host:

```bash
docker inspect --format='{{json .State.Health}}' assistente-lab-streamlit
```

Se o status ficar `unhealthy`, confira se o processo Streamlit subiu (logs: `docker compose logs streamlit`) e se a versão do Streamlit expõe `/_stcore/health` (Streamlit recente).

### Problemas comuns

- **Pasta vazia ou “Caminho não existe”:** `PROJETOS_HOST_DIR` no `.env` incorreto ou drive não montado no Linux (use caminho absoluto real).
- **LM Studio inalcançável no Diagnóstico:** servidor local desligado, firewall, ou URL errada. No Docker Desktop (Windows/macOS) use `host.docker.internal`; no Linux o Compose já inclui `extra_hosts: host.docker.internal:host-gateway`.
- **Permissão negada em arquivos:** o bind é **somente leitura**; o usuário da imagem (`uid 10001`) precisa de permissão de leitura no host (no Windows com Docker Desktop costuma funcionar sem ajuste extra).
- **Inventário vazio:** nenhum subdiretório na raiz, ou extensões na barra lateral não batem com os arquivos (veja filtro de extensões na UI).

---

## Desenvolvimento local (sem Docker)

Útil para depurar `app.py` e `projects_loader.py`:

```powershell
cd apps/streamlit
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
# Opcional: $env:ASSISTENTE_PROJETOS_DIR="D:\caminho\Projetos"
streamlit run app.py --server.port 8502
```

Fora do Docker, se `ASSISTENTE_PROJETOS_DIR` não estiver definida e não houver `/.dockerenv`, o loader usa um **fallback Windows** em `projects_loader.py` (ajuste por env em outros SO).

---

## Rastreio código ↔ produto

| Necessidade | Começar em |
|-------------|------------|
| UI, inventário | `apps/streamlit/app.py`, `projects_loader.py` |
| Chat com LM Studio (OpenAI-compat) | `apps/streamlit/app.py` (SDK `openai`, env `LLM_BASE_URL`, `LLM_MODEL`) |
| Regra “um projeto por pasta”, scan, hash opcional | `apps/streamlit/projects_loader.py` |
| Imagem e healthcheck | `docker/streamlit/Dockerfile` |
| Portas, volumes, env injetada | `docker-compose.yml` |

Próximas entregas planejadas estão descritas por fase no **playbook** (parsing, txtai, DuckDB na UI, autenticação, etc.).

---

## API FastAPI (`apps/api`, opcional)

Serviço HTTP separado para testes ou integrações; **não** substitui o MVP principal (Streamlit no Docker). Ver `apps/api/README.md` e `apps/api/.env.example`.
