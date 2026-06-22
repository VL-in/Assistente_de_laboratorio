title: Assistente de Laboratório
  emoji: 🧬
  colorFrom: blue
  colorTo: green
  sdk: docker
  app_port: 7860
  pinned: false

# Assistente de Laboratório

![Python](https://img.shields.io/badge/python-3.12-blue) ![License](https://img.shields.io/badge/License-MIT-white)

Assistente de laboratório para P&D que responde perguntas sobre experimentos passados consultando documentos internos, analisa planilhas de ensaios e prediz afinidade de pares anticorpo–antígeno. Toda a infraestrutura roda localmente via Docker — apenas as chamadas de linguagem saem para a API do OpenRouter.

## Funcionalidades

- Escaneamento de documentos (`.docx`, `.xlsx`, `.xlsm`, `.pdf`, `.txt`, `.md`, `.csv`) locais ou de volumes persistentes.
- Consulta agêntica de documentações internas (protocolos, insumos, resultados de ensaios) via RAG híbrido (BM25 + semântica E5).
- Análise de planilhas via NL→SQL (DuckDB), sem escrever SQL manualmente.
- Predição de afinidade anticorpo–antígeno por regressão AutoML (FLAML + ESM-2).
- Respostas sintetizadas com referência ao arquivo e projeto de origem.
- Camada de segurança: redação de PII (Presidio), detecção de prompt injection, sanitização de saída.

## Pré-requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) ou Docker Engine + Compose v2 (Linux)
- Conta no [OpenRouter](https://openrouter.ai/) com **chave de API** ativa (começa com `sk-or-v1-...`)
- Uma pasta de projetos no host com **um subdiretório por projeto** e documentos nos formatos listados acima

> **Por que OpenRouter?** Ele expõe centenas de modelos (incluindo os gratuitos) por uma única API compatível com OpenAI. Os modelos com sufixo `:free` são gratuitos com limites de requisição.

## Instalação

### 1. Clonar e configurar o ambiente

Na raiz do repositório, copie o arquivo de exemplo e edite com suas credenciais:

```bash
cp .env.example .env
```

Preencha no `.env` pelo menos:

| Variável | O que preencher |
|----------|-----------------|
| `PROJETOS_HOST_DIR` | Caminho absoluto da pasta de projetos no host (ex.: `D:/Lab/Projetos`) |
| `OPENROUTER_API_KEY` | Sua chave em [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys) |
| `LLM_MODEL` | Slug do modelo (padrão `openrouter/auto`) — veja [Escolha do modelo](#escolha-do-modelo) |

> Nunca faça commit do `.env` — ele contém segredos.

### 2. Subir a aplicação

```bash
docker compose up --build
```

Para rodar em segundo plano:

```bash
docker compose up -d --build
```

Acesse no navegador: **http://127.0.0.1:8502**

Após alterar o `.env`, recrie o contêiner para aplicar as mudanças:

```bash
docker compose up -d --build
```

## Uso

### Primeira execução

1. **Desenvolvimento → Diagnóstico** — confirme que a chave de API aparece como configurada e teste **GET /v1/models**.
2. **Documentos** — informe a raiz dos projetos (ou use a do `.env`) e clique em **Escanear pastas**.
3. **Documentos → Indexação RAG** — clique em **Atualizar base agora** (obrigatório na primeira vez).
4. **Desenvolvimento → Busca híbrida** — valide a recuperação de trechos (BM25 + semântica).
5. **Conversa** — envie uma pergunta; o sistema roteia automaticamente para documentos, planilhas ou ML conforme disponível.
6. **ML tradicional** — carregue o dataset **AbRank (Kaggle)**, treine o modelo FLAML (`log_Aff`), salve o `.pkl` e teste a predição via chat.

### Escolha do modelo

`LLM_MODEL` aceita qualquer slug de [openrouter.ai/models](https://openrouter.ai/models):

| Slug | Quando usar | Custo |
|------|-------------|-------|
| `openrouter/auto` | Padrão — o roteador escolhe o melhor para cada requisição | Mesmo do modelo selecionado |
| `openrouter/free` | Sorteia entre modelos gratuitos disponíveis | Grátis (limites por minuto/dia) |
| `meta-llama/llama-3.3-70b-instruct:free` | Modelo grande e multilíngue, bom para PT-BR | Grátis |
| `qwen/qwen3.5-7b-instruct:free` | Os perfis de sampling Qwen são ativados automaticamente | Grátis |
| `anthropic/claude-3.5-sonnet` | Alta qualidade para síntese complexa | Pago |

### Convenção de projetos

Cada **subdiretório imediato** da pasta raiz configurada (ex.: `Projetos/`) é um projeto. Subpastas internas como `planning/` ou `results/` pertencem ao mesmo projeto.

```
Projetos/
├── projeto-dengue/        ← project_id = "projeto-dengue"
│   ├── protocolo.docx
│   └── resultados/
│       └── ensaio_01.xlsx
└── projeto-chikungunya/   ← project_id = "projeto-chikungunya"
    └── ...
```

## Arquitetura

```
Documentos locais / volume
        │
        ▼
projects_loader → extração/chunking → txtai (RAG híbrido BM25 + E5)
                                             │
        ┌────────────────────────────────────┤
        │                                    │
  DuckDB (OLAP, NL→SQL)           OpenRouter (LLM remoto)
        │                                    │
        └──────── Pipeline multiagente ──────┘
                  Triage → Tools → Synthesizer
                        │
                   Streamlit UI
```

O LLM é **sempre remoto** (OpenRouter). Os documentos e índices ficam **locais ou em volumes persistentes** — nada de dado sensível vai para a nuvem.

## Problemas comuns

| Sintoma | Causa provável | O que fazer |
|---------|----------------|-------------|
| Pasta vazia ou "Caminho não existe" | `PROJETOS_HOST_DIR` incorreto | Use o caminho absoluto real no host; um subdiretório por projeto |
| **HTTP 401 no chat** | `OPENROUTER_API_KEY` ausente ou inválida | Verifique o `.env`, recrie a chave e rode `docker compose up -d --build` |
| **HTTP 402 / saldo insuficiente** | Modelo pago sem créditos | Troque para `openrouter/free` em `LLM_MODEL` ou adicione créditos no painel |
| **HTTP 429 (rate limit)** | Limite do plano gratuito atingido | Aguarde alguns minutos; reduza com `CREW_PARALLEL_TOOLS=0` |
| Chat sem resposta / timeout | Sem conexão do contêiner com a internet | Teste `docker exec assistente-lab-streamlit curl -sI https://openrouter.ai/api/v1/models`; aumente `LLM_TIMEOUT_S` |
| **RAG não retorna documentos esperados** | Índice desatualizado ou sem BM25 | Escaneie → Indexação RAG → **Substituir índice** → reconstrua |
| `unhealthy` no Docker | Streamlit não subiu | `docker compose logs streamlit --tail 50` |
| Mensagem bloqueada pelo guardrail | Padrão suspeito detectado (injection, segredo) | Reescreva a mensagem sem colar credenciais ou comandos de sistema |

Para mais problemas e detalhes técnicos, consulte o [TECHNICAL.md](TECHNICAL.md).

## Contribuindo

Pull requests são bem-vindos. Para mudanças maiores, abra uma issue primeiro para discutir o que você gostaria de mudar.

Antes de contribuir, consulte o [TECHNICAL.md](TECHNICAL.md) para entender a arquitetura detalhada, a estrutura de módulos e como rodar os testes.

```bash
# Rodar testes unitários (~195 testes)
cd apps/streamlit
python -m pytest tests/
```

## Licença

[MIT](https://github.com/VL-in/Assistente_de_laboratorio/blob/main/License)
