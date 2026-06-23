
# Assistente de Laboratório

![Python](https://img.shields.io/badge/python-3.12-blue) ![License](https://img.shields.io/badge/License-MIT-white) ![GitHub stars](https://img.shields.io/github/stars/VL-in/Assistente_de_laboratorio)

Assistente de laboratório para P&D que responde perguntas sobre experimentos passados consultando documentos internos, analisa planilhas de ensaios e prediz afinidade de pares anticorpo–antígeno. Toda a infraestrutura roda localmente via Docker — apenas as chamadas de linguagem saem para a API do OpenRouter.

## Funcionalidades

- Escaneamento de documentos (`.docx`, `.xlsx`, `.xlsm`, `.pdf`, `.txt`, `.md`, `.csv`) locais ou de volumes persistentes.
- Consulta agêntica de documentações internas (protocolos, insumos, resultados de ensaios) via RAG híbrido (BM25 + semântica E5).
- Análise de planilhas via lingua natural → SQL (DuckDB), sem escrever SQL manualmente.
- Predição de afinidade anticorpo–antígeno por regressão, fornecendo sequencias de fasta do heavy chain, light chain e do antígeno.
- Respostas sintetizadas com referência ao arquivo e projeto de origem.
- Camada de segurança: redação de PII (Presidio), detecção de prompt injection, sanitização de saída.

## Pré-requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) ou Docker Engine + Compose v2 (Linux)
- Conta no [OpenRouter](https://openrouter.ai/) com **chave de API** ativa
- Uma pasta de projetos no host com **um subdiretório por projeto** e documentos nos formatos listados acima


## Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/VL-in/Assistente_de_laboratorio.git
cd Assistente_de_laboratorio
```

### 2. Criar a pasta de projetos

Crie (ou aponte para) a pasta no host que contém os documentos do laboratório. Cada projeto deve ser um subdiretório separado:

```
MinhaPastaLab/
├── Projeto_A/
│   ├── protocolo.docx
│   └── resultados.xlsx
└── Projeto_B/
    └── relatorio.pdf
```

### 3. Configurar variáveis de ambiente

Copie o arquivo de exemplo e preencha com suas credenciais:

```bash
cp .env.example .env
```

Edite o `.env` e preencha pelo menos as três variáveis obrigatórias:

| Variável | O que preencher |
|----------|-----------------|
| `OPENROUTER_API_KEY` | Sua chave em [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys) |
| `PROJETOS_HOST_DIR` | Caminho **absoluto** da pasta de projetos no host (ex.: `D:/Lab/Projetos`) |
| `LLM_MODEL` | Slug do modelo (padrão `openrouter/auto`)

> **Windows:** use barras normais no caminho, ex.: `C:/Users/Seu_Nome/Lab/Projetos`.

### 4. Subir a aplicação

```bash
docker compose up --build
```

O primeiro `build` baixa a imagem de embeddings (~1 GB) e pode levar alguns minutos. Aguarde a mensagem `Healthy` do serviço `embeddings` antes de usar.

Para rodar em segundo plano após o primeiro build:

```bash
docker compose up -d
```

Acesse no navegador: **http://localhost:8502**

> Após editar o `.env`, aplique as mudanças com `docker compose up -d --build`.

## Uso

### Primeira execução

1. **Desenvolvimento → Diagnóstico** — confirme que a chave de API aparece como configurada e teste **GET /v1/models**.
2. **Documentos** — informe a raiz dos projetos (ou use a do `.env`) e clique em **Escanear pastas**.
3. **Documentos → Indexação RAG** — clique em **Atualizar base agora** (obrigatório na primeira vez).
4. **Desenvolvimento → Busca híbrida** — valide a recuperação de trechos (BM25 + semântica).
5. **Conversa** — envie uma pergunta; o sistema roteia automaticamente para documentos, planilhas ou ML conforme disponível.
6. **ML tradicional** — carregue o dataset **AbRank (Kaggle)**, treine o modelo FLAML (`log_Aff`), salve o `.pkl` e teste a predição via chat.



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
