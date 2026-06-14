# Imagem única do MVP: UI Streamlit + módulos Python (pipeline, futuro RAG).
# Build a partir da raiz do repositório: docker compose build

FROM python:3.12-slim-bookworm

# LightGBM (dependência do flaml[automl]) precisa de OpenMP em runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash --uid 10001 appuser

WORKDIR /app

COPY apps/streamlit/requirements-base.txt apps/streamlit/requirements.txt apps/streamlit/requirements-security.txt /app/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && python -c "import duckdb; print('duckdb', duckdb.__version__)" \
    && python -c "from flaml import AutoML; print('flaml', AutoML.__name__)"

# Modelos spaCy *small* (PT+EN) usados pelo Presidio (camada de PII). São os
# modelos leves — mantêm a imagem enxuta conforme decisão de projeto.
RUN python -m spacy download pt_core_news_sm \
    && python -m spacy download en_core_web_sm \
    && python -c "from presidio_analyzer import AnalyzerEngine; print('presidio ok')"

ENV HF_HOME=/data/ml/huggingface \
    TRANSFORMERS_CACHE=/data/ml/huggingface

# Pré-baixa ESM-2 8M para o cache (volume /data/ml em runtime).
RUN mkdir -p /data/ml/huggingface \
    && python -c "from transformers import AutoModel, AutoTokenizer; m='facebook/esm2_t6_8M_UR50D'; AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m); print('esm2', m, 'ok')"

COPY apps/streamlit/ /app/

# DeepEval: --no-deps evita upgrade posthog (conflito crewai/chromadb).
# Demais deps do deepeval instaladas explicitamente (posthog 5.x do chromadb permanece).
RUN pip install --no-cache-dir "deepeval>=4.0.0,<5.0.0" --no-deps \
    && pip install --no-cache-dir \
        nest-asyncio "pyfiglet" sentry-sdk "tabulate>=0.9.0,<0.10.0" wheel \
        pytest-xdist pytest-repeat pytest-rerunfailures pytest-asyncio

# Pastas persistentes (volumes nomeados ou bind no compose)
RUN mkdir -p /data/projetos /data/txtai /data/duckdb /data/ml /data/sqlite \
    && chown -R appuser:appuser /app /data

USER appuser

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    ASSISTENTE_PROJETOS_DIR=/data/projetos

EXPOSE 8502

# HTTP no endpoint interno do Streamlit (mais fiel do que só abrir TCP na porta).
HEALTHCHECK --interval=30s --timeout=8s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8502/_stcore/health', timeout=6).read(64)" || exit 1

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8502"]
