# Imagem única do MVP: UI Streamlit + módulos Python (pipeline, futuro RAG).
# Build a partir da raiz do repositório: docker compose build

FROM python:3.12-slim-bookworm

# LightGBM (dependência do flaml[automl]) precisa de OpenMP em runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash --uid 10001 appuser

WORKDIR /app

COPY apps/streamlit/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && python -c "import duckdb; print('duckdb', duckdb.__version__)" \
    && python -c "from flaml import AutoML; print('flaml', AutoML.__name__)"

COPY apps/streamlit/ /app/

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
