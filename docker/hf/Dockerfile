# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
#
# Dockerfile para Hugging Face Spaces (Docker SDK) — plano gratuito CPU Basic.
# Build a partir da raiz do repositório:
#   docker build -f docker/hf/Dockerfile -t assistente-lab-hf .
#
# Para o deploy real, copiar este arquivo como "Dockerfile" na raiz
# do repositório Git do Space. Ver DEPLOY_HUGGINGFACE.txt (Passo 7).

FROM python:3.12-slim-bookworm

# LightGBM/FLAML depende de libgomp; curl usado no HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# uid=1000 obrigatório pelo HF Spaces (não rodar como root).
RUN useradd -m -u 1000 user

# Segue o padrão do template HF: USER antes do pip e dos COPYs.
USER user

ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /home/user/app

# ── Dependências Python ────────────────────────────────────────────────────────
# Requirements copiados antes do código para aproveitar cache de layer Docker:
# mudanças no código não invalidam a layer de instalação de pacotes.
COPY --chown=user \
    apps/streamlit/requirements-base.txt \
    apps/streamlit/requirements-security.txt \
    apps/streamlit/requirements.txt \
    ./

# torch CPU-only (~800 MB); sem GPU no plano gratuito do HF Spaces.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Modelos spaCy leves (PT + EN) para o Presidio (camada de PII).
# São baixados no build porque não dependem de /data.
RUN python -m spacy download pt_core_news_sm \
    && python -m spacy download en_core_web_sm \
    && python -c "from presidio_analyzer import AnalyzerEngine; print('presidio ok')"

# ── Código da aplicação ────────────────────────────────────────────────────────
COPY --chown=user apps/streamlit/ /home/user/app/

# ── Variáveis de ambiente (não-secretas) ──────────────────────────────────────
# Secrets (OPENROUTER_API_KEY, LANGFUSE_*, KAGGLE_API_TOKEN) devem ser
# configurados em Settings → Secrets no painel do HF Space — nunca aqui.
#
# /data é o ponto de montagem do Storage Bucket (persistent storage do HF).
# Disponível APENAS em runtime; o build-time não enxerga /data.
# Na primeira inicialização o container baixa os modelos E5 e ESM-2 para
# /data/huggingface; nas reinicializações seguintes reutiliza o cache.
ENV HF_HOME=/data/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/data/huggingface \
    TRANSFORMERS_CACHE=/data/huggingface \
    KAGGLEHUB_CACHE=/data/kagglehub \
    ASSISTENTE_PROJETOS_DIR=/data/Projetos \
    ASSISTENTE_TXTAI_DIR=/data/txtai \
    ASSISTENTE_DUCKDB_DIR=/data/duckdb \
    ASSISTENTE_ML_DIR=/data/ml \
    LLM_BASE_URL=https://openrouter.ai/api/v1 \
    LLM_MODEL=openrouter/auto \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    LANGFUSE_BASE_URL=https://cloud.langfuse.com \
    LANGFUSE_TRACING_ENVIRONMENT=production \
    LANGFUSE_ENABLED=1 \
    RAG_HYBRID_ENABLED=1 \
    EMBEDDING_BATCH_SIZE=16

# HF Spaces roteia tráfego externo para a porta declarada em README.md (7860).
EXPOSE 7860

# start-period=90s: cold start inclui download do E5 (~120 MB) na 1ª execução.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=4 \
    CMD curl -sf http://127.0.0.1:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=7860"]
