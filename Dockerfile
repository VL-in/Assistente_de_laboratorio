# Dockerfile para Hugging Face Spaces (Docker SDK).
#
# Diferenças em relação ao docker/streamlit/Dockerfile de produção:
#   - uid=1000  (exigido pelo HF Spaces — não root)
#   - Porta 7860 (padrão do HF Spaces)
#   - Modelos HF e E5 NÃO baixados no build: /data só existe em runtime
#     (bucket montado pelo HF). Na primeira inicialização o container baixa
#     os modelos para /data/huggingface e nas reinicializações reutiliza.
#   - deepeval e ferramentas de eval EXCLUÍDOS (não necessário em produção)
#
# Build para teste local (a partir da raiz do repositório):
#   docker build -f docker/hf/Dockerfile -t assistente-lab-hf .
#   docker run -p 7860:7860 --env-file .env.docker.example assistente-lab-hf
#
# Para o deploy real, este arquivo deve ser copiado como "Dockerfile"
# na raiz do repositório Git do Space. Ver DEPLOY_HUGGINGFACE.txt.

FROM python:3.12-slim-bookworm

# LightGBM/FLAML depende de OpenMP; curl usado no HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces obriga uid=1000.
RUN useradd -m -u 1000 user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# ── Dependências Python ────────────────────────────────────────────────────────
# Copiamos só os requirements antes do código para maximizar cache de layer:
# mudanças no código não invalidam a layer de instalação de pacotes.
COPY --chown=user \
    apps/streamlit/requirements-base.txt \
    apps/streamlit/requirements-security.txt \
    apps/streamlit/requirements.txt \
    ./

# torch CPU-only: ~800 MB mas sem GPU no plano gratuito do HF.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Modelos spaCy leves (PT + EN) para o Presidio (camada de PII).
# São baixados aqui porque não dependem de /data.
RUN python -m spacy download pt_core_news_sm \
    && python -m spacy download en_core_web_sm \
    && python -c "from presidio_analyzer import AnalyzerEngine; print('presidio ok')"

# ── Código da aplicação ────────────────────────────────────────────────────────
COPY --chown=user apps/streamlit/ $HOME/app/

# ── Variáveis de ambiente (não-secretas) ──────────────────────────────────────
# Todas as variáveis que contêm secrets (OPENROUTER_API_KEY, LANGFUSE_*, etc.)
# devem ser configuradas em Settings → Secrets no painel do HF Space — nunca aqui.
#
# /data é o ponto de montagem do Storage Bucket (persistent storage do HF).
# Somente disponível em runtime; o build-time não enxerga /data.
ENV HF_HOME=/data/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/data/huggingface \
    TRANSFORMERS_CACHE=/data/huggingface \
    KAGGLEHUB_CACHE=/data/kagglehub \
    ASSISTENTE_PROJETOS_DIR=/data/projetos \
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

USER user

# HF Spaces roteia tráfego externo para a porta declarada no README.md (7860).
EXPOSE 7860

# Aguarda até 90 s para o cold start: na primeira execução o E5 (~120 MB)
# e o ESM-2 (~30 MB) são baixados do HF Hub para /data/huggingface.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=4 \
    CMD curl -sf http://127.0.0.1:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=7860"]
