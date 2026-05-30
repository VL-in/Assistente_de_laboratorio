# Serviço de embeddings RAG (TEI + multilingual-e5-small).
# Build a partir da raiz: docker compose build embeddings

FROM ghcr.io/huggingface/text-embeddings-inference:cpu-1.6

ENV MODEL_ID=intfloat/multilingual-e5-small

# TEI escuta na porta 80 dentro do contêiner.
EXPOSE 80

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://127.0.0.1:80/health || exit 1

CMD ["--model-id", "intfloat/multilingual-e5-small", \
     "--port", "80", \
     "--max-batch-tokens", "8192", \
     "--max-client-batch-size", "32", \
     "--payload-limit", "6000000"]
