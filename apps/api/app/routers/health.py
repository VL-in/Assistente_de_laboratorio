import os

from fastapi import APIRouter, HTTPException

from app.llm import async_openai_client

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/llm")
async def health_llm() -> dict[str, str]:
    """Verifica se o LM Studio (ou outro servidor OpenAI-compatible) está acessível."""
    client = async_openai_client()
    try:
        await client.models.list()
    except Exception as exc:  # noqa: BLE001 — diagnóstico operacional
        raise HTTPException(
            status_code=503,
            detail=f"LLM indisponível: {exc!s}. Confira se o servidor local do LM Studio está ligado.",
        ) from exc
    model_opt = os.environ.get("LLM_MODEL", "").strip()
    return {
        "status": "ok",
        "llm": "reachable",
        "model_configured": model_opt or "(defina LLM_MODEL no .env)",
    }
