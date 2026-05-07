import os

from dotenv import load_dotenv

load_dotenv()


def llm_base_url() -> str:
    return os.environ.get("LLM_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")


def llm_model() -> str:
    model = os.environ.get("LLM_MODEL", "").strip()
    if not model:
        raise RuntimeError(
            "Defina LLM_MODEL no .env (nome do modelo no LM Studio). Copie de .env.example."
        )
    return model


def llm_api_key() -> str:
    """Compatível com `.env` da API (`LLM_API_KEY`) e com o Compose/README (`OPENAI_API_KEY`)."""
    for env_name in ("LLM_API_KEY", "OPENAI_API_KEY"):
        v = os.environ.get(env_name, "").strip()
        if v:
            return v
    return "lm-studio"
