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
    return os.environ.get("LLM_API_KEY", "lm-studio")
