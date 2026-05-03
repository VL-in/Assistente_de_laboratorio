from openai import AsyncOpenAI

from app.config import llm_api_key, llm_base_url


def async_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=llm_base_url(), api_key=llm_api_key())
