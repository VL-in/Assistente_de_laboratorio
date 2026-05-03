import json
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from openai import APIError
from pydantic import BaseModel, Field

from app.config import llm_model
from app.llm import async_openai_client

router = APIRouter(tags=["chat"])


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)


def _messages_payload(messages: list[ChatMessage]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


@router.post("/chat")
async def chat(body: ChatRequest) -> dict[str, str]:
    try:
        model = llm_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    client = async_openai_client()
    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=_messages_payload(body.messages),
            stream=False,
        )
    except APIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    choice = completion.choices[0].message
    text = choice.content or ""
    return {"message": text}


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest) -> StreamingResponse:
    try:
        model = llm_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    client = async_openai_client()

    async def event_generator():
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=_messages_payload(body.messages),
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield f"data: {json.dumps({'text': delta})}\n\n"
            yield "data: [DONE]\n\n"
        except APIError as exc:
            err = json.dumps({"error": str(exc)})
            yield f"data: {err}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
