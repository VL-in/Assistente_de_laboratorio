from fastapi import FastAPI

from app.routers import chat, health

app = FastAPI(title="Assistente Lab API", version="0.1.0")

app.include_router(health.router)
app.include_router(chat.router)
