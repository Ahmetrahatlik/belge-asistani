"""Clients for any OpenAI-compatible server (Ollama, LM Studio, OpenAI, Gemini)."""

from typing import Protocol

import httpx


class ChatModel(Protocol):
    def answer(self, system: str, user: str) -> str: ...


class OpenAICompatible:
    def __init__(self, base_url: str, api_key: str, chat_model: str, embed_model: str, timeout: float = 60.0):
        self.chat_model, self.embed_model = chat_model, embed_model
        self.client = httpx.Client(base_url=base_url.rstrip("/"), headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)

    def answer(self, system: str, user: str) -> str:
        response = self.client.post(
            "/chat/completions",
            json={"model": self.chat_model, "temperature": 0, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"] or ""

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.post("/embeddings", json={"model": self.embed_model, "input": texts})
        response.raise_for_status()
        return [item["embedding"] for item in sorted(response.json()["data"], key=lambda d: d["index"])]
