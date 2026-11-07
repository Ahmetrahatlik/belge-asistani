import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from belge.arama import Index
from belge.asistan import Assistant
from belge.llm import OpenAICompatible
from belge.metin import chunk_pages, read_pages

ALLOWED = {".pdf", ".txt", ".md"}
MAX_BYTES = 10 * 1024 * 1024


def make_assistant() -> Assistant:
    model = OpenAICompatible(
        os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        os.getenv("LLM_API_KEY", "ollama"),
        os.getenv("LLM_MODEL", "qwen2.5:7b"),
        os.getenv("EMBED_MODEL", "nomic-embed-text"),
    )
    use_vectors = os.getenv("USE_EMBEDDINGS", "1") == "1"
    return Assistant(Index(model if use_vectors else None), model)


app = FastAPI(title="Belge Asistanı", description="Ask questions about your PDFs and get answers with page citations.", version="1.0.0")
assistant: Assistant | None = None


def get_assistant() -> Assistant:
    global assistant
    if assistant is None:
        assistant = make_assistant()
    return assistant


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class SourceOut(BaseModel):
    number: int
    source: str
    page: int
    text: str


class AskOut(BaseModel):
    answer: str
    grounded: bool
    sources: list[SourceOut]


@app.post("/documents", status_code=201)
async def upload(file: UploadFile):
    name = Path(file.filename or "belge.txt").name
    if Path(name).suffix.lower() not in ALLOWED:
        raise HTTPException(415, f"Supported types: {', '.join(sorted(ALLOWED))}")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File is larger than 10 MB")
    try:
        chunks = chunk_pages(read_pages(Path(name), data))
    except Exception as exc:  # pypdf raises many error types for broken files
        raise HTTPException(422, f"Could not read the document: {exc}") from exc
    if not chunks:
        raise HTTPException(422, "No text found (scanned PDFs need OCR first)")
    index = get_assistant().index
    index.remove_source(name)  # uploading the same file again replaces it
    index.add(chunks)
    return {"source": name, "chunks": len(chunks)}


@app.get("/documents")
def documents():
    counts: dict[str, int] = {}
    for chunk in get_assistant().index.chunks:
        counts[chunk.source] = counts.get(chunk.source, 0) + 1
    return [{"source": s, "chunks": n} for s, n in sorted(counts.items())]


@app.post("/ask", response_model=AskOut)
def ask(data: AskIn):
    answer = get_assistant().ask(data.question)
    return AskOut(answer=answer.text, grounded=answer.grounded, sources=[SourceOut(**vars(s)) for s in answer.sources])
