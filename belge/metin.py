"""Reading documents and splitting them into overlapping chunks."""

import io
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class Page:
    source: str
    page: int
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    page: int
    text: str


def normalize(text: str) -> str:
    """Turkish-aware lowercase: Python's lower() maps 'I' to 'i', but in Turkish it is 'ı'."""
    return text.replace("İ", "i").replace("I", "ı").lower()


WORD = re.compile(r"[0-9a-zçğıöşüâîû]+")


def words(text: str) -> list[str]:
    return WORD.findall(normalize(text))


def stems(text: str, length: int = 5) -> list[str]:
    """Crude Turkish stemming: keep the first 5 letters, so 'garantisi', 'garantili'
    and 'garanti' all match. Turkish adds suffixes, so exact-word search misses a lot."""
    return [w[:length] for w in words(text)]


def read_pages(path: Path, data: bytes | None = None) -> list[Page]:
    """PDF pages keep their real page number; text and markdown files count as one page."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path if data is None else io.BytesIO(data))
        return [Page(path.name, i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    text = (data.decode("utf-8") if data is not None else path.read_text(encoding="utf-8"))
    return [Page(path.name, 1, text)]


def chunk_pages(pages: list[Page], size: int = 700, overlap: int = 150) -> list[Chunk]:
    """Split on paragraph boundaries; a chunk never crosses pages so citations stay exact."""
    chunks: list[Chunk] = []
    for page in pages:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page.text) if p.strip()]
        current = ""
        for paragraph in paragraphs:
            for piece in _split_long(paragraph, size):
                if current and len(current) + len(piece) + 1 > size:
                    chunks.append(_make(page, len(chunks), current))
                    current = current[-overlap:] if overlap else ""
                    # Start the overlap at a word boundary so no chunk begins mid-word.
                    current = current[current.find(" ") + 1 :] if " " in current else current
                current = f"{current} {piece}".strip()
        if current:
            chunks.append(_make(page, len(chunks), current))
    return chunks


def _split_long(paragraph: str, size: int) -> list[str]:
    if len(paragraph) <= size:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    out, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > size:
            out.append(current)
            current = ""
        current = f"{current} {sentence}".strip()
    if current:
        out.append(current)
    return out


def _make(page: Page, index: int, text: str) -> Chunk:
    return Chunk(id=f"{page.source}#{page.page}-{index}", source=page.source, page=page.page, text=text)
