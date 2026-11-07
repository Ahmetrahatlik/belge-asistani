"""Question answering over the indexed documents, always with sources."""

import re
from dataclasses import dataclass

from belge.arama import Hit, Index
from belge.llm import ChatModel

SYSTEM = """Sen bir belge asistanısın. Yalnız verilen kaynak parçalarına dayanarak Türkçe cevap ver.
Kurallar:
- Her bilgiden sonra kaynağını köşeli parantezle yaz, örn. [1] veya [2].
- Kaynaklarda cevap yoksa tam olarak şunu yaz: "Bu bilgi belgelerde yok."
- Tahmin yürütme, kaynakta olmayan sayı veya tarih yazma."""

NOT_FOUND = "Bu bilgi belgelerde yok."


@dataclass
class Source:
    number: int
    source: str
    page: int
    text: str


@dataclass
class Answer:
    text: str
    sources: list[Source]
    grounded: bool


def build_prompt(question: str, hits: list[Hit]) -> str:
    parts = [f"[{i}] ({h.chunk.source}, sayfa {h.chunk.page})\n{h.chunk.text}" for i, h in enumerate(hits, 1)]
    return "Kaynaklar:\n\n" + "\n\n".join(parts) + f"\n\nSoru: {question}"


class Assistant:
    def __init__(self, index: Index, model: ChatModel, k: int = 4):
        self.index, self.model, self.k = index, model, k

    def ask(self, question: str) -> Answer:
        hits = self.index.search(question, k=self.k)
        if not hits:
            # Nothing relevant was retrieved, so the model is not asked at all.
            return Answer(NOT_FOUND, [], grounded=False)
        text = self.model.answer(SYSTEM, build_prompt(question, hits)).strip()
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(hits)})
        sources = [Source(n, hits[n - 1].chunk.source, hits[n - 1].chunk.page, hits[n - 1].chunk.text) for n in cited]
        # An answer that cites nothing is not trusted, even if the model wrote something.
        grounded = bool(sources) and NOT_FOUND not in text
        if not grounded and NOT_FOUND not in text:
            text = f"{NOT_FOUND} (Model kaynak göstermeden cevap verdi, bu yüzden cevap gösterilmedi.)"
        return Answer(text, sources, grounded)
