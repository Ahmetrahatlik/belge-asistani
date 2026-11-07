"""Retrieval evaluation: does the right passage reach the top k for each question?

Runs without any model (keyword search only), so it also runs in CI:
    python degerlendirme.py
"""

from pathlib import Path

from belge.arama import Index
from belge.metin import chunk_pages, read_pages

# (question, a phrase that must appear in a retrieved chunk)
QUESTIONS = [
    ("Ürünü kaç gün içinde iade edebilirim?", "14 gün"),
    ("Kesilmiş kablo iade alınır mı?", "Kesilmiş kablo"),
    ("Matkabın garantisi kaç yıl?", "2 yıldır"),
    ("Pazar günü açık mısınız?", "Pazar günleri"),
    ("Veresiye limiti ne kadar?", "5.000 TL"),
    ("Teslimat ücreti ne kadar?", "150 TL"),
    ("Toplu alımda indirim var mı?", "yüzde 10"),
    ("Açılmış ürünü değiştirebilir miyim?", "30 gün"),
]


def evaluate(k: int = 3) -> float:
    index = Index()
    index.add(chunk_pages(read_pages(Path(__file__).parent / "ornek" / "magaza-politikasi.md"), size=300, overlap=60))
    hits = 0
    for question, phrase in QUESTIONS:
        found = any(phrase in h.chunk.text for h in index.search(question, k=k))
        hits += found
        print(f"{'✓' if found else '✗'} {question}")
    score = hits / len(QUESTIONS)
    print(f"\nhit@{k}: {hits}/{len(QUESTIONS)} = {score:.0%}")
    return score


if __name__ == "__main__":
    raise SystemExit(0 if evaluate() >= 0.75 else 1)
