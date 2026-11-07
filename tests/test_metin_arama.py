import hashlib
from pathlib import Path

from belge.arama import BM25, Index
from belge.metin import Chunk, Page, chunk_pages, normalize, read_pages, stems

SAMPLE = Path(__file__).parent.parent / "ornek" / "magaza-politikasi.md"


class HashEmbedder:
    """Deterministic stand-in for an embedding model: hashed bag of stems."""

    def embed(self, texts):
        vectors = []
        for text in texts:
            vector = [0.0] * 64
            for stem in stems(text):
                vector[int(hashlib.md5(stem.encode()).hexdigest(), 16) % 64] += 1
            vectors.append(vector)
        return vectors


def test_turkish_normalization():
    assert normalize("İADE ISPARTA Çekiç") == "iade ısparta çekiç"
    assert stems("Garantisi garantili GARANTİ") == ["garan", "garan", "garan"]


def test_chunks_stay_on_their_page_and_overlap_at_word_boundaries():
    pages = [Page("a.pdf", 1, "Bir. " * 200), Page("a.pdf", 2, "İki paragraf.\n\nÜç paragraf.")]
    chunks = chunk_pages(pages, size=120, overlap=30)

    assert {c.page for c in chunks} == {1, 2}
    assert all(len(c.text) <= 160 for c in chunks)
    assert all(not c.text.startswith(("ir.", "r.")) for c in chunks)
    assert [c.text for c in chunks if c.page == 2] == ["İki paragraf. Üç paragraf."]
    assert len({c.id for c in chunks}) == len(chunks)


def test_markdown_is_one_page():
    pages = read_pages(SAMPLE)
    assert len(pages) == 1 and pages[0].source == "magaza-politikasi.md"


def test_bm25_prefers_rare_terms_and_matches_suffixes():
    chunks = [Chunk("1", "x", 1, "garanti süresi iki yıldır"), Chunk("2", "x", 1, "mağaza mağaza saatleri"), Chunk("3", "x", 1, "iade on dört gün")]
    scores = BM25(chunks).scores("Garantisi kaç yıl?")
    assert scores.index(max(scores)) == 0
    assert scores[1] == 0


def test_hybrid_search_ranks_the_right_passage_first():
    index = Index(HashEmbedder())
    index.add(chunk_pages(read_pages(SAMPLE), size=300, overlap=60))
    hit = index.search("Veresiye limiti ne kadar?", k=3)[0]
    assert "5.000 TL" in hit.chunk.text
    assert hit.keyword_rank == 1 and hit.vector_rank is not None


def test_search_with_nothing_relevant_returns_nothing():
    index = Index()
    index.add([Chunk("1", "x", 1, "çalışma saatleri hafta içi")])
    assert index.search("uzay mekiği yakıtı") == []
    assert Index().search("herhangi") == []


def test_reuploading_a_source_replaces_it():
    index = Index(HashEmbedder())
    index.add([Chunk("a1", "a.md", 1, "eski metin"), Chunk("b1", "b.md", 1, "diğer belge")])
    assert index.remove_source("a.md") == 1
    index.add([Chunk("a2", "a.md", 1, "yeni metin")])
    assert [c.id for c in index.chunks] == ["b1", "a2"]
    assert index.vectors.shape[0] == 2
