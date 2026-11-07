from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from belge import api
from belge.arama import Index
from belge.asistan import NOT_FOUND, Assistant, build_prompt
from belge.metin import Chunk

SAMPLE = Path(__file__).parent.parent / "ornek" / "magaza-politikasi.md"


class FakeModel:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def answer(self, system, user):
        self.prompts.append(user)
        return self.reply


def small_index():
    index = Index()
    index.add([Chunk("1", "politika.pdf", 3, "Ürünler 14 gün içinde iade edilebilir."), Chunk("2", "politika.pdf", 4, "Garanti 2 yıldır.")])
    return index


def test_answer_keeps_only_cited_sources():
    model = FakeModel("İade süresi 14 gündür [1].")
    answer = Assistant(small_index(), model).ask("İade süresi kaç gün?")

    assert answer.grounded
    assert [(s.number, s.source, s.page) for s in answer.sources] == [(1, "politika.pdf", 3)]
    assert "[1] (politika.pdf, sayfa 3)" in model.prompts[0]


def test_uncited_answer_is_not_trusted():
    answer = Assistant(small_index(), FakeModel("İade süresi 30 gündür.")).ask("İade süresi kaç gün?")
    assert not answer.grounded
    assert answer.text.startswith(NOT_FOUND)


def test_model_is_not_called_when_nothing_is_found():
    model = FakeModel("uydurma [1]")
    answer = Assistant(small_index(), model).ask("Uzay mekiği yakıtı nedir?")
    assert answer.text == NOT_FOUND and model.prompts == []


def test_citation_numbers_outside_the_sources_are_ignored():
    answer = Assistant(small_index(), FakeModel("Bilgi [7] ve [1].")).ask("iade gün")
    assert [s.number for s in answer.sources] == [1]


def test_prompt_numbers_sources_in_order():
    hits = small_index().search("iade garanti", k=2)
    prompt = build_prompt("Soru?", hits)
    assert prompt.index("[1]") < prompt.index("[2]") and prompt.endswith("Soru: Soru?")


@pytest.fixture()
def client(monkeypatch):
    model = FakeModel("Kesilmiş kablo iade alınmaz [1].")
    monkeypatch.setattr(api, "assistant", Assistant(Index(), model))
    return TestClient(api.app)


def test_upload_list_ask(client):
    with SAMPLE.open("rb") as f:
        response = client.post("/documents", files={"file": ("politika.md", f, "text/markdown")})
    assert response.status_code == 201 and response.json()["chunks"] > 0

    with SAMPLE.open("rb") as f:
        client.post("/documents", files={"file": ("politika.md", f, "text/markdown")})
    assert len(client.get("/documents").json()) == 1

    answer = client.post("/ask", json={"question": "Kesilmiş kablo iade alınır mı?"}).json()
    assert answer["grounded"] is True
    assert answer["sources"][0]["source"] == "politika.md"
    assert "Kesilmiş kablo" in answer["sources"][0]["text"]


def test_rejects_unsupported_and_empty_files(client):
    assert client.post("/documents", files={"file": ("x.exe", b"MZ", "application/octet-stream")}).status_code == 415
    assert client.post("/documents", files={"file": ("bos.txt", b"   ", "text/plain")}).status_code == 422
    assert client.post("/ask", json={"question": "a"}).status_code == 422
