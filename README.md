# Belge Asistanı

[![CI](https://github.com/Ahmetrahatlik/belge-asistani/actions/workflows/ci.yml/badge.svg)](https://github.com/Ahmetrahatlik/belge-asistani/actions/workflows/ci.yml)

PDF, TXT ve Markdown belgelerine soru sorup **sayfa numarasıyla kaynak gösteren** cevaplar alan bir RAG (retrieval-augmented generation) servisi. FastAPI ile yazıldı. Türkçe belgeler için ayarlandı.

Örnek: bir mağazanın iade politikasını yükleyip *"Kesilmiş kablo iade alınır mı?"* diye sorarsınız. Cevap şöyle gelir: *"Hayır, metreyle kesilen kablolar iade alınmaz [1]."* Altında da `[1] magaza-politikasi.md, sayfa 1` yazar ve ilgili paragraf gösterilir.

## Neler yapıyor

- **Belge okuma ve parçalama**: PDF'ler sayfa sayfa okunur. Metin paragraf sınırlarından ~700 karakterlik parçalara bölünür, parçalar arasında 150 karakter örtüşme bırakılır. Bir parça hiçbir zaman iki sayfaya yayılmaz, bu yüzden kaynak gösterilen sayfa her zaman doğru olur.
- **Hibrit arama**: BM25 anahtar kelime araması ile embedding (vektör) benzerliği birlikte çalışır. İki sıralama *reciprocal rank fusion* ile birleştirilir. Embedding modeli yoksa sistem yalnızca BM25 ile de çalışır.
- **Türkçeye uygun arama**:
  - `İ/I` harfleri doğru küçültülür. Python'un `lower()` fonksiyonu `I` harfini `i` yapar, Türkçede doğrusu `ı`.
  - Basit bir kök bulma uygulanır: kelimenin ilk 5 harfi alınır. Böylece *garantisi*, *garantili* ve *garanti* aynı terim sayılır.
- **Uydurma cevaba karşı korumalar**:
  - Model yalnızca verilen parçalardan cevap vermek ve her bilgiye `[n]` ile kaynak göstermek zorundadır.
  - Arama bir şey bulamazsa model hiç çağrılmaz ve cevap "Bu bilgi belgelerde yok." olur.
  - Model kaynak göstermeden cevap verirse o cevap kullanıcıya gösterilmez.
  - Var olmayan kaynak numaraları (ör. 3 kaynak varken `[7]`) atılır.
- **Ölçülen arama kalitesi**: `degerlendirme.py` 8 soruluk bir test setinde doğru paragrafın ilk 3 sonuçta olup olmadığını ölçer (hit@3). Şu anki sonuç **8/8 (%100)**. Bu ölçüm CI'da her push'ta çalışır ve %75'in altına düşerse build kırılır.
- **Modelden bağımsız**: OpenAI uyumlu her API ile çalışır: yerel Ollama, OpenAI veya Gemini'nin OpenAI uyumlu ucu.
- **Testler**: 14 pytest testi var. Sahte model ve sahte embedder kullandıkları için internet ve API anahtarı gerektirmezler.

## Kurulum

```bash
python -m venv .venv
.venv/Scripts/activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
python degerlendirme.py
```

### Yerel modelle çalıştırma (Ollama, ücretsiz)

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
uvicorn belge.api:app --reload
```

Swagger arayüzü için `http://localhost:8000/docs` adresini açın.

### Ayarlar (ortam değişkenleri)

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI uyumlu API adresi |
| `LLM_API_KEY` | `ollama` | API anahtarı (Ollama için herhangi bir değer olur) |
| `LLM_MODEL` | `qwen2.5:7b` | Cevap üreten model |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding modeli |
| `USE_EMBEDDINGS` | `1` | `0` verilirse yalnızca BM25 kullanılır |

## API

| Uç | Ne yapar |
|---|---|
| `POST /documents` | Belge yükler (`.pdf`, `.txt`, `.md`, en fazla 10 MB). Aynı isimle tekrar yüklenen belge eskisinin yerine geçer. |
| `GET /documents` | Yüklü belgeleri ve kaç parçaya bölündüklerini listeler |
| `POST /ask` | `{"question": "..."}` gönderilir. Dönen cevapta `answer`, `grounded` (kaynaklı mı) ve `sources` (belge, sayfa, metin) alanları olur. |

```bash
curl -F "file=@ornek/magaza-politikasi.md" http://localhost:8000/documents
curl -H "Content-Type: application/json" -d '{"question":"Veresiye limiti ne kadar?"}' http://localhost:8000/ask
```

Hata durumları:

- Desteklenmeyen dosya türü `415` döner.
- 10 MB'tan büyük dosya `413` döner.
- Okunamayan veya metin içermeyen dosya `422` döner. Taranmış PDF'lerde metin olmaz, önce OCR gerekir.

## Proje yapısı

```
belge/metin.py      belge okuma, Türkçe normalleştirme, parçalama
belge/arama.py      BM25, vektör araması, RRF birleştirme
belge/asistan.py    prompt, kaynak ayıklama, uydurma cevap kontrolü
belge/llm.py        OpenAI uyumlu sohbet ve embedding istemcisi
belge/api.py        FastAPI uçları
degerlendirme.py    arama kalitesi ölçümü (hit@3)
ornek/              uydurma bir mağaza politikası (örnek belge)
```

## Sınırlar

- İndeks bellekte tutulur, sunucu yeniden başlayınca belgeler silinir. Kalıcılık için sonraki adım pgvector olurdu.
- Kök bulma basit tutuldu (ilk 5 harf). Zemberek gibi gerçek bir morfoloji kütüphanesi daha isabetli olur.
- Canlı demo yok, çünkü cevap üretmek için çalışan bir dil modeli gerekiyor. Testler ve değerlendirme modelsiz çalışır.

---

## English summary

A RAG service that answers questions about uploaded PDF/TXT/Markdown documents and cites the exact page for every claim. Built with FastAPI and tuned for Turkish text.

- Page-aware chunking with overlap; a chunk never crosses a page, so citations stay exact.
- Hybrid retrieval: BM25 and embedding cosine similarity merged with reciprocal rank fusion. It falls back to BM25 alone when no embedding model is configured.
- Turkish-aware lowercasing (`I` becomes `ı`) and prefix stemming, so different suffixes of a word still match.
- Grounding guards:
  - The model is not called when nothing is retrieved.
  - Uncited answers are withheld.
  - Out-of-range citation numbers are dropped.
- Retrieval eval: hit@3 is 8/8 on the sample set. It runs in CI and fails the build below 75%.
- Works with any OpenAI-compatible API (Ollama, OpenAI, Gemini).
- 14 tests use fake models, so they need no network or API key.

MIT License.
