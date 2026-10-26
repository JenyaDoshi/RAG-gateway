# Document Q&A with RAG (foundation for an LLM Gateway)

Ask questions about your own `.txt` / `.md` / `.pdf` files. The app finds the most relevant
passages, shows them to you, and asks Gemini to answer using only those passages, with citations.

## Stack

| Stage | Choice | Why |
|---|---|---|
| Embeddings | `fastembed` + `BAAI/bge-small-en-v1.5` (local) | No API key or quota, small (~70 MB), no PyTorch |
| Vector index | `faiss-cpu` (exact search) | Simple, fast, and keeps "vectors" and "text" visibly separate |
| PDF text | `pypdf` | Pure Python, no system tools |
| Generation | `google-genai` -> `gemini-3.5-flash-lite` | Current stable model, free tier available |
| Config | `python-dotenv` | Key lives in `.env`, which git ignores |

## Project layout

```
rag-gateway/
├── main.py                  # entry point: ingest / search / ask / chat
├── inspect_chunks.py        # shows how documents are chunked
├── requirements.txt
├── .env.example             # copy to .env and add your key
├── .gitignore
├── data/docs/               # <-- PUT YOUR DOCUMENTS HERE (3 samples included)
├── rag/
│   ├── config.py            # settings from environment variables
│   ├── errors.py            # error types
│   ├── loader.py            # 1. document loader
│   ├── chunker.py           # 2. text chunker
│   ├── embedder.py          # 3. embedding generator
│   ├── vector_store.py      # 4a. vector index (FAISS) + saved text
│   ├── retriever.py         # 4b. retriever
│   ├── prompt.py            # 5. prompt builder
│   ├── llm.py               # 6. LLM client (Gemini behind a tiny interface)
│   └── pipeline.py          # wires the stages together
├── tests/                   # offline tests (fake embeddings, fake LLM)
└── storage/                 # created automatically: saved index + downloaded model
```

## Setup

Requires Python 3.10, 3.11, or 3.12 (check with `python --version`).

**Windows (PowerShell)**
```powershell
cd rag-gateway
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```
If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry.

**macOS / Linux**
```bash
cd rag-gateway
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### API key
1. Open https://aistudio.google.com/apikey and create a key (free tier needs no card).
2. Open `.env` in a text editor and set `GEMINI_API_KEY=` to your key, no quotes.
3. Never paste the key into code, chat, or git. `.env` is already in `.gitignore`.

## Run

```
python main.py ingest        # build the index (first run downloads the ~70 MB embedding model)
python main.py chat          # ask many questions (also the default: just `python main.py`)
python main.py ask "..."     # one question
python main.py search "..."  # retrieval only: shows chunks, needs no API key
```
Add `--full-chunks` to print full retrieved text and `--top-k 6` to retrieve more chunks.
Add or edit files in `data/docs/`, then run `python main.py ingest` again.

## Sample questions (with the included documents)

| Question | A good result |
|---|---|
| `Why do we split documents into chunks?` | Retrieved: `rag_basics.txt`. Answer gives three reasons (focused embeddings, limited context windows/cost, accuracy), cites `rag_basics.txt`. |
| `Which HTTP errors should trigger fallback to another provider?` | Retrieved: `llm_gateway_notes.txt`. Answer names 429, 503, 500 and timeouts, and says 401/403/400 should not trigger fallback. |
| `When does the supply drone land, and what happens if the wind is strong?` | Retrieved: `aurora_station_handbook.txt`. Answer: Thursday 14:20; postponed above 40 km/h; manager notifies the logistics office in Tromso before 12:00. |
| `What is the Wi-Fi password at Aurora Station?` (not in any document) | Answer starts with "I don't have enough information in the provided documents..." and shows `Sources cited: none`. |

## Test plan

Offline tests (no key, no download): `python -m unittest discover -s tests -v`

Manual checks with the real model and API:

| # | Check | How | Success looks like |
|---|---|---|---|
| 1 | Documents load | `python main.py ingest` | "Indexed 3 document(s) as 13 chunk(s)", no warnings. An empty/unsupported file gives a `warning:` line instead of a crash. |
| 2 | Chunking is sensible | `python inspect_chunks.py` | Chunks are mostly 400-800 characters, cut near sentence ends; each chunk starts with text from the end of the previous one (the overlap). |
| 3 | Embeddings generated | `ingest`, then look in `storage/index/` | `index.faiss` and `chunks.json` exist. |
| 4 | Retrieval works | `python main.py search "Why do we split documents into chunks?"` | Top result is from `rag_basics.txt`. |
| 5 | LLM uses context | `python main.py ask "When does the supply drone land?"` | Answer says 14:20 and cites `aurora_station_handbook.txt`. |
| 6 | Unanswerable question | `python main.py ask "What is the Wi-Fi password at Aurora Station?"` | Refusal sentence, no sources cited. |
| 7 | No rebuild per question | `python main.py chat`, ask three questions | No "building" message between questions; answers are quick after the first. |

If retrieval (check 4) returns the wrong chunk, fix retrieval before blaming the LLM: look at `--full-chunks`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `GEMINI_API_KEY is not set` | `.env` missing, in the wrong folder (must be next to `main.py`), or still has the placeholder. |
| `HTTP 429 ... quota` | Free-tier rate limit. Wait a minute, or set `GEMINI_MODEL` to another model. This is the situation the gateway will handle. |
| `HTTP 404 ... model not found` | Model names change. Check https://ai.google.dev/gemini-api/docs/models and update `GEMINI_MODEL` in `.env`. |
| `API key not valid` / 403 | Re-copy the key from AI Studio; no spaces or quotes. |
| `Could not load embedding model` | First run needs internet to download ~70 MB. Retry; check proxy/firewall. |
| `No index found` | Run `python main.py ingest` (or just run `ask`/`chat`, which builds it on the first run). |
| Index built with a different model | You changed `EMBEDDING_MODEL`. Re-run `python main.py ingest`. |
| `Documents folder not found` | Create `data/docs/` and add files. |
| `Skipped x.pdf: no text found` | Scanned/image PDF. This version cannot OCR; use a text-based PDF. |
| Answers ignore a document you just added | Run `python main.py ingest` again (the app warns when documents changed). |
| `pip install` fails on `faiss-cpu` | Use Python 3.10-3.12 in a fresh virtual environment. |
| `ModuleNotFoundError` | The virtual environment is not active, or `pip install -r requirements.txt` did not finish. |
