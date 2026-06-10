# semantic-search-cli

Search your documents using meaning, not keywords.

Traditional search looks for exact words. This tool converts text into vectors (embeddings) and finds documents that are *semantically similar* to your query — even when they share no common words.

```
Query:  "who invented python?"
Result: "Python was created by Guido van Rossum and first released in 1991."
```

---

## How It Works

```
Index phase (once):
  your_docs.txt → embed each line → store.npz

Search phase (every query):
  "your question" → embed → compare against store.npz → top results
```

Each line of your text file becomes a searchable document. Embeddings are generated locally using `sentence-transformers` — no API key required.

---

## Setup

**Local (Python)**

```bash
git clone https://github.com/Raviteja0524/semantic-serach-cli.git
cd semantic-serach-cli
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Docker**

```bash
docker-compose run --rm app index --file data/sample_docs.txt
docker-compose run --rm app search "your query here"
```

---

## Usage

**Step 1 — Index a file**

```bash
python -m src.cli index --file data/sample_docs.txt
```

Reads each line, generates embeddings, saves everything to `data/store.npz`.

**Step 2 — Search**

```bash
python -m src.cli search "how do neural networks learn?"
```

```
Top 3 results for: "how do neural networks learn?"

------------------------------------------------------------
#1  Score: 0.5732
    Neural networks are computing systems loosely inspired by biological neural networks in animal brains.

#2  Score: 0.5593
    Deep learning uses multiple layers of neural networks to learn representations of data with multiple levels of abstraction.

#3  Score: 0.4480
    Machine learning is a subset of artificial intelligence where systems learn from data to improve their performance.
```

**Options**

```bash
# Return more results
python -m src.cli search "machine learning" --top-k 5

# Use a custom store location
python -m src.cli index --file my_notes.md --store data/notes.npz
python -m src.cli search "my query" --store data/notes.npz
```

---

## Project Structure

```
semantic-search-cli/
├── src/
│   ├── cli.py             # Click CLI: index, search commands
│   ├── embedder.py        # Loads sentence-transformers model, returns np.ndarray
│   ├── vector_store.py    # In-memory store: add, search, delete, save, load
│   └── similarity.py      # cosine_similarity, batch_cosine_similarity, l2_distance
├── data/
│   └── sample_docs.txt    # 30 sample facts to search over
├── tests/
│   └── test_similarity.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Concepts Covered

| Concept | Where it appears |
|---|---|
| Embeddings | `embedder.py` — text converted to `np.ndarray` |
| Cosine similarity | `similarity.py` — angle between vectors |
| Vector store | `vector_store.py` — parallel lists + NumPy matrix |
| Matrix dot product | `vector_store.search()` — one operation scores all docs |
| Persistence | `np.savez` / `np.load` — store survives between runs |
| CLI design | `cli.py` — Click group with index + search commands |
| Retry + backoff | `embedder.py` — decorator-based retry logic |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| NumPy | Vector math |
| sentence-transformers | Local embeddings (no API key needed) |
| Click | CLI interface |
| Docker | Containerize the app |

---

## Running Tests

```bash
pytest tests/
```
