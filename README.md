# aml-transformer

A decoder-only GPT-style language model trained on Anti-Money Laundering (AML) regulatory documents from FATF. The project includes a custom BPE tokenizer written in C++, a full transformer architecture in PyTorch, an experiment tracking system backed by PostgreSQL, and a REST API for querying training results.

---

## Project Structure

```
aml-transformer/
├── docker-compose.yml          # PostgreSQL container setup
├── schema.sql                  # Database schema (runs + epochs tables)
├── configs.json                # Hyperparameter sweep configurations
├── src/
│   ├── cleaning_script.py      # Downloads and cleans FATF PDFs into a corpus
│   ├── main.py                 # Entry point: training sweep + visualization
│   ├── train.py                # Training loop with early stopping
│   ├── data/
│   │   ├── dataset.py          # PyTorch Dataset for sliding-window token sequences
│   │   └── prepare.py          # Tokenizer training and text encoding helpers
│   ├── model/
│   │   ├── attention.py        # SelfAttention + MultiHeadAttention
│   │   ├── transformer.py      # FeedForward + TransformerBlock
│   │   ├── gpt.py              # Full GPT model (token + positional embeddings)
│   │   └── bigram.py           # Baseline bigram language model
│   ├── tokenizer/
│   │   ├── aml_tokenizer.cpp   # BPE tokenizer implementation (C++)
│   │   ├── aml_tokenizer.hpp   # Header file
│   │   ├── bindings.cpp        # pybind11 bindings for Python use
│   │   └── CMakeLists.txt      # Build configuration
│   ├── tracker/
│   │   ├── db/
│   │   │   ├── connection.py   # psycopg connection pool
│   │   │   ├── runs.py         # DB queries for training runs
│   │   │   └── epochs.py       # DB queries for epoch metrics
│   │   └── api/
│   │       ├── main.py         # FastAPI app with lifespan pool management
│   │       ├── schemas.py      # Pydantic request/response models
│   │       └── routers/
│   │           ├── runs.py     # REST endpoints for runs
│   │           └── epochs.py   # REST endpoints for epochs
│   └── visualization/
│       ├── plot.py             # Seaborn plots for val loss and best epochs
│       ├── best_epochs.png     # Bar chart: best epoch per run
│       └── val_losses.png      # Line chart: val loss across epochs per run
└── tests/
```

---

## How It Works

### 1. Data Collection (`cleaning_script.py`)

Downloads 28 FATF Mutual Evaluation Reports and Follow-Up Reports as PDFs, extracts and cleans the text using PyMuPDF, deduplicates paragraphs, and writes everything to `aml_corpus.txt`. A `<|document|>` separator is inserted between documents.

### 2. Tokenizer (`src/tokenizer/`)

A Byte Pair Encoding (BPE) tokenizer implemented in C++ for performance, exposed to Python via **pybind11**. It starts with 256 base byte tokens and merges the most frequent pairs until reaching `target_vocab_size` (default: 2048). The trained vocabulary and merge rules are serialized to a JSON file and loaded on subsequent runs.

Build the tokenizer with CMake before running training:

```bash
cd src/tokenizer
mkdir build && cd build
cmake ..
make
```

### 3. Model Architecture (`src/model/`)

A decoder-only transformer with causal (masked) self-attention:

- **`attention.py`** — Scaled dot-product self-attention with a causal mask and dropout. `MultiHeadAttention` runs `n` heads in parallel, concatenates their outputs, and projects through W_O.
- **`transformer.py`** — A `TransformerBlock` applying Pre-LayerNorm, multi-head attention, and a position-wise feed-forward network (expansion factor 4, GELU activation), both with residual connections.
- **`gpt.py`** — The full `GPT` model: token embedding + positional embedding → N transformer blocks → LayerNorm → linear projection to vocab size. Supports autoregressive generation with context-window slicing.
- **`bigram.py`** — A simple bigram baseline for comparison.

### 4. Training (`train.py`, `main.py`)

`main.py` orchestrates a hyperparameter sweep over configurations defined in `configs.json`. For each config it:

1. Encodes the corpus once and creates a `AMLDataset` (sliding windows of `context_length=32` tokens).
2. Uses 50% of the dataset, split 90/10 into training and validation.
3. Calls `train()`, which runs up to 30 epochs with **AdamW**, logs every epoch's train/val loss to PostgreSQL, and applies early stopping with `patience=3`.

Device is auto-detected: MPS → CUDA → CPU.

### 5. Experiment Tracking (`src/tracker/`)

Every run and its per-epoch metrics are stored in PostgreSQL running in Docker.

**Database schema:**

```sql
runs   (id, d_model, lr, num_of_blocks, num_of_heads, created_at)
epochs (id, run_id → runs.id, epoch_num, train_loss, val_loss, created_at)
```

The **FastAPI** app (`src/tracker/api/`) exposes a REST API to query results:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/runs` | Create a new run |
| GET | `/runs` | List all runs |
| GET | `/runs/best` | Get the run with lowest val loss |
| GET | `/runs/{id}` | Get a specific run |
| POST | `/runs/{run_id}/epochs` | Log an epoch |
| GET | `/runs/{run_id}/epochs` | Get all epochs for a run |
| GET | `/epochs/best` | Get the best epoch per run |

### 6. Visualization (`src/visualization/plot.py`)

After the sweep completes, two plots are generated:

- `val_losses.png` — Validation loss per epoch for each run (line chart).
- `best_epochs.png` — Best validation loss achieved per run (bar chart).

---

## Setup & Usage

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- CMake + a C++17 compiler (for the tokenizer)
- pybind11

### 1. Start PostgreSQL

```bash
# Create a .env file with your DB credentials:
# DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

docker-compose up -d
```

### 2. Initialize the Database

```bash
psql -h 127.0.0.1 -U <DB_USER> -d <DB_NAME> -f schema.sql
```

### 3. Install Python Dependencies

```bash
pip install torch psycopg psycopg-pool psycopg2-binary fastapi uvicorn \
            pymupdf curl_cffi python-dotenv pandas matplotlib seaborn pybind11
```

### 4. Build the C++ Tokenizer

```bash
cd src/tokenizer && mkdir -p build && cd build
cmake .. && make
# This produces the aml_tokenizer Python module in the build directory
```

### 5. Collect the Corpus

```bash
python src/cleaning_script.py
# Produces aml_corpus.txt (~several MB of AML regulatory text)
```

### 6. Configure Hyperparameter Sweep

Edit `configs.json` to define your sweep, for example:

```json
[
  {"d_model": 64, "lr": 3e-4, "num_of_heads": 4, "num_of_blocks": 2},
  {"d_model": 128, "lr": 1e-4, "num_of_heads": 4, "num_of_blocks": 4}
]
```

### 7. Run Training

```bash
python src/main.py
```

This runs the full sweep and saves plots to `src/visualization/`.

### 8. Start the Tracking API (optional)

```bash
uvicorn src.tracker.api.main:app --reload
# API available at http://localhost:8000
```

---

## Key Design Decisions

- **C++ tokenizer** — BPE training and encoding are performance-critical; the C++ implementation with pybind11 bindings keeps Python ergonomics without the speed cost.
- **Pre-LayerNorm** — LayerNorm is applied before attention and FFN (rather than after), which stabilizes training in smaller models.
- **AdamW + early stopping** — Decoupled weight decay with patience-based early stopping prevents overfitting on the relatively small AML corpus.
- **Connection pooling** — `psycopg_pool` manages a pool of 1–5 connections; each training run borrows one connection for the duration of its logging.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Model training and inference |
| `pybind11` | C++ ↔ Python bindings for the tokenizer |
| `nlohmann/json` | JSON serialization in the C++ tokenizer |
| `psycopg` / `psycopg-pool` | PostgreSQL connectivity and pooling |
| `fastapi` / `uvicorn` | REST API for the experiment tracker |
| `pymupdf` | PDF text extraction |
| `curl_cffi` | Browser-impersonating HTTP requests for FATF PDFs |
| `pandas` / `seaborn` / `matplotlib` | Visualization |
| `python-dotenv` | Environment variable management |
