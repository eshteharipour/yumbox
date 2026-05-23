# Yumbox 🍱

> **Ya**dgiri **M**achine Tool**box** — A very yummy project!
> (Yadgiri is Persian for "learning".)

**What is Yumbox?**

Yumbox is a Machine Learning reusable toolbox to help bootstrap ML projects as quickly as possible! It follows the best practices I have accumulated over the years, from ensuring reproducibility and tracking experiments to following DRY (Don't Repeat Yourself) and VC (Version Control) on code/data principles. This project also specially focuses on Semantic Similarity, Classification, Information Retrieval, and the intersection of the three: Entity Resolution (and Entity Matching).

**Why Yumbox came to be?**

I found myself reusing the same classes and functions in my projects over the years, from my university days (pre-LLLM chatbot era) to this day, so I decided to create this project to gather all these proven functionalities, provide standard interfaces for them, and make them freely available to help everyone facilitate fast prototyping on ML projects.

**How Yumbox came to be!**

I gradually started refactoring the classes and functions from the projects I was working on in the industry and the university and I added them to Yumbox. I still continue to do so, when I find a highly valuable functionality that can be reused and shared with other projects. Since the things covered in this project have vastly different scopes, I follow a dynamic import paradigm so you only have to install the dependencies required for the functionality needed. I will later make these dynamic imports, lazy imports, just as popular libraries like HuggingFace's `transformers` do.

---

## 📚 Modules Overview

Below you'll find detailed guides for every module in Yumbox:

| Module | Purpose |
|--------|---------|
| 🔄 [Cache](cache.md) | Caching decorators & storage backends (pickle, LMDB, Redis, FAISS) |
| ⚙️ [Config](config.md) | Global config (BFG) + production-ready logger setup |
| 📦 [Data](data.md) | Flexible Datasets, Samplers & training utilities for PyTorch |
| 🏭 [Factory](factory.md) | FAISS index builders, PCA, clustering, similarity computation |
| 📊 [Metrics](metrics.md) | Classification, retrieval metrics + MLflow-ready plotting |
| 🧪 [MLflow](mlflow.md) | Experiment tracking, checkpoint management, multi-run analysis |
| ✂️ [NLP](nlp.md) | Multilingual (EN/FA) text preprocessing, tokenization, analysis |
| 🕵️ [Parse](parse.md) | Runtime dependency analysis & import tracing |
| 🕷️ [Scraper](scraper.md) | HTML parsing, text extraction, parallel image downloading |
| 🛠️ [Scripts](scripts.md) | CLI tools for MLflow analysis & checkpoint cleanup |
| 🔍 [Vectors](vectors.md) | Vector search, feature fusion, array utilities |

---

## 🚀 Quick Start Examples

### Cache results with one line
```python
from yumbox.cache import cache

@cache  # Auto-saves to BFG["cache_dir"]/{func_name}.pkl
def expensive_feature_extraction(data):
    return heavy_computation(data)

# First call: runs computation, saves result
# Subsequent calls: loads from cache instantly ✨
```

### Setup logging + config in 3 lines
```python
from yumbox.config import BFG, setup_logger

BFG["cache_dir"] = "~/.cache/yumbox"  # Auto-creates directory

logger = setup_logger(
    name="my_project",
    path="./logs",           # Save logs to file
    capture_libs=["torch"],  # Also log PyTorch messages
    suppress_libs=["httpcore"]  # Silence noisy dependencies
)

logger.info("Ready to yum! 🍱")
```

### Build a FAISS index for cosine search
```python
import numpy as np
from yumbox.factory import build_index, pca_faiss
from yumbox.vectors import normalize_vector

# Your embeddings
embeddings = np.random.randn(10000, 768).astype(np.float32)

# Option 1: Quick flat index (exact search)
index = build_index(normalize_vector(embeddings))  # Inner product = cosine for unit vectors

# Option 2: PCA reduction + HNSW for speed
embeddings_reduced = pca_faiss(embeddings, n_components=128)
index = factory.FaissIndexBuilder().build_hnsw_index(
    normalize_vector(embeddings_reduced), 
    M=32, efConstruction=200
)

# Search
distances, indices = index.search(query_vectors, k=10)
```
