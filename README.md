<div align="center">

# Yumbox 🍱

**A reusable machine learning toolbox for fast prototyping, experiment tracking, and vector search.**

[![PyPI Version](https://img.shields.io/pypi/v/yumbox.svg)](https://pypi.org/project/yumbox/)
[![Python Versions](https://img.shields.io/pypi/pyversions/yumbox.svg)](https://pypi.org/project/yumbox/)
[![Documentation Status](https://readthedocs.org/projects/yumbox/badge/?version=latest)](https://yumbox.readthedocs.io/en/latest/?badge=latest)

> **Ya**dgiri **M**achine Tool**box** — A very yummy project!<br>
> *(Yadgiri is Persian for "learning".)*

📚 **[Read the Full Documentation here!](https://yumbox.readthedocs.io/)**

</div>

---

**What is Yumbox?**

Yumbox is a Machine Learning reusable toolbox to help bootstrap ML projects as quickly as possible! It follows the best practices I have accumulated over the years, from ensuring reproducibility and tracking experiments to following DRY (Don't Repeat Yourself) and VC (Version Control) on code/data principles. This project also specially focuses on Semantic Similarity, Classification, Information Retrieval, and the intersection of the three: Entity Resolution (and Entity Matching).

**Why Yumbox came to be?**

I found myself reusing the same classes and functions in my projects over the years, from my university days (pre-LLLM chatbot era) to this day, so I decided to create this project to gather all these proven functionalities, provide standard interfaces for them, and make them freely available to help everyone facilitate fast prototyping on ML projects.

**How Yumbox came to be!**

I gradually started refactoring the classes and functions from the projects I was working on in the industry and the university and I added them to Yumbox. I still continue to do so, when I find a highly valuable functionality that can be reused and shared with other projects. Since the things covered in this project have vastly different scopes, I follow a dynamic import paradigm so you only have to install the dependencies required for the functionality needed. I will later make these dynamic imports lazy imports, just as popular libraries like HuggingFace's `transformers` do. 

You can install this project using:
```bash
pip install yumbox
pip install yumbox[faiss]  # Installs FAISS dependency
```

> ⚠️ **Heads up**: Yumbox is currently in the pre-release stage. I heavily use this repo in my projects and I try to not make breaking changes, but still, you should expect things to break, APIs to change, and documentation to play catch-up. Not all functionalities are fully tested yet since I try to cover all use-cases.

---

## 🗂️ Project Structure

```text
yumbox/
├── cache/      # 🔄 Caching decorators & storage backends (pickle, LMDB, Redis, FAISS)
├── config/     # ⚙️ Global config (BFG) + production-ready logger setup
├── data/       # 📦 Flexible Datasets, Samplers & training utilities for PyTorch
├── factory/    # 🏭 FAISS index builders, PCA, clustering, similarity computation
├── metrics/    # 📊 Classification, retrieval metrics + MLflow-ready plotting
├── mlflow/     # 🧪 Experiment tracking, checkpoint management, multi-run analysis
├── nlp/        # ✂️ Multilingual (EN/FA) text preprocessing, tokenization, analysis
├── parse/      # 🕵️ Runtime dependency analysis & import tracing
├── scraper/    # 🕷️ HTML parsing, text extraction, parallel image downloading
├── scripts/    # 🛠️ CLI tools for MLflow analysis & checkpoint cleanup
└── vectors/    # 🔍 Vector search, feature fusion, array utilities
```

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

### Preprocess multilingual text
```python
from yumbox.nlp import Preprocessor

# English + Persian text cleaning
preproc = Preprocessor(
    remove_punctuation=True,
    remove_en_stopwords=True,
    remove_fa_stopwords=True,
    normalize_fa_chars=True,  # Arabic→Farsi character mapping
    do_hard_limit=True,       # Filter words by length
    low_hard_limit=3,
    high_hard_limit=26,
)

clean_text = preproc("This is a test! این یک تست است. 🎉")
# → "this test تست"
```

### Analyze MLflow experiments from CLI
```bash
# Compare validation loss across experiments
metrics-cli compare-experiments \
  --storage-path ./mlruns \
  --experiment-names baseline_v1 improved_model \
  --metric val_loss \
  --mode epoch \
  --output-file loss_comparison.png

# Find best-performing runs
metrics-cli best-metrics \
  --storage-path ./mlruns \
  --experiment-names "exp_.*_bert" \
  --metrics f1_score inference_time \
  --min-or-max max min \
  --output-csv best_runs.csv
```

---

## 🧩 Module Highlights

### `cache/` — Caching That Just Works
- **Decorators**: `@cache`, `@np_cache`, `@timed_cache`, `@async_cache`, `@retry`
- **Backends**: pickle, NumPy `.npz`, HDF5, LMDB, Redis, FAISS indices, safetensors
- **Smart features**: kwargs-based keys, hash-based keys, lazy loading, offset tracking for pagination

👉 [Full cache documentation](https://yumbox.readthedocs.io/en/latest/cache/)

### `config/` — Logging & Configuration
- **`BFG`**: Global config store (your own configs + `cache_dir` for enabling/disabling cache functionalities of yumbox)
- **`setup_logger()`**: Color-coded console output, file logging, library capture/suppression, print redirection
- **Helpers**: `execution_wrapper` (auto-timing + error logging), `main_run`, `log_df_info`

👉 [Full config documentation](https://yumbox.readthedocs.io/en/latest/config/)

### `data/` — Datasets & Training Utilities
- **FlexibleDataset**: Mode-aware (`text`/`image`/`text_image`) datasets with transform pipelines
- **Specialized datasets**: `WebImgDataset`, `TextDataset`, `TFDocumentDataset`, `ZeroshotDataset`
- **Samplers**: `ContrastiveSampler`, `TripletSampler`, `SiameseSampler`, `ClusterSampler` for advanced training strategies

👉 [Full data documentation](https://yumbox.readthedocs.io/en/latest/data/)

### `factory/` — FAISS & Vector Processing
- **Index builders**: Flat, IVF, HNSW, PQ, IVFPQ with GPU support
- **Utilities**: `pca_faiss`, `kmeans_faiss`, `self_similarity`
- **`FaissIndexBuilder`**: Unified API for production-ready index construction

👉 [Full factory documentation](https://yumbox.readthedocs.io/en/latest/factory/)

### `metrics/` — Evaluation Made Simple
- **Classification**: `classification_scores`, `extended_classification_scores` with FPR/FNR
- **Retrieval**: `mean_ir_scores` for P@k, MAP, MRR, NDCG
- **Visualization**: PR/ROC curves auto-logged to MLflow
- **Utilities**: `AverageMeter` for training loops, `cosine_sim` for tensor similarity

👉 [Full metrics documentation](https://yumbox.readthedocs.io/en/latest/metrics/)

### `mlflow/` — Experiment Tracking & Checkpoint Management
- **Logging**: `log_config`, `log_scores_dict`, recursive OmegaConf param logging
- **Analysis**: `process_experiment_metrics`, `plot_metric_across_experiments`, `find_best_metrics`
- **Checkpoint helpers**: Intelligent cleanup based on metric performance (`manage-checkpoints` CLI)
- **Export**: `export_mlflow_data_with_flattening` for CSV analysis

👉 [Full mlflow documentation](https://yumbox.readthedocs.io/en/latest/mlflow/)

### `nlp/` — Multilingual Text Preprocessing
- **`Preprocessor`**: Flag-driven pipeline for English/Persian text cleaning
- **Unicode utilities**: Character normalization, accent stripping, CJK removal, range-based filtering
- **Analysis**: `MapRed` for frequency counting, `defaultname` for alias resolution/entity clustering

👉 [Full nlp documentation](https://yumbox.readthedocs.io/en/latest/nlp/)

### `parse/` — Dependency Analysis
- **`analyze_dependencies()`**: Runtime import tracing vs. installed/declared packages
- **Use cases**: Generate minimal `requirements.txt`, audit Docker images, CI/CD dependency checks

👉 [Full parse documentation](https://yumbox.readthedocs.io/en/latest/parse/)

### `scraper/` — Web Scraping Utilities
- **HTML parsing**: `parse_html`, `html_to_text`, enhanced `MySelector` with auto-decoding
- **`ImageDownloader`**: Parallel downloading with PIL corruption checks, custom DNS resolution, quality filtering

👉 [Full scraper documentation](https://yumbox.readthedocs.io/en/latest/scraper/)

### `scripts/` — CLI Tools
- **`metrics-cli`**: Subcommands for experiment analysis, comparison, checkpoint management, best-run discovery
- **Help system**: `metrics-cli help patterns` for usage examples, troubleshooting guides

👉 [Full scripts documentation](https://yumbox.readthedocs.io/en/latest/scripts/)

### `vectors/` — Vector Operations & Feature Fusion
- **Search**: `topk()` with parallel FAISS querying, `nested_topk()` for hierarchical retrieval
- **Normalization**: `normalize_vector()` for NumPy/Torch tensors
- **Feature fusion**: `cat_feats`, `mult_feats`, `sum_feats`, `diff_feats` with missing-value handling
- **Utilities**: `reconstruct_original_index()` for restoring filtered arrays

👉 [Full vectors documentation](https://yumbox.readthedocs.io/en/latest/vectors/)

---

## 🔁 Common Workflows

### 🔄 Incremental Embedding Pipeline
```python
from yumbox.cache import np_cache_kwargs_hash  # or @cache for pickle-based caching
from yumbox.factory import build_index
from yumbox.vectors import normalize_vector
import numpy as np

# 1. Setup cache directory (used by all @*cache decorators)
from yumbox.config import BFG
BFG["cache_dir"] = "~/.cache/my_project"  # Auto-creates if needed

# 2. Decorate your heavy inference function — caching is automatic!
@np_cache_kwargs_hash  # Caches dict[str, np.ndarray] with hash-based keys
def compute_embedding(image_path: str, model_name: str = "default", cache_kwargs: list = None, **kwargs):
    """
    Heavy embedding computation — automatically cached by (image_path, model_name).
    Returns dict for np_cache compatibility.
    """
    # This only runs if cache miss!
    img = load_and_preprocess(image_path)  # Your image loading logic
    emb = get_model(model_name)(img)       # Your model inference
    return {"embedding": emb.squeeze().cpu().numpy()}  # Must return dict for np_cache

# 3. Process your dataset — no manual cache management needed!
embeddings = {}
for hash_key, img_path in zip(df["hash"], df["path"]):
    # Automatic cache check: if (img_path, model_name) was computed before, loads instantly
    result = compute_embedding(
        img_path, 
        model_name="vit_base",
        cache_kwargs=["image_path", "model_name"]  # These kwargs affect cache key
    )
    embeddings[hash_key] = result["embedding"]

# 4. Build search index from cached embeddings
embedding_matrix = np.array(list(embeddings.values()))
index = build_index(normalize_vector(embedding_matrix))  # Cosine similarity via IP
faiss.write_index(index, "search_index.faiss")

# ✨ Next run: All previously computed embeddings load from cache in milliseconds!
```

### 📊 End-to-End Experiment Tracking
```python
from yumbox.config import BFG, setup_logger, execution_wrapper
from yumbox.mlflow import log_config, log_scores_dict
import mlflow

# 1. Global setup
BFG["cache_dir"] = "./cache"
logger = setup_logger("my_experiment", path="./logs")

# 2. Wrap main logic for auto-timing + error logging
@execution_wrapper
def train(cfg):
    with mlflow.start_run(run_name=cfg.run_name):
        # Log config recursively
        log_config(cfg)
        
        # Training loop
        for epoch in range(cfg.epochs):
            # ... train ...
            
            # Log metrics with step
            val_metrics = evaluate(model, val_loader)
            log_scores_dict(val_metrics, name="val", step=epoch)
            
            # Save checkpoint with metric-based naming
            if val_metrics["f1"] > best_f1:
                ckpt_path = f"./checkpoints/epoch{epoch}_f1{val_metrics['f1']:.3f}.pt"
                torch.save(model.state_dict(), ckpt_path)
                mlflow.log_param("best_checkpoint", ckpt_path)
                best_f1 = val_metrics["f1"]

# 3. Run with config
if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("configs/exp.yaml")
    train(cfg)
```

### 🌍 Multilingual Text Processing Pipeline
```python
from yumbox.nlp import Preprocessor
import pandas as pd

# Configure ONE preprocessor for your dataset's language mix
# Enable FA flags only if your data contains Persian/Farsi text
preproc = Preprocessor(
    # === Language-agnostic cleaning (always useful) ===
    remove_punctuation=True,
    strip_accents=True,           # é → e, ü → u
    remove_cjk=True,              # Remove Chinese/Japanese/Korean spam
    remove_ctrl_chars=True,       # Strip Unicode control/format chars
    remove_parentheticals=True,   # Remove text in () or []
    do_hard_limit=True,           # Filter words by length
    low_hard_limit=3,
    high_hard_limit=26,
    
    # === Farsi/Persian support (enable if your data has FA text) ===
    normalize_fa_chars=True,      # Map Arabic variants → Farsi (ه→ه, ی→ی)
    remove_fa_stopwords=True,     # Remove common Farsi stopwords via hazm
    do_fa_normalizer=True,        # Apply hazm.Normalizer (spacing, digit conversion)
    fa_tokenizer=True,            # Use hazm.word_tokenize (script-aware splitting)
    
    # === English support ===
    remove_en_stopwords=True,     # Remove NLTK English stopwords
    do_en_normalizer=True,        # Lowercase text
    do_en_lemmatizer=True,        # WordNet lemmatization
    en_lemmatize_method="known",  # POS-aware lemmatization for better accuracy
)

# Use the SAME preprocessor for all text — no language detection needed!
# The preprocessor gracefully handles monolingual or mixed-language input
df["clean_text"] = df["raw_text"].apply(preproc)

# Example outputs:
# "This is a test! 🎉" → "this test"
# "Mixed: hello و دنیا" → "mixed hello دنیا"  (both languages processed)
```

---

## 🤝 Contributing

Yumbox thrives on community contributions! Here's how to help:

### 🐛 Found a bug?
1. Check existing issues first
2. Create a new issue with:
   - Minimal reproducible example
   - Expected vs. actual behavior
   - Your environment (`python --version`, `pip list`)

### 💡 Have an idea?
1. Open a discussion issue to brainstorm
2. Think of: new caching backends, more samplers, additional metrics, CLI enhancements
3. What to be cautious about: breaking API changes, heavy new dependencies

### 🔧 Want to code?
```bash
# Fork and clone
git clone https://github.com/eshteharipour/yumbox.git
cd yumbox

# Create dev environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"  # Includes test/lint dependencies

# Make changes, add tests
pytest tests/  # Ensure nothing breaks
black yumbox/  # Format code
mypy yumbox/   # Type check

# Submit PR with description of changes
```

### 📝 Documentation
- Missing examples? Confusing API? PRs welcome!
- Include code snippets that actually run

---

## 📄 License

[MIT](LICENSE) — do whatever you want, just don't blame anyone if it breaks! 😉

---

## 🙏 Acknowledgments

Yumbox stands on the shoulders of giants:
- [FAISS](https://github.com/facebookresearch/faiss) for blazing-fast similarity search
- [MLflow](https://mlflow.org) for experiment tracking done right
- [hazm](https://github.com/roshan-research/hazm) for Persian NLP
- [parsel](https://github.com/scrapy/parsel) for robust HTML/XML parsing
- My university professors, work colleagues and friends, who helped shape this project.

---

## 🍱 Why "Yumbox"?

Because machine learning should be:
- **Yummy**: Enjoyable to use, satisfying results
- **Modular**: Mix and match components like ingredients
- **Ready-to-eat**: Minimal setup, maximum productivity

Now go build something delicious! 🚀

*P.S. If you use Yumbox in your project, I'd love to hear about it. Tag me, star the repo, or just send a 🍱 emoji. I collect them.*
