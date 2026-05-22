## `factory/` — FAISS Indexes & Feature Processing 🏭🔍

Factory functions and a builder class for rapid prototyping with FAISS: dimensionality reduction, clustering, index construction, and similarity computation. All functions assume `float32` inputs and default to **inner product (cosine) similarity** unless noted.

---

### 🎯 Quick Start

```python
import numpy as np
from yumbox.factory import build_index, pca_faiss, FaissIndexBuilder

# Your embeddings: (n_samples, embed_dim)
features = np.random.randn(10000, 768).astype(np.float32)

# Option 1: Quick flat index for exact search
index = build_index(features)  # IndexFlatIP
D, I = index.search(features[:5], k=10)  # top-10 neighbors for first 5 vectors

# Option 2: PCA reduction + FAISS-native transform
features_reduced = pca_faiss(features, n_components=128)
index_small = build_index(features_reduced)

# Option 3: Use the builder for advanced indexes with metrics
builder = FaissIndexBuilder(verbose=True)
hnsw_index = builder.build_hnsw_index(
    features, 
    M=32,           # neighbors per node
    efConstruction=200  # construction search depth
)
```

---

### 🧰 Standalone Functions (`factory/__init__.py`)

#### Dimensionality Reduction

| Function | Backend | Output | Best For |
|----------|---------|--------|----------|
| `pca(features, n_components)` | sklearn | `(n, n_components)` | Quick prototyping, interpretable components |
| `pca_faiss(features, n_components)` | FAISS `PCAMatrix` | `(n, n_components)` | Production pipelines, FAISS-native transforms |

```python
# sklearn PCA (random_state=362 for reproducibility)
reduced = pca(features, n_components=128)

# FAISS PCA (faster on large data, same result)
reduced = pca_faiss(features, n_components=128)
```

#### Clustering

```python
from yumbox.factory import kmeans_faiss

# Train FAISS K-means (GPU-accelerated if available)
kmeans = kmeans_faiss(
    features, 
    ncentroids=1024,  # number of clusters
    niter=20,         # training iterations
    verbose=True
)

# Assign vectors to nearest centroid
_, assignments = kmeans.index.search(features, k=1)
```

#### Index Builders

| Function | Metric | Index Type | Use Case |
|----------|--------|------------|----------|
| `build_index(features, dtype)` | Inner Product | `IndexFlatIP` | Exact cosine similarity search |
| `build_index_l2(features)` | L2 (Euclidean) | `IndexFlatL2` | Exact distance-based search |

```python
# Cosine similarity via inner product (vectors should be L2-normalized)
index_ip = build_index(features)

# Euclidean distance
index_l2 = build_index_l2(features)

# Optional: cast to float16 for memory savings (if supported)
index_fp16 = build_index(features, dtype=np.float16)
```

#### Pairwise Similarity Matrix

```python
from yumbox.factory import self_similarity

# Compute full n×n similarity matrix with batching (memory-efficient)
# Input should be L2-normalized for cosine similarity
from sklearn.preprocessing import normalize
features_norm = normalize(features, axis=1)

sim_matrix = self_similarity(features_norm, batch_size=2048)
# sim_matrix[i, j] ≈ features_norm[i] @ features_norm[j].T
```

> ⚡ **Why not just `x @ x.T`?** For large `n`, `self_similarity` uses FAISS batching to avoid OOM errors and leverages FAISS optimizations for inner product search.

---

### 🏗️ `FaissIndexBuilder` — Advanced Index Factory (`faiss_indexes.py`)

A unified interface for building production-ready FAISS indexes with consistent logging and GPU support.

#### Quick Reference

| Method | Index Type | Train Required? | Speed | Accuracy | Memory |
|--------|-----------|-----------------|-------|----------|--------|
| `build_flat_ip_index()` | `IndexFlatIP` | ❌ | 🐌 Slow | ✅ Exact | 🔴 High |
| `build_ivf_index(nlist, nprobe)` | `IndexIVFFlat` | ✅ | 🚀 Fast | 🟡 Approx | 🟡 Medium |
| `build_hnsw_index(M, efConstruction)` | `IndexHNSWFlat` | ❌ | ⚡ Very Fast | 🟢 High | 🟡 Medium |
| `build_pq_index(m, nbits)` | `IndexPQ` | ✅ | ⚡ Fast | 🔵 Lower | 🟢 Low |
| `build_ivfpq_index(nlist, m, nbits)` | `IndexIVFPQ` | ✅ | 🚀 Very Fast | 🔵 Approx | 🟢 Very Low |

#### Parameter Guide

##### IVF (Inverted File) Index
```python
builder.build_ivf_index(
    features,
    nlist=100,      # ~sqrt(n) is a good starting point
    nprobe=5,       # higher = more accurate but slower search
    use_gpu=False
)
```
- **Rule of thumb**: `nlist ≈ √n_vectors`, `nprobe ∈ [1, 10]`
- Requires `train()` before `add()`

##### HNSW (Graph-Based) Index
```python
builder.build_hnsw_index(
    features,
    M=32,              # neighbors per node (16-64 typical)
    efConstruction=200,  # search depth during build (higher = better graph)
    use_gpu=False
)
```
- **Rule of thumb**: `M=16-32` for speed, `M=48-64` for accuracy
- No training required; slower build, faster query

##### PQ (Product Quantization) Index
```python
builder.build_pq_index(
    features,
    m=8,        # subquantizers (embed_dim must be divisible by m)
    nbits=8,    # bits per subquantizer (4, 8, or 16)
    use_gpu=False
)
```
- **Compression**: `embed_dim=768, m=8, nbits=8` → 8×8=64 bits per vector vs 768×32=24,576 bits raw
- Trade accuracy for massive memory savings

##### IVFPQ (Hybrid) Index
```python
builder.build_ivfpq_index(
    features,
    nlist=100,   # IVF clusters
    m=8,         # PQ subquantizers
    nbits=8,     # bits per subquantizer
    nprobe=5,    # IVF probe count
    use_gpu=False
)
```
- Best of both worlds: IVF for coarse search, PQ for fine-grained compression
- Ideal for billion-scale vector search

#### GPU Support
All builder methods accept `use_gpu=True`:
```python
# Requires faiss-gpu installed and CUDA available
gpu_index = builder.build_hnsw_index(features, use_gpu=True)
```

#### Verbose Output
```python
builder = FaissIndexBuilder(verbose=True)
# Prints: build time, index size, dimension, and method-specific params
```

---

### 🔁 Common Patterns

#### Cosine Similarity Workflow (Recommended)
```python
from sklearn.preprocessing import normalize
from yumbox.factory import FaissIndexBuilder

# 1. L2-normalize for cosine similarity via inner product
features_norm = normalize(features, axis=1)

# 2. Build HNSW index for fast approximate search
builder = FaissIndexBuilder()
index = builder.build_hnsw_index(features_norm, M=32)

# 3. Search (results are cosine similarities)
D, I = index.search(query_vectors, k=10)
# D contains similarity scores (higher = more similar)
```

#### Incremental Index Building
```python
# FAISS indexes support adding vectors after creation
index = builder.build_flat_ip_index(initial_features)

# Later: add new embeddings without rebuilding
new_features = get_new_embeddings()
new_features_norm = normalize(new_features, axis=1)
index.add(new_features_norm)  # O(1) append
```

#### Memory-Efficient Similarity Matrix
```python
# For medium-sized datasets (10k-100k vectors)
from yumbox.factory import self_similarity

# Batch computation avoids OOM
sim_matrix = self_similarity(
    normalize(features, axis=1),  # cosine via IP
    batch_size=4096  # tune based on available RAM
)
```

#### Index Selection Heuristics
```python
def recommend_index(n_vectors: int, embed_dim: int, recall_target: float = 0.95):
    """
    Quick heuristic for index selection.
    """
    if n_vectors < 10_000:
        return "flat"  # Exact search is fast enough
    elif n_vectors < 1_000_000 and recall_target > 0.99:
        return "hnsw"  # High accuracy, fast search
    elif n_vectors < 10_000_000:
        return "ivf"   # Good speed/accuracy balance
    else:
        return "ivfpq" # Scalable to billions
```

---

### ⚠️ Gotchas & Tips

| Issue | Solution |
|-------|----------|
| `IndexFlatIP` returns unbounded scores | L2-normalize inputs first for cosine similarity in [0, 1] |
| PQ/IVFPQ requires `embed_dim % m == 0` | Pad features or choose `m` that divides `embed_dim` |
| IVF indexes require training before adding | Always call `train()` before `add()` for IVF variants |
| HNSW build time scales with `efConstruction` | Start with `efConstruction=200`, increase only if recall is low |
| GPU indexes need `faiss-gpu` package | Install via `pip install faiss-gpu` or use CPU fallback |

#### Pro Tips
```python
# Tip 1: Reuse quantizers across IVF indexes for consistency
quantizer = faiss.IndexFlatIP(embed_size)
index1 = faiss.IndexIVFFlat(quantizer, embed_size, nlist=100)
index2 = faiss.IndexIVFFlat(quantizer, embed_size, nlist=200)  # same quantizer

# Tip 2: Save/load indexes efficiently
faiss.write_index(index, "my_index.faiss")
loaded = faiss.read_index("my_index.faiss")

# Tip 3: Tune nprobe at query time (no rebuild needed)
index.nprobe = 10  # increase for higher recall, decrease for speed
```

> 💡 For production systems, start with `build_hnsw_index()` — it offers the best balance of speed, accuracy, and ease of use. Only switch to IVF/PQ variants when you hit memory or latency constraints. And always **normalize your vectors** before using inner product indexes!

---

> 🍱 **Yumbox Philosophy**: Never hardcode FAISS setup in multiple places. Centralize index creation with `yumbox.factory`—it cuts boilerplate, and when you need to optimize performance, enable GPU, or distribute load, you update one line instead of chasing down imports.

Happy indexing! 🚀 If you need a custom index configuration, `FaissIndexBuilder` is designed to be extended — subclass away.
