## `vectors/` — Vector Operations, Search & Feature Fusion 🧮🔍

High-performance utilities for vector similarity search, feature combination, and array manipulation. Built for efficient ML pipelines with support for parallel FAISS queries, multimodal feature fusion, and robust handling of missing data.

---

### 🎯 Quick Start

```python
import numpy as np
from yumbox.vectors import topk, normalize_vector, cat_feats, mult_feats
from yumbox.factory import build_index

# 1. L2-normalize embeddings for cosine similarity
embeddings = np.random.randn(1000, 768).astype(np.float32)
normalized = normalize_vector(embeddings)  # Shape: (1000, 768), unit vectors

# 2. Parallel top-k search with FAISS index
# Factory: zero-boilerplate FlatIP index
# Auto-creates IndexFlatIP, infers dim, adds data
index = build_index(normalized)

# Search 100 queries, k=10, with 4 parallel processes
distances, indices = topk(
    index=index,
    search_method="search",  # FAISS method name
    queries=normalized[:100],
    k=10,
    num_processes=4,
    keepdims=False  # Flatten output for k=1 or single query
)
# distances: (1000,) array of similarity scores
# indices: (1000,) array of result indices

# 3. Combine features from two modalities (e.g., text + image)
import pandas as pd
df = pd.DataFrame({
    "text_id": ["t1", "t2", "t3"],
    "img_id": ["i1", "i2", "i3"]
})
text_feats = {"t1": np.array([0.1, 0.2]), "t2": np.array([0.3, 0.4]), "t3": np.array([0.5, 0.6])}
img_feats = {"i1": np.array([0.7, 0.8]), "i2": np.array([0.9, 1.0]), "i3": np.array([1.1, 1.2])}

# Concatenate features (with L2 normalization after)
combined = cat_feats(
    df=df,
    feats_a=text_feats,
    feats_b=img_feats,
    colname_a="text_id",
    colname_b="img_id",
    normalize="after"  # Normalize the concatenated result
)
# → Shape: (3, 4), each row is [text_feat, img_feat] normalized

# 4. Element-wise multiplication (for gating/attention-style fusion)
gated = mult_feats(
    df=df,
    feats_a=text_feats,
    feats_b=img_feats,
    colname_a="text_id",
    colname_b="img_id",
    normalize="before"  # Normalize each modality before multiplying
)
# → Shape: (3, 2), element-wise product of normalized vectors
```

---

### 🧰 Function Reference

#### Vector Search Utilities

##### `topk()` — Parallelized Nearest Neighbor Search
```python
def topk(
    index: Type,                      # FAISS index or similar with .search() method
    search_method: str,               # Method name to call (e.g., "search")
    queries: np.ndarray | list,       # Query vectors: (n_queries, dim) or (dim,)
    k: int,                           # Number of neighbors to return
    keepdims: bool = False,           # Keep 2D output even for k=1 or single query
    search_size: int | None = None,   # Override auto batch size
    num_processes: int | None = None, # Parallel workers (default: CPU count)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform parallelized top-k search with automatic batching.

    Returns:
        distances: Similarity scores, shape (n_queries * k,) if keepdims=False else (n_queries, k)
        indices: Result indices, same shape as distances
    """
```

**Key Features:**
- **Auto-batching**: Splits large query sets into chunks for memory efficiency
- **Multiprocessing**: Uses `multiprocessing.Pool` for CPU-bound search parallelism
- **Flexible output**: `keepdims=False` flattens results for easy indexing; `True` preserves 2D structure
- **FAISS-agnostic**: Works with any index exposing a `search_method(queries, k)` callable

**Example: FAISS HNSW Search**
```python
from yumbox.vectors import topk
from yumbox.factory import FaissIndexBuilder

# Build HNSW index for fast approximate search
builder = FaissIndexBuilder(verbose=False)
# Factory: Handles init + add
index = builder.build_hnsw_index(normalized_embeddings, M=32, efConstruction=200)

# Search with parallel processing
dists, idxs = topk(
    index=index,
    search_method="search",
    queries=query_vectors,
    k=20,
    num_processes=8,
    keepdims=True  # Preserve (n_queries, 20) shape
)

# Access top-1 result for each query
top1_scores = dists[:, 0]
top1_indices = idxs[:, 0]
```

> ⚡ **Performance tip**: For >10k queries, set `num_processes` to `min(8, cpu_count())` to avoid overhead. FAISS internal threading + Python multiprocessing can conflict — test your workload.

##### `nested_topk()` — Hierarchical Candidate Refinement
```python
def nested_topk(
    create_index_func: callable,   # Function: candidates → index
    search_func_name: str,         # Method name to call on index
    topk_candids: np.ndarray,      # Candidate sets per query: (n_queries, n_candids)
    queries: np.ndarray,           # Query vectors: (n_queries, dim)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform two-stage search: first retrieve candidates, then refine within each candidate set.

    Use case: Coarse-to-fine retrieval, e.g., cluster-based filtering before exact search.
    """
```

**Example: Cluster-Then-Search Pipeline**
```python
import numpy as np
from yumbox.vectors import nested_topk
from yumbox.factory import build_index

# Scenario: Model Cascade Retrieval
# Stage 1: Fast/cheap pipeline generates candidates (e.g., BM25 + lightweight encoder + category filter)
# Stage 2: Heavy/accurate model reranks ONLY the candidates

queries = np.random.randn(2, 512).astype(np.float32)  # User queries (same dim as heavy model)
all_product_ids = np.arange(1000)

# Simulate Stage 1: External candidate generator (e.g., metadata filter + fast model)
topk_candids = np.random.choice(all_product_ids, size=(2, 50), replace=False)  # 50 candidates/query

def make_rerank_index(cand_ids: np.ndarray):
    """Build exact index using heavy, accurate embeddings for the candidate subset."""
    # In practice: heavy_embeddings = reranker_model.encode(products[cand_ids])
    heavy_embeddings = np.random.randn(len(cand_ids), 512).astype(np.float32)
    # Factory: auto-infers dim, creates IndexFlatIP, and adds data in one call
    return build_index(heavy_embeddings)

# Stage 2: Exact rerank within candidates using yumbox.vectors.nested_topk
refined_dists, refined_idxs = nested_topk(
    create_index_func=make_rerank_index,
    search_func_name="search",
    topk_candids=topk_candids,
    queries=queries
)

# Map local rerank indices → global product IDs
global_top1 = np.array([topk_candids[i][refined_idxs[i]] for i in range(len(refined_idxs))])
print(f"Reranked product IDs: {global_top1}")
```

---

#### Vector Normalization

##### `normalize_vector()` — L2 Normalization for NumPy/Torch
```python
def normalize_vector(v: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    L2-normalize vectors along the last dimension.

    Args:
        v: Input array/tensor of shape (..., dim) or (dim,)

    Returns:
        Normalized array/tensor of same shape and type

    Examples:
        >>> normalize_vector(np.array([3, 4]))  # → [0.6, 0.8]
        >>> normalize_vector(torch.randn(10, 128))  # → unit vectors, shape (10, 128)
    """
```

**Use Cases:**
- Cosine similarity via inner product: `normalize(a) @ normalize(b).T`
- Preprocessing for FAISS `IndexFlatIP` (requires unit vectors for cosine sim)
- Stabilizing gradient magnitudes in neural networks

```python
# Cosine similarity matrix (efficient batched version)
a_norm = normalize_vector(a)  # (n, d)
b_norm = normalize_vector(b)  # (m, d)
cosine_sim = a_norm @ b_norm.T  # (n, m), values in [-1, 1]
```

---

#### Feature Combination Utilities

All feature combination functions accept a DataFrame and two feature dictionaries, aligning features by ID columns with graceful handling of missing values.

##### `cat_feats()` — Concatenate Features
```python
def cat_feats(
    df: pd.DataFrame,
    feats_a: dict[str, np.ndarray],
    feats_b: dict[str, np.ndarray],
    colname_a: str,          # DataFrame column for feats_a keys
    colname_b: str,          # DataFrame column for feats_b keys
    zeros_a: np.ndarray | None = None,  # Fallback vector if feats_a missing
    zeros_b: np.ndarray | None = None,  # Fallback vector if feats_b missing
    normalize: Literal["before", "after", None] = None,
    pca_a: Callable | None = no_op,     # Optional projection for feats_a
    pca_b: Callable | None = no_op,     # Optional projection for feats_b
) -> np.ndarray:
    """
    Concatenate features from two sources along feature dimension.

    Output shape: (len(df), dim_a + dim_b) after optional PCA, before/after normalization.
    """
```

**Example: Multimodal Embedding Fusion**
```python
# Text (768-d) + Image (512-d) → 1280-d combined embedding
combined = cat_feats(
    df=product_df,
    feats_a=text_embeddings,      # {"prod_001": np.array(...), ...}
    feats_b=image_embeddings,
    colname_a="text_feature_id",
    colname_b="image_feature_id",
    normalize="after",            # L2-normalize the 1280-d result
    pca_a=lambda x: x,            # Optional: reduce text dim first
    pca_b=lambda x: x[:256],      # Optional: truncate image features
)
```

##### `mult_feats()` — Element-Wise Multiplication (Gating)
```python
def mult_feats(..., normalize: Literal["before", "after", None] = None) -> np.ndarray:
    """
    Element-wise multiply features from two sources.

    Use case: Attention-style gating, feature modulation, or Hadamard product fusion.
    """
```

**Example: Cross-Modal Attention Gating**
```python
# Use text features to gate image features (element-wise multiplication)
gated = mult_feats(
    df=df,
    feats_a=text_attn_weights,  # Values in [0, 1]
    feats_b=image_features,
    colname_a="text_id",
    colname_b="img_id",
    normalize="before",  # Normalize each before gating
)
# Result: image features modulated by text attention
```

##### `sum_feats()` — Element-Wise Addition
```python
def sum_feats(..., normalize: Literal["before", "after", None] = None) -> np.ndarray:
    """
    Element-wise add features from two sources.

    Use case: Residual connections, ensemble averaging, or additive fusion.
    """
```

##### `diff_feats()` — Absolute Difference
```python
def diff_feats(...) -> np.ndarray:
    """
    Compute absolute difference |a - b| for feature pairs.

    Use case: Distance-based features, contrastive learning signals.
    """
```

**Example: Contrastive Pair Features**
```python
# For siamese networks: compute |emb_anchor - emb_positive|
diff_features = diff_feats(
    df=pairs_df,
    feats_a=embeddings,
    feats_b=embeddings,
    colname_a="anchor_id",
    colname_b="positive_id",
)
# Feed diff_features to classifier for match/no-match prediction
```

---

#### Feature Extraction Helpers

##### `full_featdict()` / `partial_featdict()` — Dict Lookup by DataFrame Column
```python
def full_featdict(df: pd.DataFrame, feats: dict[str, np.ndarray], colname: str) -> dict[str, np.ndarray]:
    """Extract features for ALL values in df[colname]; raises KeyError if any missing."""

def partial_featdict(df: pd.DataFrame, feats: dict[str, np.ndarray], colname: str) -> dict[str, np.ndarray]:
    """Extract features for NON-MISSING values in df[colname]; skips NaN/None keys."""
```

**Example: Safe Feature Lookup**
```python
# Safe extraction: only get features for rows with valid IDs
valid_feats = partial_featdict(df, all_embeddings, colname="product_id")
# → {"prod_001": emb1, "prod_003": emb3}  (skips row with NaN product_id)
```

##### `full_feats()` / `partial_feats()` — Array Extraction by Key List
```python
def full_feats(keys: Iterable[str], feats: dict[str]) -> np.ndarray:
    """Return array of features for all keys; raises KeyError if any missing."""

def partial_feats(keys: Iterable[str], feats: dict[str]) -> np.ndarray:
    """Return array of features for non-missing keys only."""
```

---

#### Array Reconstruction

##### `reconstruct_original_index()` — Reinsert Values at Missing Indices
```python
def reconstruct_original_index(
    target: np.ndarray | list,          # Array to insert into
    missing_indices: np.ndarray | list, # Positions where values were removed
    fill_value: np.ndarray | list | None = None,  # Values to insert (or single value)
) -> np.ndarray | list:
    """
    Reconstruct an array by inserting fill values at specified indices.

    Use case: Restore original ordering after filtering out invalid/missing entries.
    """
```

**Example: Restore Order After Filtering**
```python
# Original array: [a, b, c, d, e]
# After removing invalid: [a, c, e] at indices [0, 2, 4]
# missing_indices = [1, 3] (positions that were removed)

restored = reconstruct_original_index(
    target=[a, c, e],
    missing_indices=[1, 3],
    fill_value=np.zeros(dim)  # Insert zero vector for missing entries
)
# → [a, 0, c, 0, e]  (original shape restored)
```

---

### 🔁 Common Patterns

#### Efficient Batch Similarity Search Pipeline
```python
from yumbox.vectors import topk, normalize_vector
from yumbox.factory import build_index, FaissIndexBuilder

def batch_similarity_search(
    queries: np.ndarray,
    corpus: np.ndarray,
    k: int,
    index_type: str = "hnsw",
    num_processes: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    End-to-end similarity search with normalization, indexing, and parallel query.
    """
    # 1. Normalize for cosine similarity
    queries_norm = normalize_vector(queries)
    corpus_norm = normalize_vector(corpus)
    
    # 2. Build index
    if index_type == "flat":
        index = build_index(corpus_norm)
    elif index_type == "hnsw":
        builder = FaissIndexBuilder(verbose=False)
        index = builder.build_hnsw_index(corpus_norm, M=32, efConstruction=200)
    else:
        raise ValueError(f"Unknown index_type: {index_type}")
    
    # 3. Parallel search
    distances, indices = topk(
        index=index,
        search_method="search",
        queries=queries_norm,
        k=k,
        num_processes=num_processes,
        keepdims=True
    )
    
    return distances, indices

# Usage
dists, idxs = batch_similarity_search(
    queries=query_embeddings,
    corpus=database_embeddings,
    k=50,
    index_type="hnsw",
    num_processes=8
)
```

#### Multimodal Feature Fusion for Retrieval
```python
from yumbox.vectors import cat_feats, normalize_vector

def create_multimodal_index(
    df: pd.DataFrame,
    text_feats: dict,
    image_feats: dict,
    text_col: str = "text_id",
    image_col: str = "image_id",
    fusion_method: str = "concat",  # or "multiply", "sum"
) -> tuple[np.ndarray, dict]:
    """
    Create fused embeddings and reverse lookup index for multimodal retrieval.
    """
    if fusion_method == "concat":
        fused = cat_feats(
            df=df,
            feats_a=text_feats,
            feats_b=image_feats,
            colname_a=text_col,
            colname_b=image_col,
            normalize="after"
        )
    elif fusion_method == "multiply":
        from yumbox.vectors import mult_feats
        fused = mult_feats(
            df=df,
            feats_a=text_feats,
            feats_b=image_feats,
            colname_a=text_col,
            colname_b=image_col,
            normalize="before"
        )
    else:
        raise ValueError(f"Unknown fusion_method: {fusion_method}")
    
    # Create reverse index: embedding → original row index
    id_to_idx = {row_id: i for i, row_id in enumerate(df.index)}
    
    return fused, id_to_idx

# Usage
fused_embs, id_map = create_multimodal_index(
    df=products_df,
    text_feats=text_embeddings,
    image_feats=image_embeddings,
    fusion_method="concat"
)

# Now use fused_embs with FAISS for retrieval
```

#### Handling Missing Features Gracefully
```python
import numpy as np
import pandas as pd
from yumbox.vectors import partial_featdict, reconstruct_original_index

def safe_feature_extraction(
    df: pd.DataFrame,
    all_feats: dict[str, np.ndarray],
    id_col: str,
    fill_value: np.ndarray | None = None,
) -> np.ndarray:
    """
    Extract features with automatic handling of missing IDs.
    Returns array aligned with original DataFrame order using yumbox utilities.
    """
    # 1. partial_featdict automatically skips NaN/None keys & preserves DataFrame row order
    valid_feats = partial_featdict(df, all_feats, id_col)

    # 2. Extract aligned feature array directly (no manual indexing needed)
    features_array = np.array(list(valid_feats.values()))

    # 3. Find positions where features were missing
    valid_ids = set(valid_feats.keys())
    missing_indices = df.index[~df[id_col].isin(valid_ids)].tolist()

    # 4. Auto-generate zero-vector fallback if not provided
    if fill_value is None and len(features_array) > 0:
        fill_value = np.zeros(features_array.shape[1])

    # 5. Reconstruct full-length array with gaps filled
    return reconstruct_original_index(
        target=features_array,
        missing_indices=missing_indices,
        fill_value=fill_value
    )

# Usage: Extract embeddings even when some product IDs are missing
product_embs = safe_feature_extraction(
    df=catalog_df,
    all_feats=precomputed_embeddings,
    id_col="product_id",
    fill_value=np.zeros(768)  # Zero vector fallback
)
# → Shape: (len(catalog_df), 768), aligned with original DataFrame
```

---

### ⚠️ Gotchas & Tips

| Issue | Solution |
|-------|----------|
| `topk` multiprocessing overhead for small queries | Use `num_processes=1` for <100 queries; overhead dominates for tiny batches |
| FAISS + multiprocessing can deadlock on GPU indexes | Use CPU indexes (`IndexFlatIP`, `IndexHNSWFlat`) with multiprocessing; avoid `GpuIndex` in parallel context |
| `normalize_vector` on zero vector → division by zero | Input vectors should be non-zero; add epsilon if needed: `v / (np.linalg.norm(v) + 1e-8)` |
| Feature combination functions assume dicts have array values | Ensure `feats_a[key]` returns `np.ndarray`, not list or torch.Tensor (convert first if needed) |
| Missing value handling uses `notfona()` (truthy + not NaN) | Empty strings, 0, False are treated as "missing"; use explicit checks if these are valid values |

#### Pro Tips
```python
# Tip 1: Pre-normalize corpus once, not per-query
corpus_norm = normalize_vector(corpus)
# Then for each query batch:
query_norm = normalize_vector(queries)
dists, idxs = topk(index, "search", query_norm, k=10)

# Tip 2: Use keepdims=True for consistent shaping in pipelines
# Avoid conditional logic later:
dists, idxs = topk(..., keepdims=True)  # Always (n_queries, k)
top1 = dists[:, 0]  # Clean slicing

# Tip 3: Profile multiprocessing batch size for your workload
import time
for batch_size in [64, 128, 256, 512]:
    start = time.time()
    topk(..., search_size=batch_size, num_processes=4)
    print(f"Batch {batch_size}: {time.time() - start:.2f}s")

# Tip 4: Use np.float32 for FAISS compatibility and memory savings
embeddings = embeddings.astype(np.float32)  # FAISS expects float32
```

---

Happy vectorizing! 🚀 If your use case needs a new fusion method or search strategy, the functions are intentionally composable — extend away.
