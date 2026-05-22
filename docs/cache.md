## `cache/` — Your caching Swiss Army knife 🔪

A collection of decorators and utilities to make caching *just work* when you need it, whether you're storing pickles, numpy arrays, FAISS indices, or serving from Redis. All decorators respect `BFG["cache_dir"]` from config and log their actions via the `YumBox` logger. You no longer need to redefine getters/setters on each project, with a standard interface you can cache the valuable AI model inferences and outputs very easily.

### 🎯 Quick Start

```python
from yumbox.cache import cache, np_cache, retry

@cache  # Auto-saves/loads result to cache_dir/{func_name}.pkl
def expensive_computation(x):
    return heavy_ml_pipeline(x)

@np_cache  # For functions returning dict[str, np.ndarray]
def extract_features(data, cache=None, cache_file=None):
    if cache:  # Partial cache hit!
        # merge cached + new results
        ...
    return {"embeddings": embeddings, "labels": labels}

@retry(max_tries=3, backoff_factor=1)  # Auto-retry on failure
def flaky_api_call(url):
    return requests.get(url).json()
```

---

### 🧰 Decorators Reference

#### Basic Caching
| Decorator | Format | Key Strategy | Best For |
|-----------|--------|-------------|----------|
| `@cache` | `.pkl` | function name only | Simple, deterministic functions |
| `@cache_kwargs` | `.pkl` | `func_{k1-v1,k2-v2}.pkl` | Functions where kwargs change output |
| `@cache_kwargs_hash` | `.pkl` | MD5 hash of sorted kwargs | When kwargs are long/complex |
| `@timed_cache(hours)` | `.pkl` | function name + mtime check | Stale-while-revalidate patterns |
| `@async_cache` | `.pkl` | function name | `async def` functions |

#### NumPy-Aware Caching
| Decorator | Format | Key Strategy | Notes |
|-----------|--------|-------------|-------|
| `@np_cache` | `.npz` | function name | Expects `cache`/`cache_file` params in func |
| `@np_cache_lazy` | `.npz` | function name | Loads only requested keys via mmap |
| `@np_cache_kwargs` | `.npz` | `func_{kwargs}.npz` | kwargs affect cache key |
| `@np_cache_kwargs_hash` | `.npz` | MD5 of kwargs | Safer for long/unordered kwargs |
| `@np_cache_kwargs_list_hash` | `.npz` | MD5 of `cache_kwargs` subset | Fine-grained control over what affects cache |
| `@unsafe_np_cache` | `.npz` | function name | Enables `allow_pickle=True` (⚠️ security) |

#### Specialized Formats
| Decorator | Backend | Use Case |
|-----------|---------|----------|
| `@index` | FAISS `.bin` | Caching vector indices |
| `@safe_cache` | safetensors `.safetensors` | Secure, framework-agnostic tensor storage |
| `@kv_cache` | pickle `.pkl` | Generic key-value caching pattern |
| `@hd5_cache` | HDF5 `.h5` | Large numeric datasets with key-based access |
| `@lmdb_cache` | LMDB (vector) | High-performance local key-value store for arrays |
| `@redis_cache` | Redis | Distributed caching across workers/nodes |

#### Retry & Pagination Helpers
| Decorator | Purpose |
|-----------|---------|
| `@retry(...)` | Retry on exceptions with fixed/exponential backoff, validator support, and graceful error returns |
| `@last_offset` | Cache paginated results + persist last offset (integer) for resumable iteration |
| `@last_str_offset` | Same as above, but offset is a string (e.g., cursor tokens) |

##### `@retry` Configuration
```python
@retry(
    max_tries=5,           # Total attempts (1 initial + 4 retries)
    wait=2,                # Fixed 2s between retries
    backoff_factor=1,      # OR: exponential: 1s, 2s, 4s, 8s...
    max_wait=30,           # Cap exponential backoff at 30s
    validator=my_check,    # Optional: validate response before accepting
    return_exception=True, # Return {"status":"error",...} instead of raising
    retry_on400=False      # Don't retry on client errors by default
)
def call_service(...): ...
```

---

### 🗄️ Storage Backends (`cache/*.py`)

#### `fs.py` — Safe File I/O Utilities
Atomic writes, lock retries, and format-agnostic helpers:

```python
from yumbox.cache.fs import safe_wpickle, safe_rpickle, FSImage

# Safe pickle (writes to temp, then atomic move)
safe_wpickle("result.pkl", my_obj)
data = safe_rpickle("result.pkl")

# Browse & sample images from directories
fs_img = FSImage(
    data_dirs=["/data/imgs", "/backup/imgs"],
    exts=[".jpg", ".png"],  # or "image" for MIME prefix
    depth=2  # max subdirectory depth (0 = unlimited)
)
sample = fs_img.get_random(seed=42)  # reproducible sampling
```

| Utility | Purpose |
|---------|---------|
| `safe_save*` / `safe_load*` | Atomic write/read via temp file + `shutil.move` |
| `retry_on_lock` | Decorator to retry on `EACCES`/`EAGAIN` |
| `safe_wpickle` / `safe_rpickle` | Pickle helpers with atomicity + lock retries |
| `FSImage` | Lightweight image dataset loader with MIME/extension filtering |

#### `hdf.py` — HDF5 Caching
Store `dict[str, np.ndarray]` with base64-encoded keys (HDF5 doesn't like special chars):

```python
@hd5_cache
def compute_embeddings(texts, keys=None, cache=None, cache_file=None):
    # cache comes pre-loaded for requested `keys`
    # return dict of new/updated arrays
    return {"emb_001": arr1, "emb_002": arr2}
```

- Keys are base64-encoded internally → safe for HDF5
- Supports partial loads via `keys=` kwarg
- Compression: `gzip` by default

#### `kv.py` — LMDB Stores
Three flavors for different needs:

| Class | Use Case | Serialization |
|-------|----------|---------------|
| `LMDB` | Simple key → dict | msgpack |
| `LMDBMultiIndex` | Multiple keys → same data (dedup) | msgpack + indirection |
| `VectorLMDB` | key → `np.ndarray` | gzip-compressed numpy |

```python
# VectorLMDB example
with VectorLMDB("embeddings", "/cache/dir") as db:
    db.upsert("doc_123", embedding_array)
    vec = db.get("doc_123")
    meta = db.get_info("doc_123")  # shape, dtype, size without loading
```

Bulk operations (`bulk_upsert`, `bulk_get`) use LMDB's `putmulti` for speed.

#### `redis.py` — Distributed Vector Cache
`VectorRedis` mirrors `VectorLMDB` API but over Redis:

```python
# Connect
vredis = VectorRedis(
    db_name="prod_embeddings",
    host="redis.internal",
    port=6379,
    password=os.getenv("REDIS_PASS")
)

# Use
vredis.upsert("user_42", embedding, ttl=3600)  # 1h expiry
batch = vredis.bulk_get(["user_42", "user_43"])
```

- Arrays compressed with gzip before storage
- Supports TTL, pipelines, and `scan_iter` for key listing
- Decorators `@redis_cache` / `@redis_cache_kwargs_list_hash` work like their local counterparts

---

### 🔑 Common Patterns

#### Partial Cache Hits (NumPy/HDF5/LMDB)
Functions decorated with `*_cache` receive `cache` (pre-loaded results) and `cache_file` (path/identifier). Use them to avoid recomputing:

```python
@np_cache
def process_batch(items, keys=None, cache=None, cache_file=None):
    results = {}
    for key in keys:
        if cache and key in cache:
            results[key] = cache[key]  # reuse
        else:
            results[key] = heavy_compute(key)  # compute new
    return results  # decorator saves merged results
```

#### Hash-Based Keys for Complex kwargs
When kwargs are dicts/lists/unhashable, use the `_hash` variants:

```python
@cache_kwargs_hash  # or np_cache_kwargs_hash
def train_model(config: dict, data_path: str):
    # cache key = MD5(json.dumps(sorted(config)))
    ...
```

#### Resumable Pagination
Combine `@last_offset` with limit-based fetching:

```python
@last_offset
def fetch_page(source, offset=0, limit=100, cached_output=None):
    if cached_output:
        # merge new page with cached results
        ...
    new_items, next_offset = api.get_batch(offset, limit)
    return cached_output + new_items, next_offset
```

---

### ⚙️ Configuration

All caching respects `BFG["cache_dir"]` from `yumbox.config`. Set it once at startup:

```python
# config.yaml
cache_dir: "/tmp/yumbox_cache"

# or in code
from yumbox.config import BFG
BFG["cache_dir"] = Path.home() / ".cache" / "yumbox"
```

No `cache_dir`? Decorators become no-ops (functions execute normally, no I/O).

---

> 💡 **Pro tip**: Mix and match! Use `@np_cache_lazy` for local dev, swap to `@redis_cache` for distributed training — same function signature, zero logic changes.

Happy caching! 🚀 If you find a decorator that doesn't quite fit your use case, the code is intentionally modular — extend away.
