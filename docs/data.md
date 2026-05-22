## `data/` — Datasets, Samplers & Training Utilities 📦🎯

Flexible, mode-aware `Dataset` classes and smart `Sampler` implementations for contrastive learning, triplet loss, siamese networks, and general ML prototyping. Built to work seamlessly with PyTorch, handle missing features gracefully, and support both local and web-based image loading.

---

### 🎯 Quick Start

```python
from torch.utils.data import DataLoader
from yumbox.data import FlexibleDataset, ContrastiveSampler

# Simple text dataset with preprocessing
dataset = FlexibleDataset(
    mode="text",
    texts=["hello world", "another example"],
    txt_callables=[lambda t: t.lower(), lambda t: t.strip()],
)

# Contrastive learning: batches with multiple samples per class
sampler = ContrastiveSampler(
    dataset=dataset,
    labels=[0, 0, 1, 1, 2, 2],  # one label per sample
    batch_size=32,
    num_classes_per_batch=8,
    samples_per_class=4,
)

loader = DataLoader(dataset, sampler=sampler, batch_size=32)
for batch in loader:
    texts, labels = batch
    # train your contrastive model
```

---

### 🗂️ Dataset Classes Reference

#### Core Flexible Datasets (`trainer.py`)

| Class | Mode | Returns | Use Case |
|-------|------|---------|----------|
| `FlexibleDataset` | `text` \| `image` \| `text_image` | `(item, label)` | Single-item training (classification, embedding) |
| `FlexiblePairDataset` | `text_pair` \| `image_pair` \| `text_image_pair` | `((item1, item2), label)` | Siamese networks, pairwise similarity |
| `FlexibleTripletDataset` | `text_triplet` \| `image_triplet` \| `text_image_triplet` | `((a, p, n), label)` | Triplet loss, metric learning |

##### Key Features
- **Callable pipelines**: Chain transforms via `txt_callables` / `img_callables`
- **Framework-agnostic transforms**: Supports both `torchvision.Compose` and `albumentations.Compose`
- **Reproducible shuffling**: `set_epoch(epoch)` uses epoch as RNG seed for consistent shuffling
- **Graceful degradation**: Missing images/texts don't crash — logged and skipped

```python
# Image dataset with mixed transforms
from torchvision import transforms
import albumentations as A

dataset = FlexibleDataset(
    mode="image",
    images=["img1.jpg", "img2.jpg"],
    img_callables=[
        transforms.Resize(224),  # torchvision
        A.HorizontalFlip(p=0.5),  # albumentations
        lambda img: img / 255.0,  # custom callable
    ],
)
```

#### Specialized Datasets (`__init__.py`)

| Class | Purpose | Key Params |
|-------|---------|------------|
| `WebImgDataset` | Lazy-load images from URLs | `df`, `path_col`, `hash_col`, `features` (cache) |
| `ImgDataset` | Load local images, skip already-processed | `df`, `path_col`, `hash_col`, `features` |
| `TextDataset` | Tokenize text, skip cached entries | `df`, `text_col`, `id_col`, `preprocessor`, `tokenizer` |
| `TFDocumentDataset` | Chunk long documents for transformer models | `max_seq_length`, `overlap` for sliding window |
| `ZeroshotDataset` | Format prompts for zero-shot classification | `templates` list (e.g., `["a photo of {}", "an image of {}"]`) |
| `PairDataset` | Pair dataset (text/image/mixed) with label support | `mode`, `texts1/2`, `images1/2`, `hash` for caching |

##### Cache-Aware Design
All specialized datasets accept a `features: dict[str, np.ndarray]` argument representing *already-computed embeddings*. They automatically:
1. Filter the input DataFrame to rows whose `hash_col` values are **not** in `features.keys()`
2. Return only the *missing* items for processing

```python
# Only process images we haven't embedded yet
existing_features = {"abc123": emb1, "def456": emb2}  # from previous run

dataset = WebImgDataset(
    df=image_df,
    path_col="url",
    hash_col="image_hash",
    features=existing_features,  # skip these!
    embed_dim=512,
    transform=my_transform,
)
# dataset.data now contains only rows needing processing
```

---

### 🎲 Custom Samplers (`trainer.py`)

Specialized `torch.utils.data.Sampler` implementations for advanced training strategies.

#### Contrastive Learning
```python
sampler = ContrastiveSampler(
    dataset=dataset,
    labels=[0, 0, 1, 1, 2, 2],  # class label per sample
    batch_size=64,
    num_classes_per_batch=8,   # 8 different classes per batch
    samples_per_class=4,       # 4 examples of each class
)
# → Batch contains 8 classes × 4 samples = 32 items, ideal for InfoNCE loss
```

#### Triplet Loss
```python
sampler = TripletSampler(
    dataset=dataset,
    labels=labels,
    batch_size=96,              # Must be divisible by (2 + neg_to_pos_ratio)
    neg_to_pos_ratio=2,         # 1 anchor + 1 positive + 2 negatives = 4 per triplet
)
# → Each batch: 96 / 4 = 24 triplets
```

#### Siamese Networks
```python
sampler = SiameseSampler(
    dataset=dataset,
    labels=labels,
    batch_size=64,              # Must be even (pairs)
    pos_neg_ratio=0.5,          # 50% same-class pairs, 50% different-class
)
```

#### Cluster-Aware Sampling
```python
sampler = ClusterSampler(
    dataset=dataset,
    cluster_ids=[0, 0, 1, 2, 2, 2],  # cluster assignment per sample
    samples_per_cluster=2,           # take 2 from each cluster per batch
    batch_size=32,
)
# → Ensures batch diversity across clusters
```

##### Sampler Best Practices
- All samplers respect `dataset.set_epoch(epoch)` for reproducible shuffling
- Use `warnings.filterwarnings("ignore")` if you have classes with <2 samples (triplet/siamese)
- For distributed training, wrap with `DistributedSampler` *outside* these custom samplers

---

### 🛠️ Utility Functions

#### Dataset Iteration Helpers
```python
from yumbox.data import get_subset, get_dataloader

# Process dataset in chunks (great for large-scale inference)
params = get_subset(
    dataset=my_dataset,
    epoch=0,
    iteration=0,
    batch_size=32,
    batches_per_iteration=10,  # process 10 batches at a time
    dataset_size=len(my_dataset),
)

loader = get_dataloader(
    dataset=my_dataset,
    start_idx=params["start_idx"],
    end_idx=params["end_idx"],
    batch_size=32,
)

for batch in loader:
    # process chunk
    pass
```

#### Text Utilities
```python
from yumbox.data import split_token_ids, TFDocumentDataset

# Chunk long token sequences with overlap (for transformer models)
tokens = tokenizer.encode(long_text)
chunks = list(split_token_ids(tokens, chunk_size=512, overlap=64))
# → [[tok0...tok511], [tok448...tok959], ...]
```

#### Plotting Helpers (`plt.py`)
```python
from yumbox.data.plt import fix_farsi, calc_subplot_height

# Fix RTL text rendering in matplotlib
plt.title(fix_farsi("عنوان فارسی"))

# Auto-calculate subplot grid
height = calc_subplot_height(width=3, count_of_items=10)  # → 4 rows
```

#### Pandas Display Fix
```python
from yumbox.data import fix_pandas_truncation
fix_pandas_truncation()  # No more "..." in df output!
```

---

### 🔁 Common Patterns

#### Incremental Embedding Pipeline
```python
# 1. Load existing embeddings (from cache)
existing = load_cached_embeddings("embeddings.npz")  # dict[hash, array]

# 2. Create dataset that skips cached items
dataset = ImgDataset(
    df=image_df,
    path_col="path",
    hash_col="hash",
    features=existing,
    transform=preprocess,
)

# 3. Process only missing items
model = MyEmbeddingModel()
new_features = {}
for hash_key, img in DataLoader(dataset, batch_size=32):
    with torch.no_grad():
        emb = model(img)
        new_features.update(dict(zip(hash_key, emb.numpy())))

# 4. Merge and save
all_features = {**existing, **new_features}
save_embeddings("embeddings.npz", all_features)
```

#### Zero-Shot Classification Prep
```python
from yumbox.data import ZeroshotDataset

templates = [
    "a photo of a {}",
    "a blurry image of a {}",
    "a good photo of a {}",
]

dataset = ZeroshotDataset(
    df=class_df,
    text_col="class_name",
    id_col="class_id",
    features={},  # no precomputed prompts
    templates=templates,
    preprocessor=lambda t: t.lower(),
    tokenizer=clip_tokenizer,  # returns tensor
)

# Dataset expands: 100 classes × 3 templates = 300 items
for class_id, prompt_tensor in dataset:
    # feed prompt_tensor to text encoder
    pass
```

#### Mixed Modality Training
```python
# Text-image pairs for contrastive alignment
dataset = FlexiblePairDataset(
    mode="text_image_pair",
    texts1=captions_a,
    images1=paths_a,
    texts2=captions_b,
    images2=paths_b,
    txt_callables=[normalize_text],
    img_callables=[
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ],
)

# Returns: ((text_a, img_a), (text_b, img_b)), label
```

---

### ⚠️ Gotchas & Tips

| Issue | Solution |
|-------|----------|
| `OSError: Cannot allocate memory` on image load | Wrap `Image.open` in try/except; dataset logs and skips corrupted files |
| Albumentations + torchvision transforms mixing | `FlexibleDataset` auto-detects transform type — no manual conversion needed |
| Long documents truncating in transformers | Use `TFDocumentDataset` with `overlap` to preserve context across chunks |
| RTL text garbled in plots | Wrap labels with `fix_farsi()` from `data.plt` |
| Dataset size mismatch after filtering | Call `dataset.verify_dataset_size(expected)` before training |

---

> 💡 **Pro tip**: Combine `FlexibleDataset.set_epoch(epoch)` + custom `Sampler` for *fully reproducible* multi-GPU training. Set the same seed on all ranks, and every worker sees the same shuffled order — critical for contrastive learning stability.

Happy data loading! 🚀 If your use case needs a new mode or sampler, the classes are intentionally composable — extend away.
