## `metrics/` — Evaluation Made Simple 📊✨

Classification scores, Information Retrieval metrics, Entity Resolution evaluation, cosine similarity, and MLflow-ready plotting — all in one place. Whether you're training a classifier, building a search engine, or just checking if your embeddings make sense, Yumbox metrics have them all covered.

---

### 🎯 Quick Start

```python
import numpy as np
import torch
from yumbox.metrics import (
    cosine_sim,            # tensor cosine similarity
    classification_scores, # sklearn wrapper with FPR/FNR
    mean_ir_scores,        # ranking metrics (P@k, MAP, MRR, NDCG)
    AverageMeter,          # running average tracker
)

# Cosine similarity between two embeddings
emb1 = torch.randn(1, 768)
emb2 = torch.randn(1, 768)
sim = cosine_sim(emb1, emb2)  # scalar in [-1, 1]

# Classification metrics (auto-detects binary vs multi-class)
y_true = np.array([0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 0, 0, 1])
scores = classification_scores(y_true, y_pred, avg="binary")
# → {'accuracy': 0.8, 'precision-binary': 1.0, 'recall-binary': 0.66, 'f1-binary': 0.8, 'fpr': 0.0, 'fnr': 0.33}

# Ranking metrics for retrieval
true_labels = np.array(["cat", "dog", "bird"])  # ground truth
pred_rankings = np.array([  # top-3 predictions per query
    ["cat", "dog", "fish"],
    ["dog", "cat", "bird"],
    ["bird", "fish", "cat"],
])
ir_scores = mean_ir_scores(
    true=true_labels,
    pred=pred_rankings,
    k=[1, 3],  # evaluate at k=1 and k=3
    metrics=["P", "AP", "RR"]  # Precision, Average Precision, Reciprocal Rank
)
# → {'mP-1': 1.0, 'mP-3': 0.88, 'MAP-1': 1.0, 'MAP-3': 0.94, 'mRR': 1.0}

# Track training loss with running average
loss_meter = AverageMeter()
for batch in loader:
    loss = compute_loss(batch)
    loss_meter.update(loss.item(), n=len(batch))
    print(f"Loss: {loss_meter.avg:.4f}")  # smooth, stable metric
```

---

### 🧰 Function Reference

#### Similarity & Distance

| Function | Input | Output | Use Case |
|----------|-------|--------|----------|
| `jaccard_similarity(str1, str2)` | Two strings | `float` in [0, 1] | Quick text overlap check |
| `cosine_sim(emb1, emb2)` | Two `torch.Tensor` | `float` in [-1, 1] | Embedding similarity (L2-normalized) |

```python
# Jaccard for quick text comparison
jaccard_similarity("hello world", "hello there")  # → 0.33

# Cosine similarity (auto-normalizes, handles batching)
emb1 = torch.randn(32, 512)  # batch of embeddings
emb2 = torch.randn(32, 512)
sims = torch.tensor([cosine_sim(e1, e2) for e1, e2 in zip(emb1, emb2)])
```

> 💡 `cosine_sim` squeezes inputs and uses `torch.tensordot` — efficient for single pairs. For batched matrix similarity, use `emb1 @ emb2.T` after manual normalization.

#### Entity Resolution (IR + Classification) Metrics (`er.py`)

| Function | Purpose | Key Params |
|----------|---------|------------|
| `classification_scores(true, pred, avg, is_binary)` | Accuracy, Precision, Recall, F1, FPR, FNR | `avg`: `"binary"`, `"macro"`, `"weighted"` |
| `extended_classification_scores(..., topk=(1,5))` | Adds top-k accuracy to above | `topk`: tuple of k values |
| `torch_topk_accuracy(output, target, topk)` | Top-k accuracy for model logits | `output`: `(B, C)` logits, `target`: `(B,)` labels |
| `np_topk_accuracy(true, pred, topk)` | Top-k accuracy for pre-sorted predictions | `pred`: `(B, K)` with best predictions first |

```python
# Binary classification with FPR/FNR
scores = classification_scores(
    true=[0, 1, 1, 0],
    pred=[0, 1, 0, 0],
    avg="binary",  # critical for binary metrics
    is_binary=True
)
# → includes 'fpr' and 'fnr' keys

# Multi-class with multiple averaging strategies
scores = extended_classification_scores(
    true=np.array([0, 1, 2, 1]),
    pred=np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], ...]),  # logits or probs
    topk=(1, 3),
    avgs=["macro", "weighted"]
)
# → {'top-1_accuracies': 0.75, 'top-3_accuracies': 1.0, 'precision-macro': ..., ...}
```

#### Information Retrieval Metrics (`ir.py`)

| Metric | Symbol | Meaning | Best For |
|--------|--------|---------|----------|
| Precision@k | `P@k` | % of top-k results that are relevant | Search result quality |
| Average Precision | `AP` / `MAP` | Area under precision-recall curve | Ranking quality across recall levels |
| Reciprocal Rank | `RR` / `MRR` | 1 / rank of first relevant result | "Did we get it right early?" |
| NDCG@k | `NDCG@k` | Discounted gain, normalized by ideal | Graded relevance (e.g., 5-star ratings) |
| FPR@k / FNR@k | `FPR@k` | False positive/negative rate in top-k | Error analysis for retrieval |

##### `mean_ir_scores` — The One-Stop IR Evaluator

```python
from yumbox.metrics import mean_ir_scores

# Scenario: Image retrieval — for each query image, return top-10 candidates
true_ids = np.array(["img_001", "img_002", "img_003"])  # ground truth match IDs
pred_ids = np.array([  # shape: (n_queries, top_k)
    ["img_001", "img_999", "img_888", ...],  # query 1 predictions
    ["img_002", "img_777", "img_002", ...],  # query 2 (duplicate OK)
    ["img_003", "img_003", "img_666", ...],  # query 3
])

results = mean_ir_scores(
    true=true_ids,
    pred=pred_ids,
    candidates_len=10000,  # total pool size (for recall denominator)
    k=[1, 5, 10],  # evaluate at multiple cutoffs
    metrics=["P", "AP", "RR", "NDCG"],  # choose what you need
    total_positives_per_query=[1, 1, 1],  # optional: for accurate recall
)

# Example output:
# {
#   'mP-1': 1.0, 'mP-5': 0.87, 'mP-10': 0.73,
#   'MAP-1': 1.0, 'MAP-5': 0.92, 'MAP-10': 0.85,
#   'mRR': 1.0,
#   'mean-ndcg-1': 1.0, 'mean-ndcg-5': 0.94, 'mean-ndcg-10': 0.88
# }
```

##### Label Encoding for Advanced Metrics (FPR/FNR)
For `FPR`/`FNR` calculation, `mean_ir_scores` supports a special encoding in `sorted_labels`, to help evaluate the ER system from the perspective of the percentage of errors it is making:
- `1` → True Positive (predicted match, actually match)
- `0` → True Negative (predicted no-match, actually no-match)
- `-1` → False Positive (predicted match, actually no-match)
- `-2` → False Negative (predicted no-match, actually match)

```python
# Manual encoding for fine-grained error analysis
queries = [
    [1, -1, 0, -2, 1],  # query 1: TP, FP, TN, FN, TP
    [0, 0, -1, 1],      # query 2
]
scores = mean_ir_scores(
    true=..., pred=...,  # or pass pre-encoded queries directly via internal API
    metrics=["FPR", "FNR"],
    total_positives_per_query=[3, 1],
    total_negatives_per_query=[2, 3],
)
```

#### The `Scorer` Class — Per-Query Metric Calculator

Use when you easily need a suite of mertrics from a single ranked list:

```python
from yumbox.metrics import Scorer

# A single query's ranked results: [TP, FP, TN, TP, FN] encoded as [1, -1, 0, 1, -2]
scorer = Scorer(
    sorted_labels=[1, -1, 0, 1, -2],
    k=3,  # evaluate top-3
    total_positives=3,  # total relevant items in full corpus
    total_negatives=100  # total non-relevant items
)

print(scorer.precision())      # 2/3 = 0.66 (two 1's in top-3)
print(scorer.average_precision())  # AP calculation
print(scorer.reciprocal_rank())    # 1/1 = 1.0 (first result is relevant)
print(scorer.false_positive_rate())  # FP / (FP + TN) in top-k
```

#### Utility Classes & Helpers

| Class/Function | Purpose |
|----------------|---------|
| `AverageMeter` | Track running averages (loss, accuracy) during training |
| `extra_scores_plots_binary(scores, true, outdir)` | Generate PR/ROC curves + log to MLflow |
| `save_and_log_plot(fig, filename, outdir)` | Helper to save matplotlib figures to MLflow |

```python
# AverageMeter for training loops
from yumbox.metrics import AverageMeter

acc_meter = AverageMeter()
for epoch in range(10):
    acc = evaluate(model, val_loader)
    acc_meter.update(acc)
    print(f"Epoch {epoch}: {acc:.3f} (avg: {acc_meter.avg:.3f})")

# Binary classification plots with MLflow logging
from yumbox.metrics import extra_scores_plots_binary

# scores: probability for positive class, shape (n_samples,)
# true: binary labels, shape (n_samples,)
metrics = extra_scores_plots_binary(
    scores=model_outputs,
    true=y_test,
    outdir="mlruns/eval_plots"  # auto-logged to active MLflow run
)
# → logs precision_recall_curve.png, roc_curve.png + returns {'pr_auc': ..., 'roc_auc': ...}
```

---

### 🔁 Common Patterns

#### End-to-End Evaluation Pipeline
```python
def evaluate_retrieval(model, test_loader, k_values=[1, 5, 10]):
    all_true, all_pred = [], []
    
    model.eval()
    with torch.no_grad():
        for query, candidates, labels in test_loader:
            # Get embeddings and compute similarities
            query_emb = model(query)
            cand_embs = model(candidates)
            sims = (query_emb @ cand_embs.T).squeeze().cpu().numpy()
            
            # Get top-k predicted candidate IDs
            topk_idx = np.argsort(-sims)[:max(k_values)]
            pred_ids = [candidates[i].id for i in topk_idx]
            
            all_true.append(labels[0].id)  # ground truth ID
            all_pred.append(pred_ids)
    
    # Compute aggregate metrics
    results = mean_ir_scores(
        true=np.array(all_true),
        pred=np.array(all_pred),
        k=k_values,
        metrics=["P", "AP", "RR", "NDCG"]
    )
    
    # Log to MLflow
    import mlflow
    for name, value in results.items():
        mlflow.log_metric(name, value)
    
    return results
```

#### Binary Classification with Full Reporting
```python
from yumbox.metrics import classification_scores, extra_scores_plots_binary

# After training a binary classifier
y_prob = model.predict_proba(X_test)[:, 1]  # probabilities for positive class
y_pred = (y_prob >= 0.5).astype(int)

# 1. Tabular metrics
tabular = classification_scores(y_test, y_pred, avg="binary")
print(tabular)
# → {'accuracy': 0.92, 'precision-binary': 0.89, 'recall-binary': 0.91, 'f1-binary': 0.90, 'fpr': 0.05, 'fnr': 0.09}

# 2. Visual metrics + MLflow logging
visual = extra_scores_plots_binary(y_prob, y_test, outdir="plots")
print(visual)
# → {'average_precision': 0.94, 'pr_auc': 0.93, 'roc_auc': 0.96} + saves plots
```

#### Tracking Metrics During Training
```python
from yumbox.metrics import AverageMeter

# Initialize meters
loss_meter = AverageMeter()
acc_meter = AverageMeter()

for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        loss = train_step(model, batch)
        loss_meter.update(loss.item(), n=len(batch))
    
    model.eval()
    acc = evaluate(model, val_loader)
    acc_meter.update(acc)
    
    # Log smooth metrics
    print(f"Epoch {epoch}: loss={loss_meter.avg:.4f}, acc={acc_meter.avg:.3f}")
    
    # Reset for next epoch if desired
    # loss_meter.reset()
```

---

### ⚠️ Gotchas & Tips

| Issue | Solution |
|-------|----------|
| `classification_scores` missing FPR/FNR | Ensure `is_binary=True` and labels are strictly `{0, 1}` |
| `mean_ir_scores` recall = -1.0 | Provide `total_positives_per_query` for accurate recall calculation |
| Cosine sim outside [-1, 1] | Inputs must be finite; normalize with `emb / emb.norm(dim=-1, keepdim=True)` |
| Top-k accuracy with 2D `pred` | Ensure predictions are sorted best→worst before passing to `np_topk_accuracy` |
| MLflow plot logging fails | Call `cleanup_plots()` after `log_figure` to avoid tkinter/main-thread issues |

#### Pro Tips
```python
# Tip 1: Reuse Scorer for query-by-query debugging
for i, (true_id, pred_ids) in enumerate(zip(true_ids, pred_rankings)):
    scorer = Scorer(
        sorted_labels=[1 if pid == true_id else 0 for pid in pred_ids],
        k=10
    )
    if scorer.precision() < 0.5:
        print(f"Query {i} underperforming: P@10 = {scorer.precision():.2f}")

# Tip 2: Combine metrics for a single summary score
def retrieval_score(results: dict, weights: dict = None) -> float:
    """Weighted combination of IR metrics for model selection."""
    weights = weights or {"MAP-10": 0.5, "mRR": 0.3, "mP-1": 0.2}
    return sum(results.get(k, 0) * v for k, v in weights.items())

# Tip 3: Use AverageMeter for exponential moving average (optional extension)
class EMAMeter(AverageMeter):
    def __init__(self, beta=0.99):
        super().__init__()
        self.beta = beta
        self.ema = None
    
    def update(self, val, n=1):
        super().update(val, n)
        self.ema = val if self.ema is None else self.beta * self.ema + (1 - self.beta) * val
```

---

### 📦 Module Organization

```
metrics/
├── __init__.py      # jaccard_similarity
├── cosine.py        # torch-based cosine similarity
├── er.py            # Entity Resolution Ranking: classification + IR metrics + MLflow plots
└── ir.py            # Information Retrieval: DCG/NDCG variants + same core API
```

---

> 🍱 **Yumbox Philosophy**: Metrics should be *always reused, never redefined*. With this most extensive metrics suite available, it's much easier to evaluate your target task in different aspects.

Happy evaluating! 🚀 If your use case needs a new metric or a different averaging strategy, the code is intentionally modular — extend away.
