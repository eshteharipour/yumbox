## `mlflow/` — Experiment Tracking & Checkpoint Management 🧪📈

MLflow integration that eliminates or reduces the usual boilerplate: recursive config logging, multi-experiment analysis, print-friendly plots, and intelligent checkpoint cleanup. Built for rapid prototyping and production reproducibility.

---

### 🎯 Quick Start

```python
from omegaconf import OmegaConf
from yumbox.mlflow import log_config, log_scores_dict, set_tracking_uri

# 1. Set tracking URI (auto-creates directory)
set_tracking_uri("mlruns")  # relative to cwd, or use absolute path

# 2. Start a run and log everything
import mlflow
with mlflow.start_run(run_name="baseline_v1"):
    
    # Log nested config (OmegaConf dict) as params + YAML artifact
    cfg = OmegaConf.load("configs/experiment.yaml")
    log_config(cfg, as_artifact=True, as_params=True)
    
    # Log metrics dict with optional prefix/step
    scores = {"accuracy": 0.92, "f1": 0.89, "loss": 0.34}
    log_scores_dict(scores, name="val", step=10)
    # → logs: val_accuracy=0.92, val_f1=0.89, val_loss=0.34 at step 10
    
    # Log a matplotlib figure (auto-cleanup to avoid memory leaks)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [0.8, 0.85, 0.92])
    mlflow.log_figure(fig, "accuracy_curve.png")
    from yumbox.mlflow import cleanup_plots
    cleanup_plots()  # critical in training loops!
```

---

### 🧰 Core Functions Reference

#### Config & Parameter Logging

| Function | Purpose | Key Features |
|----------|---------|-------------|
| `log_params(cfg, prefix)` | Recursively log OmegaConf params | Handles nested dicts, auto-prefixes keys |
| `log_config(cfg, as_artifact, as_params)` | Log full config | Saves YAML artifact + flat params in one call |
| `log_scores_dict(scores, name, step)` | Log metrics dict | Optional prefix + step support for curves |

```python
# Nested config logging example
cfg = OmegaConf.create({
    "model": {"name": "bert", "layers": 12},
    "training": {"lr": 1e-4, "batch_size": 32}
})
log_params(cfg)
# → logs: model.name=bert, model.layers=12, training.lr=0.0001, training.batch_size=32

# With prefix for hierarchical organization
log_params(cfg, prefix="experiment")
# → logs: experiment.model.name=bert, experiment.training.lr=0.0001, ...
```

#### Run Querying & Filtering

```python
from yumbox.mlflow import get_mlflow_runs

# Get successful parent runs from an experiment
runs = get_mlflow_runs(
    experiment_name="image_classification",
    status="success",      # "success", "failed", or None
    level="parent",        # "parent", "child", or None
    filter="params.dataset = 'imagenet'"  # MLflow filter string
)

# Convenience helpers
from yumbox.mlflow import get_last_successful_run, get_last_run_failed

last_good = get_last_successful_run("my_experiment")  # Resume from best run
last_fail = get_last_run_failed("my_experiment")      # Debug failed run
```

##### Filter String Examples
```python
# Filter by params
filter="params.model_type = 'transformer' and params.lr < 0.001"

# Filter by metrics
filter="metrics.val_accuracy > 0.9 and metrics.loss < 0.5"

# Filter by tags
filter="tags.stage = 'production' and tags.team = 'nlp'"

# Combine with status/level in get_mlflow_runs()
```

#### Tracking URI Management

```python
from yumbox.mlflow import set_tracking_uri

# Absolute path → file:/absolute/path
set_tracking_uri("/shared/mlruns")

# Relative path → resolved relative to cwd
set_tracking_uri("mlruns")  # → file:/current/working/dir/mlruns

# Get current URI
current = mlflow.get_tracking_uri()
```

---

### 📊 Visualization & Analysis

#### Single Metric Across Runs
```python
from yumbox.mlflow import plot_metric_across_runs

# Plot 'val_loss' across all successful runs in an experiment
# Must be called within an active MLflow run to log the figure
with mlflow.start_run():
    plot_metric_across_runs(
        metric_key="val_loss",
        experiment_name="bert_finetuning",
        artifact_file="val_loss_comparison.png"
    )
```

#### Multi-Experiment Metric Comparison
```python
from yumbox.mlflow import plot_metric_across_experiments

# Compare 'f1-binary' across multiple experiments with print-friendly styling
plot_metric_across_experiments(
    experiment_names=["model_v1", "model_v2", "model_v3"],
    metric_key="f1-binary",
    y_metric="F1 Score",
    mode="epoch",  # or "step"
    legend_names=["Baseline", "+Augmentation", "+Ensemble"],
    artifact_file="f1_comparison.png",
    title="Model Iteration: F1 Score Progression",
    figsize=(12, 8),
    dpi=300,  # Print-quality output
    subsample_interval=50,  # Reduce point density for large runs
    marker_size=6
)
```

> 🎨 **Print-Friendly Styling**: Uses distinct line styles (`-`, `--`, `-.`, `:`) and markers (`o`, `s`, `^`, `D`) so plots remain distinguishable in black-and-white prints.

#### Advanced Experiment Analysis
```python
from yumbox.mlflow import process_experiment_metrics

# Aggregate metrics across experiments with optional plotting
df = process_experiment_metrics(
    storage_path="mlruns",
    select_metrics=["f1-binary", "precision-binary", "recall-binary", "loss"],
    sort_metric="f1-binary",  # Sort results by this metric
    legend_names=["Model A", "Model B"],
    y_metric="Score",
    metrics_to_mean=["precision-binary", "recall-binary"],  # Compute F1 from P/R
    mean_metric_name="f1-computed",
    aggregate_all_runs=True,  # Include all runs, not just latest
    run_mode="parent",  # Analyze parent runs only
    filter="params.dataset = 'valid'",  # Additional MLflow filter
    experiment_names=["exp_001", "exp_002"],  # Limit to specific experiments
    plot_mode="epoch",
    output_file="comparison_plot.png",  # Auto-generates and logs plot
    include_params=["learning_rate", "batch_size"]  # Include these params in output
)

# df now contains aggregated metrics ready for analysis/export
print(df.head())
```

#### Interactive Plotly Visualization
```python
from yumbox.mlflow import visualize_metrics

# Create interactive scatter plot (requires plotly + kaleido for static export)
visualize_metrics(
    df=df,  # From process_experiment_metrics
    x_metric="param_learning_rate",
    y_metric="f1-binary",
    color_metric="param_batch_size",  # Color by hyperparameter
    title="Hyperparameter Sweep: LR vs F1",
    theme="plotly_dark",  # or 'plotly', 'plotly_white'
    output_file="hyperparam_sweep.html"  # Interactive HTML, or .png/.pdf with kaleido
)
```

---

### 🗄️ Checkpoint Management (`checkpoint_helpers.py`)

Intelligent cleanup: keep only the checkpoints that matter based on MLflow metrics.

#### Quick Start: Analyze & Report
```python
from yumbox.mlflow.checkpoint_helpers import analyze_checkpoint_status, format_checkpoint_report

# Define which metrics indicate "best" (for keeping checkpoints)
metric_direction_map = {
    "val_loss": "min",           # Keep checkpoint with lowest val_loss
    "f1-binary": "max",          # Keep checkpoint with highest F1
    "epoch": "max",              # Always keep latest epoch
    r"test_.*_accuracy": "max",  # Regex: keep best on any test accuracy metric
}

# Metrics to ignore during analysis (e.g., intermediate logging)
ignore_metrics = {"step", "batch_loss", "lr"}

# Run analysis
keep_set, remove_set, deleted_set, non_ref_set, reasons = analyze_checkpoint_status(
    checkpoints_dir="/models/checkpoints",
    storage_path="mlruns",
    metric_direction_map=metric_direction_map,
    ignore_metrics=ignore_metrics
)

# Generate human-readable report
report = format_checkpoint_report(
    keep_set, remove_set, deleted_set, non_ref_set, reasons,
    checkpoints_dir="/models/checkpoints"
)
print(report)
```

#### Sample Report Output
```
================================================================================
CHECKPOINT ANALYSIS REPORT
================================================================================

SUMMARY:
  Total checkpoints to keep: 3
  MLflow-referenced checkpoints to remove: 12
  Already deleted checkpoints: 2
  Non-referenced checkpoints: 5

CHECKPOINTS TO KEEP:
----------------------------------------
  KEEP: epoch_10_best_f1.pt
        Reason: best_f1-binary_image_classification, last_run_image_classification
  KEEP: epoch_15_lowest_loss.pt
        Reason: best_val_loss_image_classification
  KEEP: epoch_20_latest.pt
        Reason: last_run_image_classification

MLFLOW-REFERENCED CHECKPOINTS TO REMOVE:
----------------------------------------
  REMOVE: epoch_01.pt
  REMOVE: epoch_02.pt
  ...

DISK USAGE ANALYSIS:
----------------------------------------
  Space from MLflow-referenced removals: 4.23 GB
  Space from non-referenced checkpoints: 1.87 GB
  Total potential space to free: 6.10 GB

================================================================================
```

#### Execute Cleanup (With Dry-Run Safety)
```python
from yumbox.mlflow.checkpoint_helpers import execute_checkpoint_removal

# First, always dry-run to verify
execute_checkpoint_removal(remove_set, dry_run=True)
# Output: "DRY RUN: Would remove: /path/to/epoch_01.pt"

# Then execute for real
execute_checkpoint_removal(remove_set, dry_run=False)
# Output: "Removed: /path/to/epoch_01.pt"
```

#### Advanced: Programmatic Checkpoint Selection
```python
from yumbox.mlflow.checkpoint_helpers import get_experiment_checkpoints

# Get checkpoints to keep + reasoning (for custom logic)
keep_checkpoints, reasons, all_referenced = get_experiment_checkpoints(
    storage_path="mlruns",
    metric_direction_map={"val_loss": "min", "f1": "max"},
    ignore_metrics={"step"}
)

# Custom logic: keep checkpoints that are either best OR from last 3 runs
from yumbox.mlflow import get_mlflow_runs
recent_runs = get_mlflow_runs("my_exp", status="success")[:3]
recent_checkpoints = {
    r.data.params["model_path"] 
    for r in recent_runs 
    if "model_path" in r.data.params
}

final_keep = keep_checkpoints | recent_checkpoints
```

---

### 🔄 Batch Experiment Execution

Run multiple configs sequentially with unified logging:

```python
from yumbox.mlflow import run_all_configs

# Execute all Git-committed YAML configs in configs/
run_all_configs(
    configs_dir="configs",
    ext=".yaml",
    mode="committed",  # "committed", "all", or "list"
    executable="python",
    script="train.py",
    config_arg="--config",  # How your script accepts config path
    extra_args=["--gpu", "0"],  # Additional CLI args for all runs
    config_mode="path",  # Pass full path or just filename
    disable_tqdm=True  # Avoid progress bar spam in logs
)
```

#### Config Discovery Helpers
```python
from yumbox.mlflow import get_configs, get_committed_configs

# Get all YAML files in directory
all_configs = get_configs("configs", ext=".yaml")

# Get only Git-tracked configs (ignore local experiments)
committed = get_committed_configs("configs", ext=".yaml")

# Natural sort for consistent ordering (1, 2, 10 not 1, 10, 2)
from yumbox.mlflow import natural_sorter
sorted_configs = natural_sorter(all_configs)
```

---

### 📤 Data Export & Analysis

#### Export All Runs to CSV (With JSON Flattening)
```python
from yumbox.mlflow import export_mlflow_data_with_flattening

# Export all experiments to a single CSV with flattened JSON columns
df = export_mlflow_data_with_flattening(
    storage_path="mlruns",
    output_file="all_experiments.csv"
)

# Now you can analyze with pandas:
import pandas as pd
df = pd.read_csv("all_experiments.csv")

# Filter and group
best_runs = df[df["metrics.val_accuracy"] > 0.9].groupby("experiment_name").first()
```

#### Find Best Metrics Across Experiments (Pandas-Only)
```python
from yumbox.mlflow import find_best_metrics

# Find best values for metrics across experiments (no MLflow API calls)
results = find_best_metrics(
    storage_path="mlruns",  # or path to exported CSV
    experiment_names=["exp_.*_bert"],  # Regex patterns supported
    metrics=["f1-.*", "val_loss"],  # Regex patterns for metric columns
    min_or_max=["max", "min"],  # Corresponding optimization direction
    run_mode="parent",  # Analyze parent runs only
    filter_string="params.dataset == 'valid'",  # Pandas query filter
    aggregation_mode="best_run"  # Keep only best run per metric (or "all_runs")
)

# results is a DataFrame with best values, run IDs, and metadata
print(results[["experiment_name", "target_metric", "metric_value", "run_id"]])
```

---

### 🔁 Common Patterns

#### Resume Training From Best Checkpoint
```python
from yumbox.mlflow import get_last_successful_run, set_tracking_uri
import mlflow

set_tracking_uri("mlruns")
last_run = get_last_successful_run("image_classification")

if last_run:
    # Get best checkpoint path from params
    best_ckpt = last_run.data.params.get("model_path")
    
    # Resume training
    with mlflow.start_run(run_name="resume_v2", parent_run_id=last_run.info.run_id):
        model = MyModel.load_from_checkpoint(best_ckpt)
        train(model, resume_epoch=last_run.data.metrics["epoch"] + 1)
```

#### Hyperparameter Sweep with Metric Aggregation
```python
from yumbox.mlflow import process_experiment_metrics, run_all_configs

# 1. Run all sweep configs
run_all_configs(
    configs_dir="configs/sweep",
    mode="all",
    script="train.py"
)

# 2. Aggregate results across all sweep runs
df = process_experiment_metrics(
    storage_path="mlruns",
    select_metrics=["val_loss", "f1-binary", "precision-binary"],
    sort_metric="f1-binary",
    legend_names=[f"LR={lr}" for lr in [1e-5, 1e-4, 1e-3]],
    y_metric="F1 Score",
    metrics_to_mean=["precision-binary", "recall-binary"],
    mean_metric_name="f1-computed",
    aggregate_all_runs=True,
    include_params=["learning_rate", "weight_decay"],
    output_file="sweep_results.png"
)

# 3. Export for external analysis
df.to_csv("sweep_results.csv", index=False)
```

#### Production Checkpoint Cleanup Cron Job
```python
# cleanup_checkpoints.py - Run weekly via cron
from yumbox.mlflow.checkpoint_helpers import (
    analyze_checkpoint_status,
    format_checkpoint_report,
    execute_checkpoint_removal
)

metric_direction_map = {
    "val_loss": "min",
    "f1-macro": "max",
    "epoch": "max",
}

keep, remove, deleted, non_ref, reasons = analyze_checkpoint_status(
    checkpoints_dir="/prod/models",
    storage_path="/prod/mlruns",
    metric_direction_map=metric_direction_map,
    ignore_metrics={"step", "batch_loss"}
)

report = format_checkpoint_report(keep, remove, deleted, non_ref, reasons, "/prod/models")

# Execute removal (dry_run=False for production)
execute_checkpoint_removal(remove, dry_run=False)
```

---

### ⚠️ Gotchas & Tips

| Issue | Solution |
|-------|----------|
| `cleanup_plots()` not called → memory leak in training loops | Always call `cleanup_plots()` after `mlflow.log_figure()` in loops |
| Nested OmegaConf params logged as strings, not numbers | MLflow params are always strings; cast in analysis: `float(run.data.params["lr"])` |
| Regex patterns in `find_best_metrics` too broad | Use word boundaries: `r"\bval_loss\b"` not `"val_loss"` to avoid matching `"my_val_loss_metric"` |
| Checkpoint paths in MLflow are relative, but analysis needs absolute | `get_experiment_checkpoints` auto-converts to absolute paths via `os.path.abspath` |
| `plot_metric_across_experiments` requires active MLflow run to log figure | Wrap plotting calls in `with mlflow.start_run():` or set `artifact_file` to save locally |

#### Pro Tips
```python
# Tip 1: Use tags for cross-cutting metadata (team, stage, priority)
with mlflow.start_run() as run:
    mlflow.set_tag("team", "nlp")
    mlflow.set_tag("stage", "experiment")
    mlflow.set_tag("priority", "high")
    
# Query later:
runs = get_mlflow_runs("my_exp", filter="tags.priority = 'high' and tags.stage = 'production'")

# Tip 2: Log model signature + input example for MLflow Model Registry
from mlflow.models import Signature, infer_signature
signature = infer_signature(X_sample, model.predict(X_sample))
mlflow.pytorch.log_model(model, "model", signature=signature, input_example=X_sample)

# Tip 3: Use run names for human-readable identification (not just UUIDs)
with mlflow.start_run(run_name=f"{model_name}_lr{lr}_batch{bs}"):
    ...
# Query by pattern:
runs = get_mlflow_runs("exp", filter="tags.mlflow.runName LIKE '%_lr0.0001_%'")
```

---

### 📦 Module Organization

```
mlflow/
├── __init__.py           # Core logging, querying, plotting, analysis
├── checkpoint_helpers.py # Intelligent checkpoint cleanup based on metrics
└── tools.py              # Utilities: flatten_json_columns for CSV export
```

> 💡 **Why separate checkpoint helpers?** Checkpoint management is a distinct concern from experiment tracking. Keeping it modular lets you use `log_config` without pulling in disk I/O logic, and vice versa.

---

### 🔐 Security & Best Practices

```python
# Never log secrets as params!
cfg = OmegaConf.load("config.yaml")
# ❌ Bad: logs API keys, passwords
# log_config(cfg)

# ✅ Good: redact sensitive fields first
def redact_secrets(cfg):
    sensitive = {"api_key", "password", "secret"}
    for key in sensitive:
        if key in cfg:
            cfg[key] = "***REDACTED***"
    return cfg

log_config(redact_secrets(cfg.copy()))
```

```python
# Use MLflow's built-in artifact encryption for sensitive models
mlflow.log_artifact("model.bin", artifact_path="encrypted")
# Then configure MLflow server with artifact encryption backend
```

---

> 🍱 **Yumbox Philosophy**: Experiment tracking should *enable* iteration, not hinder it. Routine tasks like finding your best run or checkpoint cleanup should be easily accomplished, otherwise you will feel drowned and distracted by these redundant tasks, instead of being focused on solving your ML problem.

Happy tracking! 🚀 If your workflow needs a new query pattern or plot style, the code is intentionally composable — extend away.
