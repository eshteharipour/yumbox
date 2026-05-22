## `scripts/` — CLI Tools for MLflow Analysis & Checkpoint Management 🛠️📊

Command-line utilities for analyzing MLflow experiments, visualizing metrics, managing checkpoints, and finding best-performing runs. Built for reproducible ML workflows and team collaboration.

---

### 🎯 Quick Start

```bash
# Install the CLI (if not already in your PATH)
pip install -e .  # From repo root

# Analyze experiment metrics with visualization
metrics-cli analyze \
  --storage-path ./mlruns \
  --select-metrics accuracy loss f1_score \
  --metrics-to-mean accuracy f1_score \
  --mean-metric-name avg_score \
  --sort-metric avg_score \
  --x-metric epoch \
  --y-metric avg_score \
  --output-plot results.html \
  --output-csv results.csv

# Compare training curves across experiments
metrics-cli compare-experiments \
  --storage-path ./mlruns \
  --experiment-names baseline_v1 baseline_v2 improved_model \
  --legend-names "Baseline" "Baseline+Aug" "Improved" \
  --metric val_loss \
  --mode epoch \
  --output-file loss_comparison.png \
  --title "Validation Loss: Model Iterations"

# Find and clean up suboptimal checkpoints (DRY RUN FIRST!)
metrics-cli manage-checkpoints \
  --checkpoints-dir ./models \
  --storage-path ./mlruns \
  --custom-metrics "custom_f1:max" "perplexity:min" \
  --dry-run \
  --output-report cleanup_plan.txt

# Find best-performing runs across experiments
metrics-cli best-metrics \
  --storage-path ./mlruns \
  --experiment-names exp_.*_bert \
  --metrics accuracy f1_score loss \
  --min-or-max max max min \
  --output-csv best_runs.csv

# Get help
metrics-cli help
metrics-cli help details --cmd analyze
metrics-cli help patterns
```

---

### 🧰 Command Reference

#### `analyze` — Process & Visualize Experiment Metrics

```bash
metrics-cli analyze [OPTIONS]
```

**Purpose**: Aggregate metrics from MLflow runs, calculate derived metrics (like mean F1 from precision/recall), and generate interactive or static visualizations.

##### Required Arguments
| Argument | Type | Purpose | Example |
|----------|------|---------|---------|
| `--storage-path` | `str` | Path to MLflow tracking directory | `./mlruns`, `/shared/mlflow` |
| `--select-metrics` | `list[str]` | Metrics to collect from runs | `accuracy loss f1_score` |

##### Optional Arguments
| Argument | Type | Default | Purpose |
|----------|------|---------|---------|
| `--metrics-to-mean` | `list[str]` | `[]` | Metrics to average into a single derived metric |
| `--mean-metric-name` | `str` | — | Name for the derived mean metric |
| `--sort-metric` | `str` | — | Metric to sort output table by |
| `--aggregate-all-runs` | `flag` | `False` | Process all runs vs. only latest per experiment |
| `--run-mode` | `parent\|children\|both` | `both` | Filter by run hierarchy level |
| `--filter` | `str` | — | MLflow filter string (e.g., `"params.lr < 0.01"`) |
| `--output-csv` | `str` | — | Path to save results as CSV |
| `--output-plot` | `str` | — | Path to save Plotly visualization (`.html`, `.png`, etc.) |
| `--x-metric` / `--y-metric` | `str` | — | Axes for scatter plot visualization |
| `--color-metric` | `str` | — | Metric for color scale in scatter plot |
| `--title` / `--theme` | `str` | `"Experiment Metrics..."` / `"plotly_dark"` | Plot title and Plotly theme |
| `--experiment-names` | `list[str]` | — | Limit analysis to specific experiments |
| `--legend-names` | `list[str]` | — | Custom labels for legend (matches `experiment-names` order) |
| `--include-params` | `list[str]` | — | Include these hyperparameters in output table |

##### Example: Full Workflow
```bash
metrics-cli analyze \
  --storage-path ./mlruns \
  --select-metrics precision recall f1_score val_loss \
  --metrics-to-mean precision recall \
  --mean-metric-name f1_computed \
  --sort-metric f1_computed \
  --aggregate-all-runs \
  --run-mode parent \
  --filter "params.dataset = 'valid' and params.model_type != 'baseline'" \
  --x-metric epoch \
  --y-metric f1_computed \
  --color-metric val_loss \
  --title "Model Performance: F1 vs Epoch" \
  --theme plotly_white \
  --experiment-names exp_bert_v1 exp_bert_v2 exp_roberta \
  --legend-names "BERT Base" "BERT Large" "RoBERTa" \
  --include-params learning_rate batch_size weight_decay \
  --output-csv analysis_results.csv \
  --output-plot performance_scatter.html
```

##### Output
```
✅ Processed 47 runs across 3 experiments
📊 Sorted by: f1_computed (descending)

| experiment_name | run_id | f1_computed | val_loss | param_learning_rate | ... |
|-----------------|--------|-------------|----------|---------------------|-----|
| exp_roberta     | abc123 | 0.942       | 0.18     | 0.0001              | ... |
| exp_bert_v2     | def456 | 0.921       | 0.21     | 0.0002              | ... |
| ...

📈 Interactive plot saved to: performance_scatter.html
📄 CSV saved to: analysis_results.csv
```

---

#### `compare-experiments` — Print-Friendly Metric Comparison

```bash
metrics-cli compare-experiments [OPTIONS]
```

**Purpose**: Generate publication-ready line plots comparing a single metric across multiple experiments, with distinct line styles for black-and-white printing.

##### Required Arguments
| Argument | Type | Purpose |
|----------|------|---------|
| `--storage-path` | `str` | MLflow tracking directory |
| `--experiment-names` | `list[str]` | Experiments to compare |
| `--metric` | `str` | Single metric to plot (e.g., `val_loss`, `accuracy`) |

##### Optional Arguments
| Argument | Type | Default | Purpose |
|----------|------|---------|---------|
| `--legend-names` | `list[str]` | — | Custom legend labels (must match `experiment-names` length) |
| `--mode` | `step\|epoch` | `step` | X-axis: training steps or normalized epochs |
| `--output-file` | `str` | — | Save plot to file (`.png`, `.pdf`, etc.) |
| `--title` | `str` | — | Custom plot title |
| `--figsize` | `float float` | `10 6` | Figure dimensions in inches |
| `--dpi` | `int` | `300` | Output resolution for print quality |
| `--y-metric` | `str` | — | Y-axis label override (defaults to `--metric`) |

##### Example: Publication-Ready Comparison
```bash
metrics-cli compare-experiments \
  --storage-path ./mlruns \
  --experiment-names cnn_baseline resnet_transfer vit_finetuned \
  --legend-names "CNN Baseline" "ResNet Transfer" "ViT Fine-tuned" \
  --metric val_accuracy \
  --mode epoch \
  --output-file accuracy_comparison.pdf \
  --title "Validation Accuracy: Architecture Comparison" \
  --figsize 14 8 \
  --dpi 600 \
  --y-metric "Accuracy (%)"
```

##### Output Style
```
📈 Experiment comparison plot generated for metric 'val_accuracy'
📊 Experiments compared: cnn_baseline, resnet_transfer, vit_finetuned
🎨 Legend names: CNN Baseline, ResNet Transfer, ViT Fine-tuned
💾 Plot saved to: accuracy_comparison.pdf (600 DPI, print-ready)
```

> 🎨 **Print-Friendly Design**: Uses distinct line styles (`-`, `--`, `-.`, `:`) and markers (`○`, `□`, `△`, `◇`) so plots remain distinguishable when printed in black-and-white.

---

#### `manage-checkpoints` — Intelligent Checkpoint Cleanup

```bash
metrics-cli manage-checkpoints [OPTIONS]
```

**Purpose**: Analyze MLflow metrics to identify which model checkpoints to keep (best performance) vs. remove (suboptimal), with safety features like dry-run mode.

##### Required Arguments
| Argument | Type | Purpose |
|----------|------|---------|
| `--checkpoints-dir` | `str` | Directory containing `.pt`, `.pth`, `.ckpt` files |
| `--storage-path` | `str` | MLflow tracking directory |

##### Optional Arguments
| Argument | Type | Default | Purpose |
|----------|------|---------|---------|
| `--custom-metrics` | `list[str]` | — | Custom metric direction: `"metric_name:min"` or `"metric_name:max"` |
| `--ignore-metrics` | `list[str]` | `["lr", "step"]` | Metrics to exclude from analysis |
| `--output-report` | `str` | — | Save analysis report to text file |
| `--dry-run` | `flag` | `False` | Show what would be removed without deleting |
| `--remove` | `flag` | `False` | Actually delete suggested checkpoints (⚠️ irreversible) |

##### Default Metric Directions
The command includes sensible defaults for common metrics:

```python
# Built-in metric direction mapping
{
    # Minimize these (lower is better)
    "loss": "min", "error": "min", "mse": "min", "rmse": "min", 
    "mae": "min", "perplexity": "min", "neg_dist": "min",
    
    # Maximize these (higher is better)
    "accuracy": "max", "acc": "max", "precision": "max", 
    "recall": "max", "f1": "max", "auc": "max", "dice": "max", 
    "iou": "max", "bleu": "max", "rouge": "max", "pr_auc": "max", 
    "roc_auc": "max", "pos_dist": "max",
}
```

##### Example: Safe Checkpoint Cleanup
```bash
# Step 1: Analyze with dry-run (RECOMMENDED FIRST STEP)
metrics-cli manage-checkpoints \
  --checkpoints-dir ./models/checkpoints \
  --storage-path ./mlruns \
  --custom-metrics "custom_f1:max" "domain_loss:min" \
  --ignore-metrics epoch lr_scheduler_step \
  --output-report cleanup_plan.txt \
  --dry-run

# Review the report:
cat cleanup_plan.txt
# → Shows which checkpoints would be kept/removed and why

# Step 2: Execute cleanup (only after verifying dry-run output)
metrics-cli manage-checkpoints \
  --checkpoints-dir ./models/checkpoints \
  --storage-path ./mlruns \
  --remove  # Actually deletes files
```

##### Sample Report Output
```
================================================================================
CHECKPOINT ANALYSIS REPORT
================================================================================

SUMMARY:
  Total checkpoints to keep: 4
  MLflow-referenced checkpoints to remove: 23
  Already deleted checkpoints: 1
  Non-referenced checkpoints: 7

CHECKPOINTS TO KEEP:
----------------------------------------
  KEEP: epoch_15_best_f1.pt
        Reason: best_custom_f1_exp_nlp, last_run_exp_nlp
  KEEP: epoch_20_lowest_loss.pt
        Reason: best_domain_loss_exp_nlp
  ...

MLFLOW-REFERENCED CHECKPOINTS TO REMOVE:
----------------------------------------
  REMOVE: epoch_01.pt
  REMOVE: epoch_02.pt
  ...

DISK USAGE ANALYSIS:
----------------------------------------
  Space from MLflow-referenced removals: 8.42 GB
  Space from non-referenced checkpoints: 2.15 GB
  Total potential space to free: 10.57 GB

================================================================================
```

---

#### `best-metrics` — Find Top-Performing Runs

```bash
metrics-cli best-metrics [OPTIONS]
```

**Purpose**: Search across experiments to find runs with optimal metric values, supporting multi-metric optimization and regex pattern matching.

##### Required Arguments
| Argument | Type | Purpose |
|----------|------|---------|
| `--storage-path` | `str` | MLflow tracking directory |
| `--experiment-names` | `list[str]` | Experiments to search (supports regex: `exp_.*_bert`) |
| `--metrics` | `list[str]` | Metrics to optimize (supports regex: `f1.*`, `val_.*_acc`) |
| `--min-or-max` | `list[str]` | Optimization direction for each metric: `min` or `max` |

##### Optional Arguments
| Argument | Type | Default | Purpose |
|----------|------|---------|---------|
| `--run-mode` | `parent\|children\|both` | `parent` | Filter by run hierarchy |
| `--filter` | `str` | — | MLflow filter string for additional filtering |
| `--aggregation-mode` | `all_runs\|best_run` | `all_runs` | Return all runs or only best per metric |
| `--output-csv` | `str` | — | Save results to CSV |

##### Example: Multi-Metric Leaderboard
```bash
metrics-cli best-metrics \
  --storage-path ./mlruns \
  --experiment-names "exp_.*_classification" \
  --metrics "f1.*" "precision.*" "val_loss" "inference_time" \
  --min-or-max max max min min \
  --run-mode parent \
  --filter "params.dataset = 'test' and tags.stage = 'production'" \
  --aggregation-mode best_run \
  --output-csv production_leaderboard.csv
```

##### Output
```
Best Metrics Results:
================================================================================================
| experiment_name     | run_id | target_metric      | metric_value | param_learning_rate | ... |
|---------------------|--------|--------------------|--------------|---------------------|-----|
| exp_bert_clf_v1     | abc123 | f1_macro           | 0.942        | 0.0001              | ... |
| exp_roberta_clf_v2  | def456 | precision_weighted | 0.938        | 0.0002              | ... |
| ...                 | ...    | ...                | ...          | ...                 | ... |

Summary:
- Total experiments processed: 12
- Total best values found: 4

📄 Results saved to production_leaderboard.csv
```

---

#### `help` — Built-in Documentation

```bash
metrics-cli help [list|details|quick-start|patterns|troubleshooting] [--cmd COMMAND]
```

**Purpose**: Display comprehensive, contextual help without leaving the terminal.

##### Help Types
| Type | Purpose | Example |
|------|---------|---------|
| `list` | Show all commands with brief descriptions | `metrics-cli help` |
| `details` | Show detailed info for a specific command | `metrics-cli help details --cmd analyze` |
| `quick-start` | Getting started guide for new users | `metrics-cli help quick-start` |
| `patterns` | Common usage patterns and workflows | `metrics-cli help patterns` |
| `troubleshooting` | Common issues and solutions | `metrics-cli help troubleshooting` |

##### Example Output
```bash
$ metrics-cli help patterns

============================================================
  COMMON USAGE PATTERNS
============================================================

🔍 EXPLORATORY ANALYSIS
   Start by analyzing your experiments to understand the data
   Command: metrics-cli analyze --storage-path ./mlflow --select-metrics accuracy loss ...

📊 MODEL COMPARISON
   Compare training curves across different model architectures
   Command: metrics-cli compare-experiments --storage-path ./mlflow ...

🧹 CHECKPOINT CLEANUP
   Clean up disk space by removing suboptimal checkpoints
   Command: metrics-cli manage-checkpoints --checkpoints-dir ./models ...

🏆 LEADERBOARD CREATION
   Find the best performing models across experiments
   Command: metrics-cli best-metrics --storage-path ./mlflow ...

🎯 FILTERED ANALYSIS
   Analyze only specific runs matching certain criteria
   Command: metrics-cli analyze ... --filter "params.learning_rate > 0.001"
```

---

### 🔁 Common Patterns

#### Automated Experiment Reporting Pipeline
```bash
#!/bin/bash
# scripts/run_analysis.sh

STORAGE="./mlruns"
OUTPUT_DIR="./reports/$(date +%Y-%m-%d)"
mkdir -p "$OUTPUT_DIR"

# 1. Analyze all classification experiments
metrics-cli analyze \
  --storage-path "$STORAGE" \
  --select-metrics accuracy f1_score precision recall val_loss \
  --metrics-to-mean precision recall \
  --mean-metric-name f1_computed \
  --sort-metric f1_computed \
  --aggregate-all-runs \
  --run-mode parent \
  --filter "params.task = 'classification'" \
  --x-metric epoch \
  --y-metric f1_computed \
  --output-csv "$OUTPUT_DIR/classification_analysis.csv" \
  --output-plot "$OUTPUT_DIR/classification_scatter.html"

# 2. Compare top 3 models side-by-side
metrics-cli compare-experiments \
  --storage-path "$STORAGE" \
  --experiment-names bert_base bert_large roberta_base \
  --legend-names "BERT Base" "BERT Large" "RoBERTa Base" \
  --metric val_loss \
  --mode epoch \
  --output-file "$OUTPUT_DIR/loss_comparison.png" \
  --title "Validation Loss: Transformer Comparison" \
  --figsize 14 8 \
  --dpi 300

# 3. Find best models for deployment
metrics-cli best-metrics \
  --storage-path "$STORAGE" \
  --experiment-names "exp_.*_prod" \
  --metrics f1_score inference_time model_size_mb \
  --min-or-max max min min \
  --output-csv "$OUTPUT_DIR/deployment_candidates.csv"

# 4. Clean up old checkpoints (dry-run first!)
metrics-cli manage-checkpoints \
  --checkpoints-dir "./models" \
  --storage-path "$STORAGE" \
  --output-report "$OUTPUT_DIR/cleanup_plan.txt" \
  --dry-run

echo "✅ Analysis complete. Reports saved to $OUTPUT_DIR"
```

#### CI/CD Integration: Fail on Metric Regression
```yaml
# .github/workflows/metric-check.yml
name: Metric Regression Check
on: [pull_request]

jobs:
  check-metrics:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -e .
      - name: Run metric analysis
        run: |
          metrics-cli best-metrics \
            --storage-path ./mlruns \
            --experiment-names main_branch \
            --metrics val_accuracy \
            --min-or-max max \
            --output-csv main_best.csv
      
      - name: Compare with PR branch
        run: |
          metrics-cli best-metrics \
            --storage-path ./mlruns \
            --experiment-names pr_${{ github.event.pull_request.number }} \
            --metrics val_accuracy \
            --min-or-max max \
            --output-csv pr_best.csv
      
      - name: Check for regression
        run: |
          python scripts/check_regression.py \
            --baseline main_best.csv \
            --candidate pr_best.csv \
            --metric val_accuracy \
            --threshold 0.01  # Fail if PR is >1% worse
```

#### Scheduled Checkpoint Cleanup (Cron)
```bash
# /etc/cron.d/yumbox-cleanup
# Run checkpoint cleanup every Sunday at 2 AM
0 2 * * 0 user /path/to/scripts/weekly_cleanup.sh

# scripts/weekly_cleanup.sh
#!/bin/bash
set -e

STORAGE="/prod/mlruns"
CHECKPOINTS="/prod/models"
REPORT="/var/log/yumbox/cleanup_$(date +%Y-%m-%d).txt"

# Analyze and generate report
metrics-cli manage-checkpoints \
  --checkpoints-dir "$CHECKPOINTS" \
  --storage-path "$STORAGE" \
  --output-report "$REPORT" \
  --dry-run

# If report shows >10GB savings, execute cleanup
SAVINGS=$(grep "Total potential space to free" "$REPORT" | grep -oP '[\d.]+')
if (( $(echo "$SAVINGS > 10" | bc -l) )); then
  echo "⚠️  Large cleanup detected ($SAVINGS GB)."
  metrics-cli manage-checkpoints \
    --checkpoints-dir "$CHECKPOINTS" \
    --storage-path "$STORAGE" \
    --remove
  echo "✅ Cleanup completed. Report: $REPORT"
fi
```

#### Interactive Jupyter Integration
```python
# In a Jupyter notebook
import subprocess
import pandas as pd
from IPython.display import HTML, display

def run_metrics_cli(command: str, **kwargs) -> pd.DataFrame:
    """Run metrics-cli and return results as DataFrame."""
    cmd = ["metrics-cli", command]
    for k, v in kwargs.items():
        if isinstance(v, bool) and v:
            cmd.append(f"--{k.replace('_', '-')}")
        elif v is not None:
            if isinstance(v, list):
                for item in v:
                    cmd.extend([f"--{k.replace('_', '-')}", str(item)])
            else:
                cmd.extend([f"--{k.replace('_', '-')}", str(v)])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"CLI failed: {result.stderr}")
    
    # Parse CSV output if generated
    if "--output-csv" in kwargs and kwargs["--output-csv"]:
        return pd.read_csv(kwargs["--output-csv"])
    return None

# Usage in notebook
df = run_metrics_cli(
    "analyze",
    storage_path="./mlruns",
    select_metrics=["accuracy", "f1_score"],
    metrics_to_mean=["accuracy", "f1_score"],
    mean_metric_name="avg_score",
    sort_metric="avg_score",
    output_csv="/tmp/results.csv"
)

display(df.head())
```

---

### ⚠️ Gotchas & Tips

| Issue | Solution |
|-------|----------|
| `--metrics-to-mean` must be subset of `--select-metrics` | Ensure all mean metrics are also selected; CLI doesn't auto-add them |
| `--legend-names` length must match `--experiment-names` | Mismatch causes silent mislabeling; CLI now validates this |
| Checkpoint cleanup is irreversible | Always use `--dry-run` first and review the report before `--remove` |
| Regex patterns in `--experiment-names`/`--metrics` use Python `re` syntax | Use `exp_.*_bert` not `exp_*_bert`; test patterns with `re.match()` first |
| Plotly interactive plots require browser; static exports need `kaleido` | Install with `pip install kaleido` for PNG/PDF export; use `.html` for interactive |
| MLflow filter strings use SQL-like syntax | Use `"params.lr < 0.01 and tags.stage = 'prod'"`; escape quotes in shell |
| `--aggregate-all-runs` can produce large outputs | Use `--filter` to narrow scope before enabling this flag |

#### Pro Tips
```bash
# Tip 1: Chain commands with xargs for batch processing
# Find all experiments, then analyze each
metrics-cli best-metrics \
  --storage-path ./mlruns \
  --experiment-names ".*" \
  --metrics val_accuracy \
  --min-or-max max \
  --output-csv /tmp/all_best.csv | \
  cut -d, -f1 | tail -n +2 | \
  xargs -I {} metrics-cli analyze \
    --storage-path ./mlruns \
    --experiment-names {} \
    --select-metrics accuracy loss \
    --output-csv "reports/{}_analysis.csv"

# Tip 2: Use environment variables for common paths
export YUMBOX_MLFLOW="./mlruns"
export YUMBOX_CHECKPOINTS="./models"

# Then in commands:
metrics-cli analyze --storage-path $YUMBOX_MLFLOW ...
metrics-cli manage-checkpoints --checkpoints-dir $YUMBOX_CHECKPOINTS ...

# Tip 3: Create aliases for frequent commands
# In ~/.bashrc or ~/.zshrc:
alias yum-analyze='metrics-cli analyze --storage-path ./mlruns --aggregate-all-runs'
alias yum-compare='metrics-cli compare-experiments --storage-path ./mlruns --mode epoch --dpi 300'
alias yum-cleanup='metrics-cli manage-checkpoints --checkpoints-dir ./models --storage-path ./mlruns --dry-run'

# Tip 4: Pipe output to jq for JSON processing (if you add --output-json)
metrics-cli best-metrics ... --output-csv - | csvtool format '%(1)s,%(3)s\n' - | jq -R -s -c 'split("\n") | map(split(",") | {exp: .[0], metric: .[1], value: .[2]})'
```

---

### 🔐 Security & Best Practices

```bash
# Never run CLI with untrusted storage paths
# Validate paths before execution
if [[ ! "$STORAGE_PATH" =~ ^/safe/base/ ]]; then
  echo "❌ Storage path must be under /safe/base/"
  exit 1
fi

# Log all CLI executions for audit trails
LOG_FILE="/var/log/yumbox/cli.log"
echo "$(date) $(whoami) metrics-cli $*" >> "$LOG_FILE"
```

```python
# scripts/validate_cli_input.py
import re
import sys
from pathlib import Path

def validate_path(path: str, must_exist: bool = False) -> Path:
    """Validate and normalize a path argument."""
    p = Path(path).resolve()
    
    # Prevent path traversal
    if ".." in str(p):
        raise ValueError(f"Path traversal detected: {path}")
    
    # Restrict to allowed base directories
    allowed_bases = [Path("/safe/mlflow"), Path("/safe/models")]
    if not any(str(p).startswith(str(base)) for base in allowed_bases):
        raise ValueError(f"Path not in allowed directories: {p}")
    
    if must_exist and not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    
    return p

def validate_metric_direction(metric: str, direction: str) -> bool:
    """Validate metric direction specification."""
    if direction not in ("min", "max"):
        return False
    # Optional: check metric name format
    if not re.match(r'^[a-zA-Z0-9_.-]+$', metric):
        return False
    return True

# Usage in CLI wrapper
if __name__ == "__main__":
    try:
        storage = validate_path(sys.argv[1], must_exist=True)
        # ... proceed with validated inputs
    except Exception as e:
        print(f"❌ Input validation failed: {e}", file=sys.stderr)
        sys.exit(1)
```

---

### 📦 Module Organization

```
scripts/
├── cli.py               # Main CLI entry point with argparse subcommands
├── metric_cli_helper.py # Help system: command details, patterns, troubleshooting
└── generate_docs.py     # Auto-generate CLI documentation from --help output
```

> 💡 **Why a separate helper module?** The help system is complex enough to warrant its own file. `metric_cli_helper.py` can be run standalone (`python metric_cli_helper.py patterns`) or integrated into the main CLI (`metrics-cli help patterns`), providing flexibility for both interactive and scripted use.

---

Happy analyzing! 🚀 If your workflow needs a new subcommand or output format, the argparse structure is intentionally modular — extend away.
