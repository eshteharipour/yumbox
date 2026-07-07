"""MLflow helpers for tracking dataset-build metrics and comparing recent runs."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import mlflow
from mlflow import MlflowClient, entities

from .tools import set_tracking_uri

logger = logging.getLogger("YumBox")

DEFAULT_DATASET_EXPERIMENT = "ner_dataset_build"

_COMPARISON_METRICS: tuple[str, ...] = (
    "rows",
    "div_score",
    "div_ref_score",
    "dup_surface_pct",
    "dup_row_pct",
    "tok_gini",
    "tok_entropy",
    "tok_top10_pct",
    "fa_title_pct",
    "fa_token_pct",
    "mfr_gini",
    "mfr_top10_pct",
    "mfr_fa_pct",
    "mfr_en_pct",
    "cat_gini",
    "cat_top10_pct",
    "cat_fa_pct",
    "cat_en_pct",
    "pn_gini",
    "pn_fa_pct",
    "pn_en_pct",
    "short_pct",
    "medium_pct",
    "long_pct",
)

_METRIC_LABELS: dict[str, str] = {
    "rows": "rows",
    "div_score": "diversity score (abs, lower=better)",
    "div_ref_score": "diversity score vs valid (lower=better)",
    "dup_surface_pct": "duplicate surface %",
    "dup_row_pct": "duplicate row %",
    "tok_gini": "token Gini",
    "tok_entropy": "token entropy",
    "tok_top10_pct": "token top-10 %",
    "fa_title_pct": "Persian title %",
    "fa_token_pct": "Persian token %",
    "mfr_gini": "manufacturer Gini",
    "mfr_top10_pct": "manufacturer top-10 %",
    "mfr_fa_pct": "manufacturer Farsi %",
    "mfr_en_pct": "manufacturer English %",
    "cat_gini": "category Gini",
    "cat_top10_pct": "category top-10 %",
    "cat_fa_pct": "category Farsi %",
    "cat_en_pct": "category English %",
    "pn_gini": "part-number Gini",
    "pn_fa_pct": "part-number Farsi %",
    "pn_en_pct": "part-number English %",
    "short_pct": "short length %",
    "medium_pct": "medium length %",
    "long_pct": "long length %",
}


def _ensure_experiment(client: MlflowClient, experiment_name: str) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return client.create_experiment(experiment_name)
    return experiment.experiment_id


def _format_metric(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    if abs(number) >= 100:
        return f"{number:.0f}"
    if abs(number) >= 10:
        return f"{number:.1f}"
    return f"{number:.3f}"


def _run_label(run: entities.Run) -> str:
    name = run.data.tags.get("mlflow.runName") or run.info.run_id[:8]
    started = datetime.fromtimestamp(run.info.start_time / 1000.0)
    stamp = started.strftime("%m-%d %H:%M")
    profiles = run.data.params.get("profiles", "")
    if profiles and len(profiles) <= 28:
        return f"{stamp} {profiles}"
    return f"{stamp} {name}"


def _metrics_from_run(run: entities.Run) -> dict[str, float]:
    return {key: float(value) for key, value in run.data.metrics.items()}


def log_dataset_build_metrics(
    metrics: Mapping[str, float],
    *,
    params: Optional[Mapping[str, Any]] = None,
    tags: Optional[Mapping[str, str]] = None,
    tracking_uri: str | Path = "mlruns",
    experiment_name: str = DEFAULT_DATASET_EXPERIMENT,
    run_name: Optional[str] = None,
    artifacts: Optional[Sequence[tuple[str, str]]] = None,
) -> str:
    """
    Log final dataset-build metrics to MLflow.

    Returns the new run id.
    """
    set_tracking_uri(str(tracking_uri))
    client = MlflowClient()
    experiment_id = _ensure_experiment(client, experiment_name)

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
    ) as run:
        if params:
            for key, value in params.items():
                mlflow.log_param(key, value)
        if tags:
            mlflow.log_dict(dict(tags), "tags.json")
            for key, value in tags.items():
                mlflow.set_tag(key, value)
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))
        if artifacts:
            for local_path, artifact_path in artifacts:
                mlflow.log_artifact(local_path, artifact_path=artifact_path)
        run_id = run.info.run_id

    logger.info(
        "Logged dataset build metrics to MLflow experiment '%s' (run %s)",
        experiment_name,
        run_id,
    )
    return run_id


def fetch_recent_dataset_build_runs(
    *,
    tracking_uri: str | Path = "mlruns",
    experiment_name: str = DEFAULT_DATASET_EXPERIMENT,
    n: int = 5,
    exclude_run_id: Optional[str] = None,
) -> list[entities.Run]:
    """Return up to ``n`` most recent finished parent runs (newest first)."""
    set_tracking_uri(str(tracking_uri))
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        run_view_type=entities.ViewType.ACTIVE_ONLY,
        order_by=["start_time DESC"],
        max_results=max(n + 5, n),
    )
    runs = [run for run in runs if "mlflow.parentRunId" not in run.data.tags]
    if exclude_run_id:
        runs = [run for run in runs if run.info.run_id != exclude_run_id]
    return runs[:n]


def _run_column_label(run: entities.Run, *, highlight_run_id: Optional[str]) -> str:
    marker = "→" if highlight_run_id and run.info.run_id == highlight_run_id else ""
    return f"{marker}{_run_label(run)}"


def format_dataset_build_comparison_table(
    runs: Sequence[entities.Run],
    *,
    metrics: Sequence[str] = _COMPARISON_METRICS,
    highlight_run_id: Optional[str] = None,
) -> str:
    """Render a pivoted fixed-width table: metrics as rows, runs as columns."""
    if not runs:
        return "No previous dataset builds found in MLflow."

    run_metrics = [_metrics_from_run(run) for run in runs]
    metric_keys = [m for m in metrics if m in _METRIC_LABELS]
    # Omit reference-only score when no run logged it.
    if not any("div_ref_score" in metrics for metrics in run_metrics):
        metric_keys = [m for m in metric_keys if m != "div_ref_score"]

    run_labels = [
        _run_column_label(run, highlight_run_id=highlight_run_id) for run in runs
    ]
    metric_w = max(len("metric"), *(len(_METRIC_LABELS[k]) for k in metric_keys))
    col_w = max(
        10,
        *(len(label) for label in run_labels),
        *(
            len(_format_metric(run_metrics[i].get(key)))
            for i in range(len(runs))
            for key in metric_keys
        ),
    )

    header = f"{'metric':<{metric_w}}"
    for label in run_labels:
        header += f"  {label:>{col_w}}"
    sep = "-" * len(header)
    lines = [header, sep]

    for key in metric_keys:
        row = f"{_METRIC_LABELS[key]:<{metric_w}}"
        for metrics_for_run in run_metrics:
            row += f"  {_format_metric(metrics_for_run.get(key)):>{col_w}}"
        lines.append(row)

    lines.extend(
        [
            "",
            "Notes:",
            "  → marks the current run | duplicate surface % = same token string",
            "  duplicate row % = identical (tokens + labels) | diversity score =",
            "  diversity_tune --objective absolute (lower = more diverse)",
            "  diversity score vs valid = diversity_tune --objective reference",
        ]
    )
    return "\n".join(lines)


def print_dataset_build_comparison(
    *,
    tracking_uri: str | Path = "mlruns",
    experiment_name: str = DEFAULT_DATASET_EXPERIMENT,
    n: int = 5,
    highlight_run_id: Optional[str] = None,
    exclude_run_id: Optional[str] = None,
) -> None:
    """Log a table comparing the last ``n`` dataset builds (including current run)."""
    runs = fetch_recent_dataset_build_runs(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        n=n,
        exclude_run_id=exclude_run_id,
    )
    if highlight_run_id:
        client = MlflowClient()
        current = client.get_run(highlight_run_id)
        runs = [current] + [run for run in runs if run.info.run_id != highlight_run_id]
        runs = runs[:n]

    table = format_dataset_build_comparison_table(
        runs,
        highlight_run_id=highlight_run_id,
    )
    logger.info("Dataset build comparison (last %d runs):\n%s", len(runs), table)
