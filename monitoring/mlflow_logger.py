"""
MLflow logger — logs eval harness runs as MLflow experiments.

Experiment: medsignal_eval
Run name:   {baseline}_{timestamp}

Logged metrics (per baseline run):
  - faithfulness_mean
  - answer_relevancy_mean
  - retrieval_hit_rate
  - router_accuracy
  - avg_latency_ms
  - n_questions

Logged params:
  - baseline
  - golden_set_version (hash of golden_set.json)
  - model (Groq model used)

Logged artifacts:
  - Full eval results JSON

Usage:
    from monitoring.mlflow_logger import log_eval_run
    log_eval_run(summary_dict)

Or via CLI:
    mlflow ui  ← open http://localhost:5000 to view experiments
"""

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "medsignal_eval"
GOLDEN_SET_PATH = Path(__file__).parent.parent / "eval" / "golden_set.json"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def _golden_set_hash() -> str:
    """Short hash of golden_set.json for versioning."""
    try:
        content = GOLDEN_SET_PATH.read_bytes()
        return hashlib.md5(content).hexdigest()[:8]
    except Exception:
        return "unknown"


def log_eval_run(summary: dict) -> None:
    """
    Log a single eval harness run to MLflow.
    Non-blocking — logs warning and returns on any failure.
    """
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        run_name = f"{summary['baseline']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name):
            # Params
            mlflow.log_params({
                "baseline": summary["baseline"],
                "n_questions": summary["n_questions"],
                "golden_set_hash": _golden_set_hash(),
                "groq_model": "llama-3.3-70b-versatile",
            })

            # Metrics
            metrics = {
                "router_accuracy": summary.get("router_accuracy", 0.0),
                "retrieval_hit_rate": summary.get("retrieval_hit_rate", 0.0),
                "avg_latency_ms": summary.get("avg_latency_ms", 0.0),
            }
            if "faithfulness_mean" in summary:
                metrics["faithfulness_mean"] = summary["faithfulness_mean"]
            if "answer_relevancy_mean" in summary:
                metrics["answer_relevancy_mean"] = summary["answer_relevancy_mean"]

            mlflow.log_metrics(metrics)

            # Artifact — full results JSON (without embedding the full results list in params)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(summary, f, indent=2, default=str)
                tmp_path = f.name

            mlflow.log_artifact(tmp_path, artifact_path="eval_results")
            os.unlink(tmp_path)

            logger.info(f"MLflow: logged run '{run_name}' to experiment '{EXPERIMENT_NAME}'")

    except Exception as e:
        logger.warning(f"MLflow logging failed (non-fatal): {e}")


def get_best_run(metric: str = "faithfulness_mean") -> dict | None:
    """
    Return the MLflow run with the highest value of `metric`.
    Returns None if MLflow unavailable or no runs exist.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if not experiment:
            return None

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
        )
        if not runs:
            return None

        run = runs[0]
        return {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "metrics": run.data.metrics,
            "params": run.data.params,
        }

    except Exception as e:
        logger.warning(f"MLflow get_best_run failed: {e}")
        return None


def print_experiment_summary() -> None:
    """Print a table of all eval runs for this experiment."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if not experiment:
            print("No MLflow experiment found. Run the harness first.")
            return

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=20,
        )

        print(f"\n{'='*80}")
        print(f"MLflow Experiment: {EXPERIMENT_NAME}")
        print(f"{'='*80}")
        print(f"{'Run Name':<35} {'Baseline':<15} {'Faith':>7} {'AnsRel':>7} {'HitRate':>8} {'RouterAcc':>10}")
        print("-" * 80)

        for run in runs:
            m = run.data.metrics
            p = run.data.params
            print(
                f"{run.info.run_name[:34]:<35}"
                f" {p.get('baseline',''):<15}"
                f" {m.get('faithfulness_mean', 'N/A'):>7}"
                f" {m.get('answer_relevancy_mean', 'N/A'):>7}"
                f" {m.get('retrieval_hit_rate', 'N/A'):>8}"
                f" {m.get('router_accuracy', 'N/A'):>10}"
            )

    except Exception as e:
        logger.warning(f"MLflow summary failed: {e}")


if __name__ == "__main__":
    print_experiment_summary()
