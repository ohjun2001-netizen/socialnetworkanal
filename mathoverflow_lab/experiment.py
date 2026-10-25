"""Run chronological MathOverflow recurrence prediction experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import dataset_summary, load_events
from .evaluation import balanced_repeats, binary_metrics, choose_threshold, percentile_rank
from .features import (
    FEATURE_NAMES, HAWKES_FEATURES, HISTORY_FEATURES, SNA_FEATURES,
    build_examples, load_examples, save_examples,
)
from .gnn import train_and_score

ROOT = Path(__file__).resolve().parents[1]


def _split(metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    cutoffs = np.sort(metadata.cutoff.unique())
    val_start = cutoffs[int(len(cutoffs) * 0.60)]
    test_start = cutoffs[int(len(cutoffs) * 0.80)]
    train = metadata.cutoff.to_numpy() < val_start
    validation = (metadata.cutoff.to_numpy() >= val_start) & (metadata.cutoff.to_numpy() < test_start)
    test = metadata.cutoff.to_numpy() >= test_start
    return train, validation, test, {
        "train_windows": int(len(np.unique(metadata.loc[train, "cutoff"]))),
        "validation_windows": int(len(np.unique(metadata.loc[validation, "cutoff"]))),
        "test_windows": int(len(np.unique(metadata.loc[test, "cutoff"]))),
        "validation_start": int(val_start), "test_start": int(test_start),
    }


def _fit_family(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    columns: list[int],
    seed: int,
) -> tuple[HistGradientBoostingClassifier, dict]:
    configurations = [
        {"max_iter": 180, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 2.0},
        {"max_iter": 260, "learning_rate": 0.03, "max_leaf_nodes": 31, "min_samples_leaf": 30, "l2_regularization": 6.0},
        {"max_iter": 220, "learning_rate": 0.04, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 10.0},
    ]
    candidates = []
    for config_index, config in enumerate(configurations):
        for positive_weight in (1.0, 2.0, 4.0):
            weights = np.where(y[train] == 1, positive_weight, 1.0)
            model = HistGradientBoostingClassifier(
                **config, random_state=seed + config_index, early_stopping=False,
            ).fit(x[train][:, columns], y[train], sample_weight=weights)
            score = model.predict_proba(x[validation][:, columns])[:, 1]
            candidates.append((
                average_precision_score(y[validation], score),
                roc_auc_score(y[validation], score), model, config, positive_weight,
            ))
    ap, auc, model, config, weight = max(candidates, key=lambda item: (item[0], item[1]))
    return model, {"validation_ap": float(ap), "validation_auc": float(auc), "positive_weight": weight, **config}


def _plot_metrics(actual: pd.DataFrame, balanced: pd.DataFrame, result_dir: Path) -> None:
    order = actual.sort_values("average_precision", ascending=False).model.tolist()
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    actual.set_index("model").loc[order, ["average_precision", "roc_auc", "f1"]].plot.bar(ax=axes[0])
    axes[0].set_title("Actual candidate distribution (test)")
    axes[0].set_ylim(0, 1); axes[0].grid(axis="y", alpha=.25); axes[0].tick_params(axis="x", rotation=25)
    means = balanced.groupby("model")[["average_precision", "roc_auc", "f1"]].mean().loc[order]
    means.plot.bar(ax=axes[1])
    axes[1].set_title("Balanced 1:1 evaluation (100 repeats)")
    axes[1].set_ylim(0, 1); axes[1].grid(axis="y", alpha=.25); axes[1].tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(result_dir / "model_metrics.png", dpi=180)
    plt.close(figure)


def run(*, rebuild_features: bool = False, repeats: int = 100) -> Path:
    result_dir = ROOT / "results"
    model_dir = ROOT / "models"
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    raw = ROOT / "data" / "raw"
    events, users = load_events(raw)
    cache = result_dir / "examples_4h_to_24h.npz"
    if rebuild_features or not cache.exists():
        metadata, x = build_examples(events, lookback_hours=4, horizon_hours=24)
        save_examples(cache, metadata, x)
    else:
        metadata, x = load_examples(cache)
    y = metadata.label.to_numpy(dtype=np.uint8)
    train, validation, test, split_summary = _split(metadata)

    families = {
        "History HGB": [FEATURE_NAMES.index(name) for name in HISTORY_FEATURES],
        "SNA HGB": [FEATURE_NAMES.index(name) for name in SNA_FEATURES],
        "Hawkes HGB": [FEATURE_NAMES.index(name) for name in HAWKES_FEATURES],
        "Integrated HGB": list(range(len(FEATURE_NAMES))),
    }
    scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    tuning, fitted_models = {}, {}
    for model_number, (name, columns) in enumerate(families.items()):
        print(f"training: {name}")
        model, selected = _fit_family(x, y, train, validation, columns, 2026 + model_number * 100)
        scores[name] = (
            model.predict_proba(x[validation][:, columns])[:, 1],
            model.predict_proba(x[test][:, columns])[:, 1],
        )
        tuning[name] = selected
        fitted_models[name] = {"model": model, "columns": columns, "feature_names": [FEATURE_NAMES[i] for i in columns]}

    pair_frequency = x[:, FEATURE_NAMES.index("pair_count_lifetime")]
    recency = -x[:, FEATURE_NAMES.index("hours_since_pair_event")]
    scores["Pair Frequency"] = (percentile_rank(pair_frequency[validation]), percentile_rank(pair_frequency[test]))
    scores["Recency"] = (percentile_rank(recency[validation]), percentile_rank(recency[test]))

    print("training: Static GAE (GPU when available)")
    node_count = len(users)
    validation_codes = metadata.loc[validation, "pair_code"].to_numpy(np.uint64)
    test_codes = metadata.loc[test, "pair_code"].to_numpy(np.uint64)
    validation_gae = train_and_score(
        events, split_summary["validation_start"], validation_codes, node_count, seed=4026
    )
    test_gae = train_and_score(
        events, split_summary["test_start"], test_codes, node_count, seed=5026
    )
    scores["Static GAE"] = (percentile_rank(validation_gae.scores), percentile_rank(test_gae.scores))
    pd.DataFrame({
        "split": ["validation"] * len(validation_gae.losses) + ["test"] * len(test_gae.losses),
        "epoch": list(range(1, len(validation_gae.losses) + 1)) + list(range(1, len(test_gae.losses) + 1)),
        "loss": validation_gae.losses + test_gae.losses,
    }).to_csv(result_dir / "gae_training_loss.csv", index=False)

    actual_rows, balanced_rows, prediction = [], [], metadata.loc[test].copy()
    for name, (validation_score, test_score) in scores.items():
        threshold = choose_threshold(y[validation], validation_score)
        actual_rows.append({"model": name, **binary_metrics(y[test], test_score, threshold)})
        for row in balanced_repeats(y[validation], validation_score, y[test], test_score, repeats=repeats):
            balanced_rows.append({"model": name, **row})
        prediction[name] = test_score
    actual = pd.DataFrame(actual_rows).sort_values("average_precision", ascending=False)
    balanced = pd.DataFrame(balanced_rows)
    actual.to_csv(result_dir / "actual_test_metrics.csv", index=False)
    balanced.to_csv(result_dir / "balanced_metrics_repeats.csv", index=False)
    prediction.to_csv(result_dir / "test_predictions.csv", index=False)

    best_name = actual.iloc[0].model
    if best_name in fitted_models:
        joblib.dump(fitted_models[best_name], model_dir / "best_model.joblib")
        fitted = fitted_models[best_name]
        importance = permutation_importance(
            fitted["model"], x[validation][:, fitted["columns"]], y[validation],
            scoring="average_precision", n_repeats=5, random_state=2026, n_jobs=-1,
        )
        pd.DataFrame({
            "feature": fitted["feature_names"],
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }).sort_values("importance_mean", ascending=False).to_csv(result_dir / "feature_importance.csv", index=False)

    balanced_summary = balanced.groupby("model").agg(
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        ap_mean=("average_precision", "mean"), ap_std=("average_precision", "std"),
        auc_mean=("roc_auc", "mean"), auc_std=("roc_auc", "std"),
        precision_mean=("precision", "mean"), recall_mean=("recall", "mean"),
    ).reset_index()
    balanced_summary.to_csv(result_dir / "balanced_metrics_summary.csv", index=False)
    _plot_metrics(actual, balanced, result_dir)

    summary = {
        "research_question": "Will a directed user pair active in the past 4 hours reconnect within the next 24 hours?",
        "dataset": dataset_summary(events, users, raw),
        "examples": int(len(metadata)), "positives": int(y.sum()),
        "split": split_summary,
        "split_counts": {
            key: {"examples": int(mask.sum()), "positives": int(y[mask].sum()), "positive_rate": float(y[mask].mean())}
            for key, mask in (("train", train), ("validation", validation), ("test", test))
        },
        "model_selection": "hyperparameters and threshold use validation only; test is touched once",
        "actual_test_metrics": actual.to_dict(orient="records"),
        "balanced_test_metrics": balanced_summary.to_dict(orient="records"),
        "tuning": tuning,
        "gnn": {
            "model": "directed graph autoencoder with GCN encoder and asymmetric bilinear decoder",
            "validation_snapshot_edges": validation_gae.positive_edges,
            "test_snapshot_edges": test_gae.positive_edges,
            "device": test_gae.device,
            "note": "Static boundary-snapshot baseline; it is not presented as a dynamic TGN.",
        },
        "best_model_by_actual_test_ap": best_name,
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(actual[["model", "average_precision", "roc_auc", "f1", "precision", "recall"]].to_string(index=False))
    return result_dir / "summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    run(rebuild_features=args.rebuild_features, repeats=args.repeats)


if __name__ == "__main__":
    main()

