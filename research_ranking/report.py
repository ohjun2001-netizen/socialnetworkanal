"""Aggregate CUDA ranking runs and compare against the frozen baseline."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.data import ROOT
from research.evaluation import metrics
from research.report import markdown_table
from research_extensions.uncertainty import compare
from .model import source_ranking_metrics, source_ranking_records
from .run import MODELS, OUT

SCORE_KEYS = dict(zip(MODELS, ("score_baseline", "score_ranker", "score_ensemble")))
THRESHOLD_KEYS = dict(zip(MODELS, ("threshold_baseline", "threshold_ranker", "threshold_ensemble")))


def ndcg_comparison(candidate, baseline, repeats=1000, block_days=7, seed=92):
    keys = ["fold", "cutoff", "src"]
    paired = candidate[keys + ["ndcg_at_5"]].merge(
        baseline[keys + ["ndcg_at_5"]], on=keys, suffixes=("_candidate", "_baseline"), validate="one_to_one")
    if len(paired) != len(candidate) or len(paired) != len(baseline):
        raise ValueError("unmatched source-ranking groups")
    paired["delta"] = paired.ndcg_at_5_candidate - paired.ndcg_at_5_baseline
    block, _ = pd.factorize(pd.MultiIndex.from_arrays([
        paired.fold, paired.cutoff // (86400 * block_days)]), sort=True)
    number = int(block.max() + 1); rng = np.random.default_rng(seed); values = []
    for _ in range(repeats):
        multiplicity = np.bincount(rng.integers(0, number, number), minlength=number)
        weight = multiplicity[block]
        if weight.sum():
            values.append(float(np.average(paired.delta, weights=weight)))
    return {"delta_ndcg_at_5": float(paired.delta.mean()), "ci_low": float(np.quantile(values, .025)),
            "ci_high": float(np.quantile(values, .975)), "eligible_groups": len(paired),
            "block_days": block_days, "blocks": number, "repeats": len(values)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    completed = sorted(OUT.glob("fold_*/seed_*/complete.json"))
    if len(completed) != 15:
        raise RuntimeError(f"expected 15 CUDA runs, found {len(completed)}")
    signatures = {json.loads(path.read_text())["signature"] for path in completed}
    if len(signatures) != 1:
        raise AssertionError("ranking run signature mismatch")
    for path in completed:
        record = json.loads((path.parent / "training_record.json").read_text())
        if not record["device"]["device"].startswith("cuda"):
            raise AssertionError("non-CUDA ranking run detected")
    rows, ranking_rows, seed_frames = [], [], {}
    for seed in (2026, 2027, 2028):
        model_parts = {name: [] for name in MODELS}
        for fold in range(5):
            path = OUT / f"fold_{fold}" / f"seed_{seed}" / "locked_test_scores.npz"
            before = hashlib.sha256(path.read_bytes()).hexdigest(); z = np.load(path, allow_pickle=False)
            base = pd.DataFrame({key: z[key] for key in ("cutoff", "src", "dst", "label")})
            base["fold"] = fold
            for name in MODELS:
                frame = base.copy(); frame["score"] = z[SCORE_KEYS[name]]
                threshold_value = float(z[THRESHOLD_KEYS[name]][0])
                pair_value = metrics(frame.label, frame.score, threshold_value, frame.cutoff)
                rank_value = source_ranking_metrics(frame.cutoff, frame.src, frame.label, frame.score)
                rows.append({"fold": fold, "seed": seed, "model": name, **pair_value})
                ranking_rows.append({"fold": fold, "seed": seed, "model": name, **rank_value})
                model_parts[name].append(frame)
            if hashlib.sha256(path.read_bytes()).hexdigest() != before:
                raise AssertionError("locked ranking score changed during reporting")
        for name, parts in model_parts.items():
            seed_frames[(seed, name)] = pd.concat(parts, ignore_index=True)
    pair_metrics = pd.DataFrame(rows); rank_metrics = pd.DataFrame(ranking_rows)
    pair_metrics.to_csv(OUT / "all_pair_metrics.csv", index=False)
    rank_metrics.to_csv(OUT / "all_source_ranking_metrics.csv", index=False)
    per_fold_pair = pair_metrics.groupby(["model", "fold"], observed=True).agg(
        ap=("ap", "mean"), auc=("auc", "mean"), f1=("f1", "mean"), precision=("precision", "mean"),
        recall=("recall", "mean"), mcc=("mcc", "mean"), brier=("brier", "mean"),
        prevalence=("prevalence", "mean"), n=("n", "max"), positive=("positive", "max")).reset_index()
    per_fold_rank = rank_metrics.groupby(["model", "fold"], observed=True).agg(
        recall_at_1=("recall_at_1", "mean"), recall_at_5=("recall_at_5", "mean"),
        ndcg_at_5=("ndcg_at_5", "mean"), map_at_5=("map_at_5", "mean"),
        eligible_groups=("eligible_groups", "max"), median_candidates=("median_candidates", "mean")).reset_index()
    per_fold = per_fold_pair.merge(per_fold_rank, on=["model", "fold"])
    summary = per_fold.groupby("model", observed=True).agg(
        ap_mean=("ap", "mean"), ap_fold_sd=("ap", "std"), auc_mean=("auc", "mean"), f1_mean=("f1", "mean"),
        precision_mean=("precision", "mean"), recall_mean=("recall", "mean"), mcc_mean=("mcc", "mean"),
        brier_mean=("brier", "mean"), prevalence=("prevalence", "mean"),
        recall_at_1=("recall_at_1", "mean"), recall_at_5=("recall_at_5", "mean"),
        ndcg_at_5=("ndcg_at_5", "mean"), map_at_5=("map_at_5", "mean"),
        eligible_groups=("eligible_groups", "sum"), median_candidates=("median_candidates", "mean"),
        test_n=("n", "sum"), test_positive=("positive", "sum")).reset_index()
    per_fold.to_csv(OUT / "fold_metrics.csv", index=False); summary.to_csv(OUT / "model_summary.csv", index=False)

    ensembles = {}
    for name in MODELS:
        frames = [seed_frames[(seed, name)] for seed in (2026, 2027, 2028)]
        average = frames[0].copy()
        for frame in frames[1:]:
            pd.testing.assert_frame_equal(average.drop(columns="score"), frame.drop(columns="score"))
            average["score"] += frame.score.to_numpy()
        average["score"] /= len(frames); ensembles[name] = average
    base = ensembles[MODELS[0]]
    ap_comparisons = []
    for name in MODELS[1:]:
        ap_comparisons.append({"model": name, "baseline": MODELS[0], **compare(ensembles[name], base)})
    ap_comparisons = pd.DataFrame(ap_comparisons); ap_comparisons.to_csv(OUT / "ap_comparisons.csv", index=False)
    ranking_records = {}
    for name, frame in ensembles.items():
        parts = []
        for fold, group in frame.groupby("fold"):
            value = source_ranking_records(group.cutoff, group.src, group.label, group.score)
            value["fold"] = int(fold); parts.append(value)
        ranking_records[name] = pd.concat(parts, ignore_index=True)
    ndcg = ndcg_comparison(ranking_records["CUDA RankNet"], ranking_records[MODELS[0]])
    (OUT / "ndcg_comparison.json").write_text(json.dumps(ndcg, indent=2), encoding="utf-8")
    primary_ap = ap_comparisons[ap_comparisons.model == "CUDA RankNet"].iloc[0]
    success = bool(ndcg["ci_low"] > 0 and primary_ap.delta_ap >= -0.005)
    draw(summary, ap_comparisons, ndcg)
    output = {"status": "complete", "exploratory_not_confirmatory": True,
              "generated_at": datetime.now(timezone.utc).isoformat(), "run_signature": next(iter(signatures)),
              "completed_gpu_runs": 15, "models": summary.to_dict("records"),
              "ap_comparisons": ap_comparisons.to_dict("records"), "ranknet_ndcg_comparison": ndcg,
              "exploratory_success_rule_met": success,
              "interpretation": "exploratory ranking improvement" if success else "exploratory ranking rule not met"}
    write_report(output, summary, ap_comparisons, ndcg)
    (OUT / "summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"success": success, "ranknet_ap_delta": float(primary_ap.delta_ap),
                      "ndcg_delta": ndcg["delta_ndcg_at_5"], "ndcg_ci": [ndcg["ci_low"], ndcg["ci_high"]]}), flush=True)


def draw(summary, comparisons, ndcg):
    order = MODELS; table = summary.set_index("model").loc[order]
    labels = ["Baseline", "CUDA\nRankNet", "50/50\nensemble"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for ax, key, title, color in ((axes[0], "ap_mean", "Overall pair AP", "#287C8E"),
                                  (axes[1], "ndcg_at_5", "Source-conditional NDCG@5", "#C26D4B")):
        bars = ax.bar(labels, table[key], color=color); ax.bar_label(bars, fmt="%.4f", padding=3)
        ax.set_title(title); ax.set_ylim(0, max(table[key]) * 1.20); ax.grid(axis="y", alpha=.2)
    fig.suptitle("Super User exploratory GPU ranking extension")
    fig.tight_layout(); fig.savefig(OUT / "ranking_comparison.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.errorbar([ndcg["delta_ndcg_at_5"]], [0],
                xerr=[[ndcg["delta_ndcg_at_5"] - ndcg["ci_low"]], [ndcg["ci_high"] - ndcg["delta_ndcg_at_5"]]],
                fmt="o", color="#287C8E", capsize=5)
    ax.axvline(0, color="black", lw=1); ax.set_yticks([0], ["CUDA RankNet"])
    ax.set_xlabel("NDCG@5 difference vs frozen baseline (95% 7-day block interval)")
    ax.grid(axis="x", alpha=.2); fig.tight_layout(); fig.savefig(OUT / "ndcg_difference.png", dpi=170); plt.close(fig)


def write_report(output, summary, comparisons, ndcg):
    columns = ["model", "ap_mean", "auc_mean", "f1_mean", "precision_mean", "recall_mean",
               "recall_at_1", "recall_at_5", "ndcg_at_5", "map_at_5", "eligible_groups"]
    lines = ["# CUDA 사용자별 상대 랭킹 실험", "",
             "이 실험은 Super User 확인 결과를 본 뒤 설계한 **후향적 탐색 실험**입니다. 손대지 않은 독립 확인으로 해석하면 안 됩니다.", "",
             f"탐색적 성공 규칙 충족: **{'예' if output['exploratory_success_rule_met'] else '아니오'}**", "",
             "## 결과", "", markdown_table(summary[columns]), "", "## 전체 후보 AP 차이", "",
             markdown_table(comparisons), "", "## 사용자별 NDCG@5 차이", "",
             f"CUDA RankNet - 기준모델 = {ndcg['delta_ndcg_at_5']:+.6f}, 95% 구간 [{ndcg['ci_low']:+.6f}, {ndcg['ci_high']:+.6f}]", "",
             "## 해석", "",
             "- Recall@K, NDCG@5, MAP@5는 다음 24시간에 실제 재연결이 하나 이상 있는 (시점, 사용자) 그룹에서 후보 상대 순위를 평가합니다.",
             "- 전체 AP는 활동하지 않는 사용자와 모든 음성 후보까지 포함한 원래 링크 예측 난도를 유지합니다.",
             "- 새 RankNet은 모든 학습 후보의 이진 손실과 같은 사용자 안의 양성-음성 상대 순위 손실을 함께 학습했습니다.",
             "- 학습 기록의 device가 모두 cuda:0인지 검증했으며 CPU로 자동 대체하지 않았습니다.",
             "- 고정 50/50 앙상블은 테스트로 가중치를 고르지 않았습니다.",
             "- 이 결과는 확인 실험이 아니라 후속 탐색이며 인과관계나 SOTA를 주장하지 않습니다."]
    (OUT / "report_ko.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
