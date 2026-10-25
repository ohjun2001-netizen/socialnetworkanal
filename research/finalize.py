"""Verify the finished suite and write a plain-language handoff from real results."""
from datetime import datetime, timezone
import json
import hashlib
import importlib.metadata
import platform
import subprocess
import sys
import pandas as pd
import torch
from .data import ROOT
from .audit import audit
from .export_model import export


def main():
    tests=subprocess.run([sys.executable,"-m","pytest","-q"],cwd=ROOT,capture_output=True,text=True)
    if tests.returncode:
        raise RuntimeError(tests.stdout+tests.stderr)
    result_root = ROOT / "results" / "research_v2"
    summaries = {}
    for dataset in ("mathoverflow", "askubuntu"):
        path = result_root / dataset / "summary.json"
        if not path.exists():
            raise RuntimeError(f"Missing final report: {dataset}")
        summaries[dataset] = json.loads(path.read_text(encoding="utf-8"))
        if (summaries[dataset]["completed_runs"] != 15 or not summaries[dataset].get("final_report")
                or summaries[dataset].get("bootstrap_repeats",0) < 1000):
            raise RuntimeError(f"Incomplete final report: {dataset}")
    audits = {}
    for dataset in summaries:
        audits[dataset] = audit(dataset)
        export(dataset)
    lines = ["24시간 재연결 예측 프로젝트 — 최종 결과와 지표 설명", "",
             "1. 이번에 무엇을 끝냈나", "",
             "MathOverflow와 AskUbuntu 각각 5개 시간 구간 × 3개 시드, 총 30회 실험을 완료했습니다.",
             "각 실행에서 17개 모델/기준선을 같은 후보로 평가했습니다. 30회가 30배의 새 데이터라는 뜻은 아닙니다.",
             "원본 행동 종류 복구, 과거 정보 특징 생성, 시간순 학습/튜닝/보정/테스트, GPU 신경망, 특징 제거, 별도 데이터 재현, 시간 블록 신뢰구간, 저장 예측 재검산을 포함합니다.", "",
             "2. 예측 질문", "",
             "UTC 자정 직전 4시간에 관찰한 A→B 쌍이 이후 24시간에 다시 나타나면 정답 1, 없으면 0입니다. B→A는 별개입니다. 전체 24시간을 볼 수 없는 자료 끝부분은 제외했습니다.",
             "새로운 모든 사용자 쌍을 찾는 문제가 아니라 최근 관찰된 쌍의 재연결 문제입니다. 댓글·답변 상호작용이며 밈 전파나 인과관계가 아닙니다.", "",
             "3. 실제 결과", "",
             "아래 AP/AUC/F1/정밀도/재현율은 3개 시드 평균 후 5개 시간 구간을 평균한 값입니다. 아래의 통합 AP와는 집계 방식이 다릅니다."]
    for dataset, result in summaries.items():
        table = pd.DataFrame(result["models"]).set_index("model")
        pooled = pd.DataFrame(result["pooled_ensemble"]).set_index("model")
        lines += ["", f"[{dataset}]",
                  f"원본 {result['data']['raw_events']:,}이벤트, 자기 연결 제외 후 {result['data']['used_events']:,}이벤트, 사용 사용자 {result['data']['users']:,}명.",
                  f"중복 시드 제외 테스트 후보 {result['unique_test_candidates']:,}건, 실제 재연결 {result['unique_test_positives']:,}건, 양성률 {result['test_prevalence']:.2%}.",
                  "후보 1건은 사용자 1명이 아니라 특정 날짜의 방향성 사용자 쌍입니다.", "",
                  "모델 | AP | ROC-AUC | F1 | 정밀도 | 재현율 | MCC"]
        for name in ("History HGB","Full LightGBM","TGN recurrence","Rolling GraphSAGE","Hybrid ensemble","Always positive"):
            row=table.loc[name]
            lines.append(f"{name} | {row.ap_mean:.4f} | {row.auc_mean:.4f} | {row.f1_mean:.4f} | {row.precision_mean:.4f} | {row.recall_mean:.4f} | {row.mcc_mean:.4f}")
        full=table.loc["Full LightGBM"]; history=table.loc["History HGB"]
        lines += ["",f"주 모델 Full LightGBM의 구간 평균 AP 차이: {full.ap_mean-history.ap_mean:+.4f} (이력 HGB 대비).",
                  f"시드 평균 점수를 전체 기간에 모은 통합 AP: {pooled.loc['Full LightGBM','ap']:.4f}, 양성률 대비 AP lift {pooled.loc['Full LightGBM','ap_lift']:.2f}배.",
                  "이 lift는 정확도가 몇 배라는 뜻이 아니며 다른 데이터의 난도를 완전히 보정하지도 않습니다."]
        primary=[c for c in result["comparisons"] if c["model_a"]=="Full LightGBM" and c["model_b"]=="History HGB"]
        for c in primary:
            lines.append(f"{c['block_days']}일 블록 재표집: 통합 AP 차이 {c['delta_ap']:+.4f}, 95% 구간 [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}].")
        robust=all(c["ci_low"]>0 for c in primary)
        lines.append("세 블록 길이 모두 구간이 0보다 높아 이 설정에서 기준모델 대비 개선을 지지합니다." if robust else "일부 구간이 0을 포함하거나 아래로 내려갑니다. 모든 블록 설정에서 확실한 개선이라고 주장하지 않습니다.")
        for c in result["comparisons"]:
            if c["model_b"] in ("No types LightGBM","Full LightGBM"):
                lines.append(f"탐색적 비교 {c['model_a']} - {c['model_b']}: AP 차이 {c['delta_ap']:+.4f}, 95% 구간 [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}].")
        weights=[]; tgn_epochs=[]; sage_epochs=[]
        for f in range(5):
            for seed in (2026,2027,2028):
                d=json.loads((result_root/dataset/f"fold_{f}"/f"seed_{seed}"/"details.json").read_text())
                weights.append(d["Hybrid ensemble"]["neural_weight"])
                tgn_epochs.append(d["TGN recurrence"]["epoch"])
                sage_epochs.append(d["Rolling GraphSAGE"]["epoch"])
        lines.append(f"검증에서 선택한 앙상블 TGN 가중치: {weights}. 0이면 그 실행의 앙상블은 트리 점수만 사용한 것입니다.")
        lines.append(f"검증으로 선택한 TGN epoch: {tgn_epochs}; GraphSAGE epoch: {sage_epochs}. 최대 4 epoch 예산이며 충분한 수렴이나 대규모 하이퍼파라미터 탐색을 보장하지 않습니다.")
        # Concrete confusion counts from an explicitly named single run, never
        # manufacture counts by multiplying macro-average precision by 100.
        p=pd.read_csv(result_root/dataset/"fold_4"/"seed_2026"/"predictions.csv.gz")
        p=p[p.model=="Full LightGBM"]
        tp=int(((p.label==1)&p.prediction).sum());fp=int(((p.label==0)&p.prediction).sum())
        fn=int(((p.label==1)&~p.prediction).sum());tn=int(((p.label==0)&~p.prediction).sum())
        lines += ["", "지표를 실제 건수로 읽는 예시(마지막 구간, 시드 2026, Full LightGBM 한 번의 실행):",
                  f"재연결로 고른 {tp+fp:,}건 중 맞힌 {tp:,}건, 잘못 고른 {fp:,}건. 실제 재연결 {tp+fn:,}건 중 놓친 {fn:,}건. 음성을 맞힌 {tn:,}건.",
                  f"이 실행의 정밀도 {tp/max(tp+fp,1):.2%}, 재현율 {tp/max(tp+fn,1):.2%}. 위 표는 여러 실행 평균이므로 이 단일 예시와 수치가 다를 수 있습니다.",
                  f"Full LightGBM 평균 Brier {full.brier_mean:.4f}, 학습 양성률 상수 모델 {table.loc['Train prevalence','brier_mean']:.4f}. 낮을수록 확률 오차가 작습니다."]
    lines += ["", "4. 어떤 지표를 발표해야 하나", "",
              "AP: 실제 재연결 후보를 높은 순위로 올리는 능력입니다. AP 0.25는 정확도 25%가 아닙니다. 후보 양성률과 함께 제시합니다.",
              "ROC-AUC: 임의의 양성/음성 쌍에서 양성에 높은 점수를 주는 정도입니다. 0.5가 무작위 수준입니다.",
              "정밀도: 재연결이라고 고른 것 중 진짜 재연결 비율. 재현율: 실제 재연결 중 찾아낸 비율.",
              "F1: 정밀도와 재현율의 조화평균. 판정 임계값에 의존하며 임계값은 테스트 이전 보정 구간에서 정했습니다.",
              "균형 표본에서는 모두 양성이라고 해도 F1=0.667입니다. 현재 자연 양성률의 F1과 이전 균형 F1을 직접 비교하지 마세요.",
              "MCC: 양성·음성의 맞음/틀림을 모두 보는 지표. 1이 완벽하고 상수 예측은 여기서 0입니다.",
              "Brier: 출력 확률의 제곱 오차. 낮을수록 좋으며 연결 횟수 같은 원시 점수에는 적용하지 않습니다.",
              "95% 블록 구간: 같은 날짜 블록을 함께 재표집한 두 모델의 AP 차이 범위입니다. 0을 걸치면 개선이 확실하다고 말하기 어렵습니다. 단일 사람/쌍의 반복 의존성을 모두 없애는 것은 아닙니다.", "",
              "5. 논문 관점의 정당성과 한계", "",
              "시간 누수 방지, 정답 관찰 가능성, 강한 기준선, 특징 제거, 여러 시기/시드, 다른 데이터 재현, 불확실성, 예측 파일 검증을 갖췄습니다. 그러나 이 자체가 새로운 알고리즘이나 논문 게재를 보장하지 않습니다.",
              "MathOverflow는 기존 테스트를 이미 본 후향적 연구입니다. AskUbuntu도 해당 데이터로 다시 학습했으므로 제로샷 일반화는 아닙니다.",
              "TGN은 과제별 구현이며 공식 재현이 아닙니다. 128개 이벤트 단위 메모리 갱신, 최근 10이웃, 4 epoch 예산을 사용합니다. 트리와 신경망의 탐색 예산이 같지는 않습니다.",
              "평가 시각은 UTC 자정이며 직전 4시간 활동 쌍만 후보입니다. 전체 시간대·모든 사용자 쌍으로 일반화하지 않습니다. 최대 28일보다 긴 의존성과 서비스 변화가 남을 수 있습니다.",
              "이번 결과를 보고 같은 테스트에 계속 맞추면 검증의 신뢰도가 낮아집니다. 후속 개선은 새 기간/새 데이터의 미확인 평가를 먼저 확보하는 것이 좋습니다.", "",
              "6. 어디서 확인하나", "",
              "start_app.ps1: 새 결과 화면. 열람만으로 재학습/유료 API를 호출하지 않습니다.",
              "results/research_v2/<데이터>/report_ko.md: 표와 신뢰구간.",
              "같은 폴더의 ap_comparison.png / fold_stability.png / pr_and_calibration.png: 그래프.",
              "fold_N/seed_N/predictions.csv.gz: 모든 후보의 정답·점수·판정.",
              "audit.json: 저장 예측 재계산과 원본 정답 검사 결과.",
              "models/research_v2/<데이터>/full_lightgbm.joblib: 마지막 구간 시드 2026의 사전 지정 모델. 테스트로 재학습하지 않았고 저장 예측 재현을 검증했습니다.",
              "RESEARCH_PROTOCOL_KO.md / IMPLEMENTATION_AUDIT_KO.md: 세부 규칙·이전 구현 정정."]
    (ROOT/"최종_결과_쉽게_설명.txt").write_text("\n".join(lines),encoding="utf-8")
    validation=["# 최종 검증 — research v2", "", "두 데이터 각각 15회, 총 30회 실험을 완료하고 저장 예측을 재검산했습니다.", "",
                "자동 테스트:", "", "```text", tests.stdout.strip(), "```", ""]
    for dataset,r in audits.items():
        validation += [f"## {dataset}", "",f"- 모델 실행 {r['verified_model_runs']}개 지표 재계산 일치.",
                       f"- 원본 이벤트로 후보·정답 {r['label_spotchecks']}건 재검산.",
                       f"- 시드 중복 제외 테스트 후보 {r['test_candidates']:,}건.",
                       f"- 신경망 실행 장치 집계: {r['neural_device_counts']}.",
                       "- 분할 경계의 정답 관찰 완료, 테스트 구간 비중복, 동일 후보, 유한 점수, 상수 기준선 검사 통과.",
                       "- 내보낸 Full LightGBM 모델이 저장된 테스트 예측을 재현함을 확인.", ""]
    validation += ["이 검증은 구현 일관성 확인이며 과학적 새로움·서비스 적용 가능성·논문 채택의 보증이 아닙니다."]
    (ROOT/"VALIDATION_KO.md").write_text("\n".join(validation),encoding="utf-8")
    packages=("numpy","pandas","scikit-learn","lightgbm","optuna","torch","torch-geometric","scipy","matplotlib","streamlit","joblib","pytest")
    environment={"python":platform.python_version(),"system":platform.platform(),
                 "packages":{p:importlib.metadata.version(p) for p in packages},
                 "cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for folder in ("research","mathoverflow_lab","tests") for p in (ROOT/folder).glob("*.py")}
    (result_root/"finalization.json").write_text(json.dumps({"status":"complete","finished":datetime.now(timezone.utc).isoformat(),
            "audits":audits,"tests":tests.stdout.strip(),"environment":environment,"final_source_sha256":source_hashes},indent=2),encoding="utf-8")
    print("Final verification, model exports, and Korean handoff complete.",flush=True)


if __name__=="__main__":
    main()
