"""Run five CUDA folds sequentially, then create one exploratory report."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from research.data import ROOT

OUT = ROOT / "results" / "ranking_v1"


def dump(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True); status_path = OUT / "status.json"
    status = {"status": "cuda_training", "started": datetime.now(timezone.utc).isoformat(),
              "gpu_required": True, "jobs": {}, "report_runs": 0}
    dump(status_path, status)
    for fold in range(5):
        log_path = OUT / f"fold_{fold}_ranking.log"
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, "-u", "-m", "research_ranking.run", "--fold", str(fold)],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        status["jobs"][str(fold)] = {"exit_code": result.returncode, "finished": datetime.now(timezone.utc).isoformat()}
        dump(status_path, status); print(f"fold {fold} exit={result.returncode}", flush=True)
        if result.returncode:
            status["status"] = "failed"; dump(status_path, status); raise SystemExit(result.returncode)
    status["status"] = "reporting"; dump(status_path, status)
    report_log = OUT / "final_report.log"
    with report_log.open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, "-u", "-m", "research_ranking.report"], cwd=ROOT,
                                stdout=log, stderr=subprocess.STDOUT)
    status["report_runs"] = 1; status["report_exit_code"] = result.returncode
    status["status"] = "complete" if result.returncode == 0 else "report_failed"
    status["finished"] = datetime.now(timezone.utc).isoformat(); dump(status_path, status)
    if result.returncode:
        raise SystemExit(result.returncode)
    print("ranking_v1 CUDA experiment complete", flush=True)


if __name__ == "__main__":
    main()
