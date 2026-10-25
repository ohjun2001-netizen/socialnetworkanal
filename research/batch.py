"""Bounded parallel, resumable full research suite with persistent status."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import time

from .data import ROOT
from .features import prepare


def main():
    folder = ROOT / "results" / "research_v2"
    folder.mkdir(parents=True, exist_ok=True)
    status_path = folder / "batch_status.json"
    started = datetime.now(timezone.utc).isoformat()
    state = {"started": started,"pid":os.getpid(),"jobs":{},"status":"preparing_data"}
    def save():
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state,indent=2),encoding="utf-8")
        temporary.replace(status_path)
    save()
    # Prepare each shared cache once before parallel folds. On a fresh clone,
    # parallel downloads must not race on the same .part or feature-cache file.
    for dataset in ("mathoverflow", "askubuntu"):
        prepare(dataset)
    state["status"] = "running"
    save()
    def job(dataset,fold):
        out = folder / dataset
        out.mkdir(exist_ok=True)
        with (out / f"fold_{fold}_run.log").open("w",encoding="utf-8") as log:
            env = os.environ.copy();env["OMP_NUM_THREADS"]="4";env["MKL_NUM_THREADS"]="4"
            run = subprocess.run([sys.executable,"-u","-m","research.run","--dataset",dataset,"--fold",str(fold)],
                                 cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
        return dataset,fold,run.returncode
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(job,d,f):(d,f) for d in ("mathoverflow","askubuntu") for f in range(5)}
        for future in as_completed(futures):
            dataset,fold,code = future.result()
            state["jobs"][f"{dataset}/{fold}"] = {"exit_code":code,"finished":datetime.now(timezone.utc).isoformat()}
            save();print(f"completed {dataset} fold={fold} exit={code}",flush=True)
    if any(x["exit_code"] for x in state["jobs"].values()):
        state["status"]="failed";save();raise SystemExit(1)
    state["status"]="reporting";save()
    for dataset in ("mathoverflow","askubuntu"):
        with (folder / dataset / "report_run.log").open("w",encoding="utf-8") as log:
            run=subprocess.run([sys.executable,"-u","-m","research.report","--dataset",dataset],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        state[f"report_{dataset}"]=run.returncode
        save()
    state["status"]="complete" if state["report_mathoverflow"]==state["report_askubuntu"]==0 else "report_failed"
    state["finished"]=datetime.now(timezone.utc).isoformat();save()


if __name__=="__main__":
    main()
