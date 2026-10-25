"""Bounded, resumable Super User training followed by exactly one final report."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import os
import subprocess
import sys

from research.data import ROOT


def main():
    output=ROOT/"results"/"confirmation_superuser_v1";output.mkdir(parents=True,exist_ok=True)
    status_path=output/"status.json"
    state={"status":"training_locked_scores","started":datetime.now(timezone.utc).isoformat(),"pid":os.getpid(),"jobs":{},"test_report_runs":0}
    def save():
        temporary=status_path.with_suffix(".tmp");temporary.write_text(json.dumps(state,indent=2),encoding="utf-8");temporary.replace(status_path)
    save()
    def job(fold):
        path=output/f"fold_{fold}_confirmation.log"
        with path.open("w",encoding="utf-8") as log:
            env=os.environ.copy();env["OMP_NUM_THREADS"]="4";env["MKL_NUM_THREADS"]="4"
            result=subprocess.run([sys.executable,"-u","-m","research_confirmation.run","--fold",str(fold)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
        return fold,result.returncode
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures={executor.submit(job,fold):fold for fold in range(5)}
        for future in as_completed(futures):
            fold,code=future.result();state["jobs"][str(fold)]={"exit_code":code,"finished":datetime.now(timezone.utc).isoformat()};save()
            print(f"fold {fold} locked exit={code}",flush=True)
    if any(job["exit_code"] for job in state["jobs"].values()):
        state["status"]="training_failed_before_report";save();raise SystemExit(1)
    if (output/"summary.json").exists():
        state["status"]="refused_existing_report";save();raise RuntimeError("summary already exists")
    state["status"]="first_final_report";state["test_report_runs"]=1;save()
    with (output/"first_final_report.log").open("w",encoding="utf-8") as log:
        result=subprocess.run([sys.executable,"-u","-m","research_confirmation.report"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    state["report_exit_code"]=result.returncode
    state["status"]="complete" if result.returncode==0 else "report_failed"
    state["finished"]=datetime.now(timezone.utc).isoformat();save()
    if result.returncode:raise SystemExit(result.returncode)
    print("Super User confirmation complete; first final report sealed.",flush=True)


if __name__=="__main__":main()
