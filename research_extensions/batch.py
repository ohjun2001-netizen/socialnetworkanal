"""Resume the twelve pre-specified sensitivity arms with bounded CPU parallelism."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import os
import subprocess
import sys
from research.data import ROOT
from .features import prepare
from .run import fixed_configs


def main():
    output=ROOT/"results"/"lookback_density_v1";output.mkdir(parents=True,exist_ok=True)
    state={"status":"preparing","started":datetime.now(timezone.utc).isoformat(),"pid":os.getpid(),"jobs":{}}
    def save():
        path=output/"status.tmp";path.write_text(json.dumps(state,indent=2),encoding="utf-8");path.replace(output/"status.json")
    save()
    for d in ("mathoverflow","askubuntu"):
        prepare(d);fixed_configs(d)
    state["status"]="running";save()
    def job(dataset,lookback,stride):
        with (output/f"{dataset}_{lookback}_{stride}.log").open("w",encoding="utf-8") as log:
            env=os.environ.copy();env["OMP_NUM_THREADS"]="4";env["MKL_NUM_THREADS"]="4";env["NUMBA_NUM_THREADS"]="4"
            p=subprocess.run([sys.executable,"-u","-m","research_extensions.run","--dataset",dataset,"--lookback",str(lookback),"--stride",str(stride)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        return p.returncode
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs={pool.submit(job,d,l,s):(d,l,s) for d in ("mathoverflow","askubuntu") for l in (4,12,24) for s in (24,6)}
        for future in as_completed(jobs):
            d,l,s=jobs[future];code=future.result();state["jobs"][f"{d}/{l}/{s}"]={"exit_code":code,"finished":datetime.now(timezone.utc).isoformat()};save()
    if any(x["exit_code"] for x in state["jobs"].values()):
        state["status"]="failed";save();raise SystemExit(1)
    state["status"]="reporting";save()
    with (output/"report.log").open("w",encoding="utf-8") as log:
        p=subprocess.run([sys.executable,"-u","-m","research_extensions.report"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    state["status"]="complete" if p.returncode==0 else "report_failed"
    state["finished"]=datetime.now(timezone.utc).isoformat();save()


if __name__=="__main__":main()
