$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
python -u -m research.batch
if ($LASTEXITCODE -ne 0) { throw 'Research run failed. See results/research_v2/batch_status.json.' }
python -u -m research.finalize
if ($LASTEXITCODE -ne 0) { throw 'Final verification failed.' }
