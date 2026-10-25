$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
python -u -m research_extensions.batch
if ($LASTEXITCODE -ne 0) { throw 'Extension run failed; check results/lookback_density_v1/status.json' }
