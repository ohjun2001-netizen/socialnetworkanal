$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
python -m pytest tests/test_remedy_models.py -q
if ($LASTEXITCODE -ne 0) { throw "Remedy unit tests failed" }
python -u -m research_extensions.remedy_run --dataset mathoverflow
if ($LASTEXITCODE -ne 0) { throw "MathOverflow remedy run failed" }
python -u -m research_extensions.remedy_run --dataset askubuntu
if ($LASTEXITCODE -ne 0) { throw "AskUbuntu remedy run failed" }
python -u -m research_extensions.remedy_report
if ($LASTEXITCODE -ne 0) { throw "Remedy verification/report failed" }
