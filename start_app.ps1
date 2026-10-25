$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
python -m streamlit run research_app.py

