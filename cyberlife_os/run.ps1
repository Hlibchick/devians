$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path .venv/Scripts/python.exe)) { throw "Run setup.ps1 first" }
$env:OLLAMA_MAX_LOADED_MODELS = "2"
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_CONTEXT_LENGTH = "4096"
Start-Process "http://127.0.0.1:8000"
& .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
exit $LASTEXITCODE
