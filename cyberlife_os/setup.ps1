$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
function CheckExit { if ($LASTEXITCODE -ne 0) { throw "Command failed: $LASTEXITCODE" } }
python -c "import sys; assert sys.version_info >= (3,11)"; CheckExit
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) { throw "Install Ollama from https://ollama.com first" }
if (-not (Test-Path .venv)) { python -m venv .venv; CheckExit }
& .\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio; CheckExit
& .\.venv\Scripts\python.exe -m pip install -r requirements-audio.txt; CheckExit
New-Item -ItemType Directory -Force models/piper | Out-Null
$voices = @("ru/ru_RU/irina/medium/ru_RU-irina-medium", "en/en_US/lessac/medium/en_US-lessac-medium")
foreach ($voice in $voices) {
  $name = ($voice -split '/')[-1]
  foreach ($suffix in @(".onnx", ".onnx.json")) {
    $dest = "models/piper/$name$suffix"
    if (-not (Test-Path $dest)) {
      Invoke-WebRequest -UseBasicParsing -Uri "https://huggingface.co/rhasspy/piper-voices/resolve/main/$voice$suffix" -OutFile "$dest.part"
      Move-Item "$dest.part" $dest
    }
  }
}
$env:OLLAMA_MAX_LOADED_MODELS = "2"
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_CONTEXT_LENGTH = "4096"
[Environment]::SetEnvironmentVariable("OLLAMA_MAX_LOADED_MODELS", "2", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_NUM_PARALLEL", "1", "User")
[Environment]::SetEnvironmentVariable("OLLAMA_CONTEXT_LENGTH", "4096", "User")
ollama pull nomic-embed-text; CheckExit
ollama pull qwen2.5:3b; CheckExit
ollama pull moondream; CheckExit
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
& .\.venv\Scripts\python.exe -m app.db.database; CheckExit
Write-Host "Setup complete. Configure .env, then run run.ps1. Restart Ollama with two-model settings (restart the Ollama tray application); see README."
