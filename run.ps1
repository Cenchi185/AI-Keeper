$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Virtual environment missing. Run .\setup.ps1 first.'
}
& $pythonPath -m keeper.server
if ($LASTEXITCODE -ne 0) { throw "Server failed (exit code $LASTEXITCODE)" }
