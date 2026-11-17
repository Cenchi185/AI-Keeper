param(
    [Alias('PdfPath')]
    [string]$Pdf
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
function Assert-Exit([string]$Step) {
    if ($LASTEXITCODE -ne 0) { throw "$Step failed (exit code $LASTEXITCODE)" }
}
if ($args.Count -gt 0) {
    throw 'Unexpected arguments. Put the entire PDF path in quotes: .\setup.ps1 -Pdf "C:\folder\file name.pdf"'
}
if ($Pdf -and -not (Test-Path -LiteralPath $Pdf -PathType Leaf)) {
    throw "PDF not found: $Pdf"
}
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw 'Install Python 3.11.9 and Python Launcher (py) first.'
}
$version = & py -3.11 -c 'import platform; print(platform.python_version())'
Assert-Exit 'Python check'
if ($version.Trim() -ne '3.11.9') {
    Write-Warning "Detected Python $version. The assignment requires 3.11.9."
}
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & py -3.11 -m venv .venv
    Assert-Exit 'Virtual environment creation'
}
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$venvVersion = & $pythonPath -c 'import platform; print(platform.python_version())'
Assert-Exit 'Virtual environment check'
if ($venvVersion.Trim() -ne '3.11.9') {
    Write-Warning "Virtual environment uses Python $venvVersion. The assignment requires 3.11.9."
}
& $pythonPath -m pip install -r requirements.txt
Assert-Exit 'Package installation'
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw 'Install and start Ollama, then run this script again.'
}
& ollama pull qwen3:8b
Assert-Exit 'qwen3:8b download'
& ollama pull bge-m3
Assert-Exit 'bge-m3 download'
if ($Pdf) {
    & $pythonPath -m keeper.ingest --pdf $Pdf
    Assert-Exit 'PDF indexing'
} else {
    Write-Host 'Installed. Index your PDF separately using the README command.'
}
Write-Host 'When ready, run .\run.ps1.'
