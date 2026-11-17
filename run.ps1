$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw '가상환경이 없습니다. 먼저 .\setup.ps1 을 실행하세요.'
}
& $pythonPath -m keeper.server
if ($LASTEXITCODE -ne 0) { throw "서버 실행 실패 (종료 코드 $LASTEXITCODE)" }
