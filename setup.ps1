param(
    [Alias('PdfPath')]
    [string]$Pdf
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

function Assert-Exit([string]$Step) {
    if ($LASTEXITCODE -ne 0) { throw "$Step 실패 (종료 코드 $LASTEXITCODE)" }
}

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw 'Python 3.11.9와 Python Launcher(py)를 먼저 설치하세요.'
}
$version = & py -3.11 -c 'import platform; print(platform.python_version())'
Assert-Exit 'Python 확인'
if ($version.Trim() -ne '3.11.9') {
    Write-Warning "현재 Python은 $version 입니다. 과제 지정 버전은 3.11.9입니다."
}
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & py -3.11 -m venv .venv
    Assert-Exit '가상환경 생성'
}
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$venvVersion = & $pythonPath -c 'import platform; print(platform.python_version())'
Assert-Exit '가상환경 확인'
if ($venvVersion.Trim() -ne '3.11.9') {
    Write-Warning "가상환경 Python은 $venvVersion 입니다. 과제 지정 버전은 3.11.9입니다."
}
& $pythonPath -m pip install -r requirements.txt
Assert-Exit '패키지 설치'
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw 'Ollama를 설치하고 실행한 뒤 이 스크립트를 다시 실행하세요.'
}
& ollama pull qwen3:8b
Assert-Exit 'qwen3:8b 다운로드'
& ollama pull bge-m3
Assert-Exit 'bge-m3 다운로드'
if ($Pdf) {
    if (-not (Test-Path -LiteralPath $Pdf -PathType Leaf)) {
        throw "PDF를 찾을 수 없습니다: $Pdf"
    }
    & $pythonPath -m keeper.ingest --pdf $Pdf
    Assert-Exit 'PDF 인덱싱'
} else {
    Write-Host '설치 완료. PDF 인덱싱은 README의 명령으로 별도 실행하세요.'
}
Write-Host '준비가 완료되면 .\run.ps1 로 실행하세요.'
