# AI Keeper

크툴루 퀵스타트 PDF를 검색하며 한국어로 진행하는 1인용 AI GM 실험입니다. 플레이어는 시나리오의 비밀을 모르는 상태에서 시작합니다. 이야기는 `qwen3:8b`가 생성하고, 자료 검색·주사위·게임 상태는 Python 도구가 처리합니다.

이 프로젝트는 크툴루 전체 규칙의 완전한 구현이 아닙니다. PDF 검색에 근거한 진행을 시험하는 과제용 프로토타입이며, 모델이 규칙이나 비밀 공개 시점을 잘못 판단할 가능성은 실제 플레이로 확인해야 합니다.

## 준비

- Windows, Python **3.11.9**, Ollama 설치
- 사용자가 직접 받은 `CHA23131 Call of Cthulhu 7th Edition Quick-Start Rules.pdf`
- 모델 다운로드와 Python 패키지 설치 시 인터넷 연결
- 게임용 PC 기준: i5-14400F / RTX 5060 Ti 16GB / RAM 32GB

원본 PDF는 이 배포물에 포함하지 않습니다. 출처: [Chaosium 공식 퀵스타트](https://www.chaosium.com/cthulhu-quickstart/). 생성한 검색 DB에도 원문 조각이 저장되므로 원본과 함께 공개 배포하지 마세요.

## 설치와 자료 준비

PowerShell에서 이 폴더로 이동한 뒤 실행합니다. Ollama가 실행 중이어야 합니다.

```powershell
.\setup.ps1 -PdfPath 'C:\자료\CHA23131 Call of Cthulhu 7th Edition Quick-Start Rules.pdf'
```

스크립트는 `.venv` 생성, 의존성 설치, `qwen3:8b`와 `bge-m3` 다운로드, PDF 인덱싱을 순서대로 진행합니다. `-PdfPath`를 생략하면 인덱싱은 수행하지 않습니다. 인덱싱은 플레이 전에 한 번 실행하며, PDF를 바꾸면 다시 실행합니다.

수동 실행 방법:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen3:8b
ollama pull bge-m3
.\.venv\Scripts\python.exe -m keeper.ingest --pdf 'C:\자료\CHA23131 Call of Cthulhu 7th Edition Quick-Start Rules.pdf'
```

`bge-m3`는 영문 PDF와 한국어 질문을 검색 벡터로 변환하는 모델이며, GM 답변 모델은 과제 조건인 `qwen3:8b`입니다.

## 실행

```powershell
.\run.ps1
```

브라우저에서 [http://127.0.0.1:8000](http://127.0.0.1:8000)을 엽니다. 이름을 입력하고 새 게임을 시작합니다. 세션 ID를 보관하면 같은 서버의 저장된 게임을 다시 불러올 수 있습니다. 서버를 다른 PC로 옮길 때에는 저장 데이터도 함께 옮겨야 합니다.

사용 예: `주변을 살펴볼게`, `상대에게 상황을 물어볼게`, `현재 소지품을 알려줘`. 원본 PDF의 시나리오 부분이나 내부 저장 데이터를 플레이 전에 읽으면 비밀을 알게 될 수 있습니다.

## 원격 접속

먼저 게임용 PC에서 로컬 실행을 확인하세요. 기본 설정은 해당 PC에서만 접속할 수 있습니다. 외부에서 쓸 때에는 SSH 터널이나 신뢰할 수 있는 사설 네트워크를 사용하세요. **Ollama의 11434 포트를 인터넷에 직접 공개하지 마세요.**

SSH 서버가 별도로 설정된 경우, 앱을 기본 로컬 주소로 실행한 상태에서 접속 PC에서 터널을 만들 수 있습니다.

```powershell
ssh -N -L 8000:127.0.0.1:8000 사용자명@게임용PC주소
```

접속 PC의 브라우저에서 `http://127.0.0.1:8000`을 엽니다. Windows의 SSH 서버 설치와 인증 설정은 이 프로젝트가 자동 수행하지 않습니다.

사설망에서 직접 접속하려면 서버 PowerShell에서 다음처럼 실행합니다.

```powershell
$env:KEEPER_HOST = '0.0.0.0'
$env:KEEPER_TOKEN = 'replace-with-your-own-random-token-32chars'
.\run.ps1
```

클라이언트는 `http://게임용PC의사설IP:8000`으로 접속하고 화면의 토큰 입력란에 같은 값을 넣습니다. 로컬 주소가 아닌 곳에 서버를 바인딩할 때에는 **24자 이상의 토큰**이 필수입니다. 예시 토큰을 그대로 쓰지 말고 집 PC에서 무작위 값을 생성하세요: `.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"`. 일반 HTTP에서는 토큰이 암호화되지 않으므로 공용 인터넷에는 그대로 노출하지 말고, 암호화된 터널 또는 HTTPS를 사용하세요. 방화벽 규칙과 네트워크 연결은 사용 환경에 맞게 별도로 설정해야 합니다. 단일 사용자용 앱이므로 토큰을 공유하면 저장된 게임에도 접근할 수 있습니다.

## 집 PC에서 설치하고 테스트하기

집 PC에서 이 저장소를 복제하고 다음 순서로 실행합니다. Python 3.11.9와 Ollama를 먼저 설치하세요.

```powershell
git clone https://github.com/Cenchi185/AI-Keeper.git
cd AI-Keeper
.\setup.ps1 -Pdf 'C:\PDF가있는폴더\CHA23131 Call of Cthulhu 7th Edition Quick-Start Rules.pdf'
.\run.ps1
```

같은 PC의 브라우저에서 `http://127.0.0.1:8000`을 엽니다. PDF 경로는 집 PC의 실제 위치를 지정해야 합니다. PDF는 저장소에 포함되지 않으므로 각 PC에서 직접 준비합니다.

설치와 인덱싱을 나누어 진행할 수도 있습니다.

```powershell
.\setup.ps1
.\.venv\Scripts\python.exe -m keeper.ingest --pdf 'C:\PDF경로\파일.pdf'
.\run.ps1
```

GitHub에는 코드와 안내만 올리세요. PDF, 검색 DB, 저장 게임, 토큰 등은 올리지 않습니다. 프로젝트의 `.gitignore`와 `git status`를 확인한 뒤 커밋하세요. 설정은 아래의 환경 변수로 전달하며 `.env` 파일을 자동으로 읽는다고 가정하지 않습니다.

처음 업로드할 때는 GitHub에서 비어 있는 저장소를 생성하고 이 프로젝트 폴더에서 아래 명령을 직접 실행합니다. 원격 주소는 본인의 것으로 바꾸고, 기존 Git 저장소 안이라면 먼저 현재 저장소 범위를 확인하세요. 이 명령은 안내이며 자동 실행되지 않았습니다.

```powershell
git init
git add .
git status
git commit -m "Add AI Keeper prototype"
git branch -M main
git remote add origin https://github.com/Cenchi185/AI-Keeper.git
git push -u origin main
```

의존성 버전은 `requirements.txt`에 고정되어 있습니다. GitHub Actions는 Python 3.11에서 단위 테스트와 문법 검사를 실행합니다. 모델·원본 PDF가 필요 없는 검사이며, 실제 이야기 품질과 집 PC의 GPU 동작은 별도로 테스트해야 합니다.

## 설정

환경 변수는 서버를 실행하는 PowerShell에서 설정합니다.

| 변수 | 기본값 | 의미 |
|---|---|---|
| `KEEPER_HOST` | `127.0.0.1` | 앱 수신 주소 |
| `KEEPER_PORT` | `8000` | 앱 포트 |
| `KEEPER_TOKEN` | 없음 | API 접근 토큰, 비로컬 바인딩 시 필수 |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama 주소 |
| `KEEPER_NUM_CTX` | `8192` | 모델 입력 문맥 크기 |

16GB VRAM PC를 고려해 문맥을 8,192토큰으로 시작합니다. 실제 GPU 적재량과 응답 속도는 드라이버·동시 실행 프로그램·모델 버전에 따라 달라집니다. 메모리가 부족하면 문맥을 4,096으로 줄이고 다시 실행해보세요. 모델 상태는 `ollama ps`로 확인할 수 있습니다. 이 안내는 원격 PC에서 실제 배포하거나 성능을 측정했다는 뜻은 아닙니다.

## 과제 설명과 확인

이 버전은 실험용 프로토타입입니다. **별도의 LLM 검토 단계가 상태 변경과 공개할 장면을 심사**하며, 장소별 단서 공개 조건을 코드로 확정한 방식은 아닙니다. LLM이 비밀을 조기에 말하지 않도록 제한하더라도 의미상 스포일러를 완벽하게 차단한다고 보장할 수 없습니다. 상태와 검색 범위를 확인하면서 테스트해야 합니다. 기본 조사자는 **구현 테스트용 간소화 캐릭터 하나**이며 이름만 지정합니다. 공식 준비된 조사자 4명 중 선택하는 기능은 없습니다. 규칙의 일부만 구현하므로 공식 크툴루 규칙 전체를 재현하는 제품이 아닙니다.

수업 예제와 달리 LangChain의 `create_agent`나 Chroma를 직접 사용하지 않습니다. **Ollama API를 호출하는 Python 도구 반복 처리**로 Agent를 구성하고, **SQLite에 문서와 임베딩을 저장**해 코사인 유사도로 검색합니다. 규칙 검색은 Agent가 호출하는 도구이며, 시나리오 검색은 턴 시작 시 수행합니다. `bge-m3`가 임베딩을, `qwen3:8b`가 판단과 한국어 서술을 담당합니다.

한 턴에는 검색 질의 준비, 도구 사용 판단, 공개 내용 검토, 최종 서술 등 여러 단계가 있습니다. 도구 호출 횟수에 따라 **LLM 호출이 약 4회 이상** 발생할 수 있어 단순 챗봇보다 느릴 수 있습니다. 실제 PC에서 응답 시간과 GPU 메모리 사용량은 아직 측정하지 않았습니다.

구현 컨셉은 **“플레이어의 자유 행동을 해석하고 PDF 검색·주사위·상태 관리 도구를 사용하는 한국어 AI GM”**입니다. 수업 자료의 Tool과 Agent 개념을 사용하며, 이 프로젝트의 에이전트 반복 처리는 Ollama API와 Python으로 직접 구성합니다.

시연 전에 새 게임 시작, 자료가 필요한 질문, 주사위 판정, 상태 변경, 저장 후 재개를 확인하세요. 검색 결과를 그대로 플레이어에게 노출하지 않는지와 발견하지 않은 단서를 미리 말하지 않는지도 별도 확인이 필요합니다. 검색·도구 실행 기록은 과제 설명에 활용하되, 시나리오 원문과 비밀이 담긴 내부 데이터는 플레이 화면에 표시하지 마세요.

과제 안내의 **실습실 컴퓨터에서 구현** 조건에 맞추어 실습실에서 설치·실행·수정·최종 검증을 진행하고, 집 PC를 사용하는 범위가 수업 조건에 맞는지도 확인하세요. 집 PC에서 동작한 결과만으로 실습실 평가 환경에서의 동작을 보장할 수 없습니다.


## 검증 결과와 남은 확인

- Python 3.13에서 모델 응답을 대체한 자동 검사 18개가 통과했습니다. 판정, 저장·복원, 인증, 실패 시 저장 유지, 비공개 자료의 최종 서술 입력 분리, 검색 영역 분리, 색인 교체 실패 복구를 확인했습니다.
- 제공된 실제 PDF에서 96개 조각을 추출하고 임베딩을 대체한 SQLite 색인 생성·교체를 확인했습니다. 실제 검색 정확도는 측정하지 않았습니다.
- PowerShell 및 JavaScript 문법과 Python 3.11 구문 호환성을 확인했습니다. Python 3.11.9 실제 실행은 집 PC에서 확인해야 합니다. GitHub Actions는 Python 3.11에서 자동 검사를 수행하도록 구성했습니다.
- 현재 컴퓨터에 Ollama가 없어 실제 모델의 도구 호출, 검색 품질, 응답 속도, 시나리오 완주, 스포일러 방지는 검증하지 않았습니다.
- 화면의 준비 상태는 PDF 색인 존재 여부입니다. Ollama 실행 여부와 설치 모델은 게임 시작 시 확인됩니다.

자동 검사: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`
