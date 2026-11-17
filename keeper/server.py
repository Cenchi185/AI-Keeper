import os
import secrets
import threading
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import rag
from .agent import MODEL, run_turn
from .game import load, new_session, save

TOKEN = os.getenv('KEEPER_TOKEN', '')
HOST = os.getenv('KEEPER_HOST', '127.0.0.1')
if HOST not in ('127.0.0.1', 'localhost', '::1') and len(TOKEN) < 24:
    raise RuntimeError('원격 접속에는 24자 이상의 KEEPER_TOKEN이 필요합니다.')

app = FastAPI(title='AI Keeper', docs_url=None, redoc_url=None, openapi_url=None)
LOCK = threading.Lock()


def authorize(authorization: str = Header(default='')):
    if TOKEN and not secrets.compare_digest(authorization.encode('utf-8'), ('Bearer ' + TOKEN).encode('utf-8')):
        raise HTTPException(401, '접속 토큰을 확인하세요.')


class Start(BaseModel):
    name: str = Field(default='조사자', min_length=1, max_length=40)


class Turn(BaseModel):
    session_id: str = Field(max_length=36)
    message: str = Field(min_length=1, max_length=1500)


@app.get('/')
def index():
    return FileResponse(Path(__file__).resolve().parents[1] / 'static' / 'index.html')


@app.get('/api/status', dependencies=[Depends(authorize)])
def status():
    count = rag.document_count()
    return {'ready': count > 0, 'model': MODEL, 'documents': count}


def execute(session, action):
    if not LOCK.acquire(blocking=False):
        raise HTTPException(409, 'GM이 다른 행동을 처리 중입니다. 잠시 후 다시 시도하세요.')
    try:
        if rag.document_count() == 0:
            raise HTTPException(409, '먼저 PDF 색인을 생성하세요.')
        result = run_turn(session, action)
        save(session)  # Only complete turns commit, including state and history together.
        return result
    except httpx.HTTPError:
        raise HTTPException(503, 'Ollama 연결 또는 모델 응답에 실패했습니다. 실행 상태와 모델 설치를 확인하세요.')
    except (ValueError, KeyError, TypeError, RuntimeError):
        raise HTTPException(422, 'GM 처리 결과를 검증하지 못했습니다. 저장은 변경되지 않았습니다. 행동을 짧게 나누어 다시 시도하세요.')
    finally:
        LOCK.release()


@app.post('/api/start', dependencies=[Depends(authorize)])
def start(body: Start):
    return execute(new_session(body.name.strip() or '조사자'), 'GAME_START: 의뢰인의 사무실에서 사건의 의뢰를 듣는다. 비밀을 공개하지 말고 시작한다.')


@app.post('/api/turn', dependencies=[Depends(authorize)])
def turn(body: Turn):
    # Lock before loading prevents concurrent requests from overwriting a newer save.
    if not LOCK.acquire(blocking=False):
        raise HTTPException(409, 'GM이 처리 중입니다. 잠시 후 다시 시도하세요.')
    try:
        try:
            session = load(body.session_id)
        except KeyError:
            raise HTTPException(404, '저장된 게임이 없습니다.')
        if not body.message.strip():
            raise HTTPException(422, '행동을 입력하세요.')
        result = run_turn(session, body.message.strip())
        save(session)
        return result
    except httpx.HTTPError:
        raise HTTPException(503, 'Ollama 연결 또는 모델 응답 오류입니다. 저장된 상태는 유지됩니다.')
    except (ValueError, KeyError, TypeError, RuntimeError):
        raise HTTPException(422, 'GM 처리 결과를 검증하지 못했습니다. 행동을 짧게 나누어 다시 시도하세요.')
    finally:
        LOCK.release()


@app.get('/api/session/{session_id}', dependencies=[Depends(authorize)])
def session_get(session_id: str):
    try:
        session = load(session_id)
        return {key: session[key] for key in ('session_id', 'reply', 'state')}
    except ValueError:
        raise HTTPException(422, '올바르지 않은 저장 ID입니다.')
    except KeyError:
        raise HTTPException(404, '저장된 게임이 없습니다.')




if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=HOST, port=int(os.getenv('KEEPER_PORT', '8000')), access_log=False)
