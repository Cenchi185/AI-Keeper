import os
import secrets
import threading
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import rag
from .agent import MODEL, run_turn, gm_question
from .conversation import contacts, select_contact
from .progress import TurnResolutionError
from .rolls import confirm_pending, public_rolls
from . import client_dialogue
from .agent import chat
from .game import load, new_session, save
from .onboarding import INTRO, help_reply, suggestions

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
    target_id: str = Field(default='gm', min_length=1, max_length=80)


class RollDecision(BaseModel):
    session_id: str = Field(max_length=36)
    request_id: str = Field(max_length=36)
    confirm: bool


def public_response(session, trace=None):
    pending = session.get('pending_roll', {})
    displayed_rolls = (public_rolls([*pending.get('approved', []), *([pending['result']] if pending.get('result') else [])])
                       if pending else session.get('last_rolls', []))
    return {**{key: session[key] for key in ('session_id', 'reply', 'state')},
            'trace': trace or [], 'suggestions': suggestions(session['state']),
            'messages': session.get('reply_messages', [{'speaker_id': 'gm', 'speaker': 'GM',
                         'kind': 'narration', 'text': session['reply']}]),
            'contacts': contacts(session['state']), 'rolls': displayed_rolls,
            'roll_request': ({**session['pending_roll']['request'], 'already_rolled': bool(session['pending_roll'].get('result'))} if session.get('pending_roll') else None),
            'status': 'awaiting_roll' if session.get('pending_roll') else 'complete'}


@app.get('/')
def index():
    return FileResponse(Path(__file__).resolve().parents[1] / 'static' / 'index.html')


@app.get('/api/status', dependencies=[Depends(authorize)])
def status():
    count = rag.document_count()
    return {'ready': count > 0, 'model': MODEL, 'documents': count, 'build': 'client-knowledge-5'}


def execute(session, action):
    if not LOCK.acquire(blocking=False):
        raise HTTPException(409, 'GM이 다른 행동을 처리 중입니다. 잠시 후 다시 시도하세요.')
    try:
        if rag.document_count() == 0:
            raise HTTPException(409, '먼저 PDF 색인을 생성하세요.')
        result = run_turn(session, action)
        save(session)  # Only complete turns commit, including state and history together.
        return result
    except TurnResolutionError as exc:
        raise HTTPException(422, str(exc))
    except httpx.HTTPError:
        raise HTTPException(503, 'Ollama 연결 또는 모델 응답에 실패했습니다. 실행 상태와 모델 설치를 확인하세요.')
    except (ValueError, KeyError, TypeError, RuntimeError):
        raise HTTPException(422, 'GM 처리 결과를 검증하지 못했습니다. 저장은 변경되지 않았습니다. 행동을 짧게 나누어 다시 시도하세요.')
    finally:
        LOCK.release()


@app.post('/api/start', dependencies=[Depends(authorize)])
def start(body: Start):
    session = new_session(body.name.strip() or '조사자')
    session['reply'] = INTRO
    session['history'] = [{'role': 'assistant', 'content': INTRO}]
    session['reply_messages'] = [{'speaker_id': 'gm', 'speaker': 'GM', 'kind': 'narration', 'text': INTRO}]
    save(session)
    return public_response(session)


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
        if session.get('pending_roll'):
            raise HTTPException(409, '먼저 대기 중인 주사위 판정을 승인하거나 취소해주세요.')
        try:
            select_contact(session['state'], body.target_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        if client_dialogue.applicable(session['state'], body.target_id, body.message):
            result = client_dialogue.run(session, body.message.strip(), chat)
        else:
            result = run_turn(session, body.message.strip(), body.target_id)
        save(session)
        result['suggestions'] = suggestions(session['state'])
        return result
    except TurnResolutionError as exc:
        raise HTTPException(422, str(exc))
    except httpx.HTTPError:
        raise HTTPException(503, 'Ollama 연결 또는 모델 응답 오류입니다. 저장된 상태는 유지됩니다.')
    except (ValueError, KeyError, TypeError, RuntimeError):
        raise HTTPException(422, 'GM 처리 결과를 검증하지 못했습니다. 행동을 짧게 나누어 다시 시도하세요.')
    finally:
        LOCK.release()


@app.post('/api/roll', dependencies=[Depends(authorize)])
def roll_endpoint(body: RollDecision):
    if not LOCK.acquire(blocking=False):
        raise HTTPException(409, 'GM이 처리 중입니다. 잠시 후 다시 시도하세요.')
    approved = []
    session = {}
    try:
        session = load(body.session_id)
        pending = session.get('pending_roll')
        if not pending or pending['request']['id'] != body.request_id:
            raise HTTPException(409, '이미 처리했거나 만료된 판정 요청입니다.')
        if not body.confirm:
            if pending.get('result'):
                raise HTTPException(409, '이미 굴린 결과는 취소할 수 없습니다. 같은 결과로 이어서 처리해주세요.')
            session.pop('pending_roll')
            session['last_rolls'] = public_rolls(pending.get('approved', []))
            save(session)
            result = public_response(session)
            result.update(messages=[], cancelled=True, reply='대기 중인 판정을 취소했습니다. 행동의 후속 처리는 실행되지 않았고 캐릭터 상태도 바뀌지 않았습니다. 이전에 굴린 주사위 기록은 유지됩니다.')
            return result
        approved = confirm_pending(session, body.request_id)
        # Persist the actual random result BEFORE any fallible model/network calls.
        save(session)
        result = run_turn(session, pending['action'], pending['target_id'], approved_rolls=approved)
        save(session)
        result['suggestions'] = suggestions(session['state'])
        return result
    except HTTPException:
        raise
    except KeyError:
        raise HTTPException(404, '저장된 게임이 없습니다.')
    except (ValueError, TypeError, RuntimeError, httpx.HTTPError):
        return JSONResponse(status_code=422, content={
            'detail': '판정 결과는 보관했지만 GM의 후속 설명을 생성하지 못했습니다. 같은 결과로 다시 이어갈 수 있습니다.',
            'rolls': public_rolls(approved), 'roll_request': ({**session['pending_roll']['request'], 'already_rolled': True} if session.get('pending_roll') else None)})
    finally:
        LOCK.release()


@app.get('/api/session/{session_id}', dependencies=[Depends(authorize)])
def session_get(session_id: str):
    try:
        session = load(session_id)
        return public_response(session)
    except ValueError:
        raise HTTPException(422, '올바르지 않은 저장 ID입니다.')
    except KeyError:
        raise HTTPException(404, '저장된 게임이 없습니다.')


@app.post('/api/help', dependencies=[Depends(authorize)])
def help_endpoint(body: Turn):
    try:
        session = load(body.session_id)
    except ValueError:
        raise HTTPException(422, '올바르지 않은 저장 ID입니다.')
    except KeyError:
        raise HTTPException(404, '저장된 게임이 없습니다.')
    if not body.message.strip():
        raise HTTPException(422, '질문을 입력하세요.')
    # Read-only: no GM context, model call, random roll, save or history mutation.
    return {'reply': help_reply(session['state'], body.message), 'state': session['state'],
            'suggestions': suggestions(session['state']), 'trace': [], 'mode': 'help'}


@app.post('/api/gm', dependencies=[Depends(authorize)])
def gm_endpoint(body: Turn):
    if body.target_id != 'gm':
        raise HTTPException(422, 'GM 설명은 GM을 선택하고 질문하세요.')
    if not body.message.strip():
        raise HTTPException(422, '질문을 입력하세요.')
    if not LOCK.acquire(blocking=False):
        raise HTTPException(409, 'GM이 처리 중입니다. 잠시 후 다시 시도하세요.')
    try:
        session = load(body.session_id)
        result = gm_question(session, body.message.strip())
        result['suggestions'] = suggestions(session['state'])
        return result
    except KeyError:
        raise HTTPException(404, '저장된 게임이 없습니다.')
    except httpx.HTTPError:
        raise HTTPException(503, 'GM 연결에 실패했습니다. Ollama를 확인하거나 기본 도움말을 사용하세요.')
    except (ValueError, TypeError, RuntimeError):
        raise HTTPException(422, 'GM 설명을 생성하지 못했습니다. 다시 질문하세요.')
    finally:
        LOCK.release()




if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=HOST, port=int(os.getenv('KEEPER_PORT', '8000')), access_log=False)
