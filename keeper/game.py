import json
import os
import secrets
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path
from .onboarding import upgrade_state

DATA = Path(os.getenv('KEEPER_DATA_DIR', str(Path(__file__).resolve().parents[1] / 'data')))



def initial_state(name):
    # A deliberately simplified original character, not an official pre-generated sheet.
    return upgrade_state({'name': name, 'hp': 12, 'max_hp': 12, 'sanity': 60,
            'location': '의뢰인의 사무실', 'inventory': ['수첩', '연필', '손전등'],
            'clues': [], 'visited': ['의뢰인의 사무실'],
            'skills': {'관찰': 60, '자료조사': 60, '듣기': 50, '설득': 50,
                       '은밀행동': 40, '회피': 40, '근접전': 40, '행운': 50}})


def roll_check(state, skill, difficulty='regular'):
    if skill not in state['skills']:
        raise ValueError('사용 가능한 기능: ' + ', '.join(state['skills']))
    if difficulty not in ('regular', 'hard', 'extreme'):
        raise ValueError('난이도는 regular/hard/extreme 중 하나입니다.')
    value = state['skills'][skill]
    target = value // {'regular': 1, 'hard': 2, 'extreme': 5}[difficulty]
    rolled = secrets.randbelow(100) + 1
    level = ('critical' if rolled == 1 else 'fumble' if rolled == 100 or
             (value < 50 and rolled >= 96) else 'extreme' if rolled <= value // 5
             else 'hard' if rolled <= value // 2 else 'regular' if rolled <= value else 'failure')
    success = level not in ('fumble', 'failure') and rolled <= target
    return {'skill': skill, 'roll': rolled, 'target': target, 'difficulty': difficulty,
            'level': level, 'success': success}


def roll_dice(count, sides):
    if type(count) is not int or type(sides) is not int or not 1 <= count <= 10 or not 2 <= sides <= 100:
        raise ValueError('주사위는 1~10개, 면수는 2~100만 허용합니다.')
    values = [secrets.randbelow(sides) + 1 for _ in range(count)]
    return {'dice': values, 'total': sum(values)}


def _db():
    DATA.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DATA / 'sessions.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
    return db


def save(session):
    with closing(_db()) as db:
        with db:
            db.execute('INSERT OR REPLACE INTO sessions VALUES (?, ?)',
                       (session['session_id'], json.dumps(session, ensure_ascii=False)))


def load(session_id):
    try:
        uuid.UUID(session_id)
    except (ValueError, TypeError, AttributeError):
        raise ValueError('올바르지 않은 저장 ID입니다.')
    with closing(_db()) as db:
        row = db.execute('SELECT payload FROM sessions WHERE id=?', (session_id,)).fetchone()
    if row is None:
        raise KeyError('저장된 게임을 찾지 못했습니다.')
    session = json.loads(row[0])
    upgrade_state(session['state'])
    return session


def new_session(name):
    return {'session_id': str(uuid.uuid4()), 'state': initial_state(name), 'history': [],
            'reply': '', 'turn': 0}
