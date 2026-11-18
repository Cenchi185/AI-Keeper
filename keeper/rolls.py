"""Explicit consent for every real random draw, with replay-safe cached results."""
import copy
import uuid
from .game import roll_check, roll_dice


def request_roll(session, action, target_id, kind, args, approved, *, required=False):
    state = session['state']
    req = {'id': str(uuid.uuid4()), 'kind': kind, 'args': args}
    if kind == 'skill':
        skill, difficulty = args['skill'], args['difficulty']
        if skill not in state['skills'] or difficulty not in ('regular', 'hard', 'extreme'):
            raise ValueError('지원하지 않는 기능 또는 난이도입니다.')
        req.update(skill=skill, difficulty=difficulty, base=state['skills'][skill],
                   target=state['skills'][skill] // {'regular':1,'hard':2,'extreme':5}[difficulty])
    elif kind == 'dice':
        count, sides = args['count'], args['sides']
        if type(count) is not int or type(sides) is not int or not 1 <= count <= 10 or not 2 <= sides <= 100:
            raise ValueError('지원하지 않는 주사위입니다.')
        req.update(count=count, sides=sides)
    else:
        raise ValueError('알 수 없는 판정 종류입니다.')
    if required:
        req['required'] = True
    session['pending_roll'] = {'request': req, 'action': action, 'target_id': target_id,
                               'approved': copy.deepcopy(approved)}
    return {'session_id': session['session_id'], 'status': 'awaiting_roll', 'roll_request': req,
            'reply': '', 'messages': [], 'state': state, 'rolls': public_rolls(approved), 'trace': []}


def public_rolls(approved):
    return [{'id': r['id'], 'kind': r['kind'], **r['result']} for r in approved]


def confirm_pending(session, request_id):
    pending = session.get('pending_roll')
    if not pending or pending['request']['id'] != request_id:
        raise ValueError('만료되었거나 이미 처리된 판정 요청입니다.')
    if 'result' not in pending:
        req = pending['request']
        result = (roll_check(session['state'], **req['args']) if req['kind'] == 'skill'
                  else roll_dice(**req['args']))
        if req['kind'] == 'skill':
            result['base'] = session['state']['skills'][req['args']['skill']]
        else:
            result.update(req['args'])
        pending['result'] = {'id': req['id'], 'kind': req['kind'], 'args': req['args'], 'result': result}
    return [*pending['approved'], pending['result']]
