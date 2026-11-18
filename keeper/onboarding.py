"""Player-public onboarding and out-of-character help. Never reads GM evidence."""
INTRO = '''1920년, 미국 보스턴. 당신은 낡은 주택에 얽힌 소문을 조사해 달라는 의뢰를 받고 의뢰인을 만나러 왔습니다.

당신은 사건을 조사하는 '조사자'입니다. 지금은 의뢰인의 사무실에서 설명을 듣는 단계이며, 아직 현장을 방문하거나 숨겨진 사실을 알아낸 상태는 아닙니다. 목표는 의뢰 내용을 확인하고, 자료와 현장을 조사해 그 집에서 무슨 일이 있었는지 알아내는 것입니다.

처음부터 규칙을 알 필요는 없습니다. 하고 싶은 행동을 평범한 문장으로 말하세요. 결과가 불확실하면 Keeper가 필요한 기능을 알려주고 주사위를 굴립니다. 쉽고 당연한 행동은 판정 없이 진행합니다.

먼저 의뢰인에게 무엇을 조사해 달라는 것인지 물어보거나, 받은 자료를 살펴보거나, 현장으로 갈 준비를 할 수 있습니다. 아래 행동 예시는 선택지의 전부가 아닙니다.'''

SUGGESTIONS = ['의뢰인에게 조사할 집과 의뢰 내용에 대해 자세히 묻는다.',
               '현재 주변에서 직접 볼 수 있는 것들을 설명해 달라고 한다.',
               '조사 전에 확인할 자료가 있는지 의뢰인에게 묻는다.']


def upgrade_state(state):
    # Add public display metadata to existing saves without resetting game values.
    state.setdefault('occupation', '민간 조사자')
    state.setdefault('background', '기이한 사건의 사실관계를 확인하는 조사자. 직업과 수치는 테스트용 설정입니다.')
    state.setdefault('character_source', '간소화 테스트 조사자 · 공식 미리 생성된 조사자 아님')
    state.setdefault('max_sanity', 99)
    state.setdefault('goal', '주택에 관한 의뢰를 확인하고 자료와 현장을 조사한다.')
    state.setdefault('participants', [{'id': 'client', 'name': '의뢰인', 'role': '주택 조사를 의뢰한 인물',
                                      'available': True}] if state.get('location') == '의뢰인의 사무실' else [])
    return state


def suggestions(state):
    choices = state.get('last_action', {}).get('options', [])
    if isinstance(choices, list) and choices:
        return choices[:3]
    if state.get('location') == '의뢰인의 사무실':
        return SUGGESTIONS
    return ['현재 위치에서 눈에 보이는 것과 갈 수 있는 곳을 살펴본다.',
            '이미 발견한 단서를 바탕으로 다음 조사 방법을 생각한다.',
            '주변에 대화할 수 있는 사람이 있는지 확인한다.']


def help_reply(state, question):
    if any(word in question for word in ('이성', '정신', 'SAN')):
        return f"이성(SAN)은 충격을 견디는 정도를 나타내는 수치입니다. 현재 {state['sanity']}입니다. 정신력(POW) 능력치와는 다른 값입니다. 이 버전은 관련 규칙을 간소화했습니다. 도움말 질문은 이성을 줄이거나 게임을 진행하지 않습니다."
    if any(word in question for word in ('체력', 'HP')):
        return f"체력(HP)은 다치거나 위험에 처했을 때 감소하는 수치입니다. 현재 {state['hp']} / {state.get('max_hp', 12)}입니다. 오른쪽 또는 화면 상단의 내 조사자 카드에서 항상 확인할 수 있습니다."
    if any(word in question for word in ('누구', '캐릭터', '조사자', '직업')):
        return f"당신은 {state['name']}, 직업은 {state.get('occupation', '민간 조사자')}입니다. {state.get('background', '')} 현재는 간소화 테스트 캐릭터이며 공식 조사자 시트를 가져온 버전은 아닙니다."
    if any(word in question for word in ('주사위', '판정', '기능', '규칙')):
        return '위험하거나 결과가 불확실한 행동은 1~100 주사위로 기능 판정을 합니다. 보통 목표 수치 이하이면 성공합니다. 어려운 판정은 목표가 낮아집니다. 기능 수치는 캐릭터 시트에서 확인하세요. 숫자와 성공 여부는 Python 도구가 계산합니다. 모든 행동에 판정이 필요한 것은 아닙니다.'
    return (f"현재 위치: {state['location']}\n목표: {state.get('goal', '사건을 조사한다.')}\n\n"
            '살펴보기, 질문하기, 이미 아는 장소로 이동하기 같은 행동을 자유롭게 입력할 수 있습니다. 알려지지 않은 비밀까지 미리 알아야 할 필요는 없습니다. 장면이 불분명하면 현재 눈에 보이는 상황을 설명해 달라고 요청하세요.\n\n'
            '아래 행동 예시는 힌트이며 실제로 가능하다는 보장은 아닙니다. 위험하거나 상황에 맞지 않으면 Keeper가 설명합니다. 이 도움말은 게임 시간·위치·주사위·상태에 영향을 주지 않습니다.')
