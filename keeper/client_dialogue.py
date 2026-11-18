"""The landlord's opening knowledge, separated from GM-only scenario secrets.

Paraphrased from the public introduction, physical PDF page 19 of the supplied
50-page Quick-Start. The model selects facts; it cannot author new facts here.
"""
import json
import re
from .conversation import contacts
from .progress import hostile_action
from .onboarding import suggestions

FACTS = {
    'identity': '저는 이 주택을 최근 상속받은 소유주입니다. 임대하거나 매각하고 싶지만 좋지 않은 평판 때문에 어려움을 겪고 있습니다.',
    'commission': '보스턴 중심부의 코빗 하우스를 조사해 주세요. 이전 세입자인 마카리오 가족에게 비극적인 일이 있은 뒤로 집을 임대하지 못하고 있습니다. 그 집에서 무슨 일이 일어났는지 확인하고 나쁜 평판을 바로잡고 싶습니다.',
    'payment': '수고비는 하루 20달러입니다. 집의 열쇠와 주소를 드리고 20달러를 선금으로 지급하겠습니다.',
    'research': '먼저 과거 기록을 조사해 보는 것은 어떨까요? 보스턴 글로브 신문사, 중앙 도서관, 기록 보관소에서 자료를 찾아볼 수 있습니다. 바로 집을 방문할 수도 있습니다.',
    'limits': '마카리오 가족에게 비극이 있었다는 것은 알지만, 정확한 경위나 초자연적인 원인은 제가 확인한 사실이 아닙니다. 그 점을 알아봐 달라는 의뢰입니다.',
    'corbitt': '그 집은 코빗 하우스라고 불립니다. 코빗이라는 이름에 얽힌 자세한 사정은 기록을 조사해 확인해 주세요. 제가 그 사람의 현재 상태나 의도를 알고 있는 것은 아닙니다.',
    'unknown': '그 부분은 제가 확실히 아는 정보가 없습니다. 추측해서 말씀드리기보다는 기록이나 현장을 조사해 확인해 주셨으면 합니다.',
}


REPORT_REQUEST = re.compile(
    r'조사\s*(?:상황|진행|경과|결과|내용)[^.!?]{0,25}(?:이야기|전달|전하|전한|알리|알려|보고|설명|말하|말한)'
)
RECORD_OFFICES = ('보스턴 글로브 신문사', '중앙 도서관', '기록 보관소')

HANDOVER_ITEMS = ('코빗 하우스 열쇠', '코빗 하우스 주소 메모', '조사 선금 20달러')
HANDOVER_REQUEST = re.compile(
    r'의뢰(?:를)?\s*수락|(?:열쇠|주소|선금)[^.!?]{0,30}'
    r'(?:받(?:는다|을게|겠습니다|을 수 있)|(?:건네|전해|주)\s*(?:주세|세요|실 수|실래|실래요)|주세요|줘)'
)


def _handover(session):
    if session.get('client_handover_done'):
        return '열쇠와 주소 메모, 선금은 아까 드렸습니다.'
    inventory = session['state']['inventory']
    inventory.extend(item for item in HANDOVER_ITEMS if item not in inventory)
    session['client_handover_done'] = True
    return '여기 코빗 하우스 열쇠와 주소를 적은 메모입니다. 선금 20달러도 함께 드리죠.'


def applicable(state, target_id, action):
    if state.get('location') != '의뢰인의 사무실':
        return False
    if state.get('client_agitated') or hostile_action(action) or any(
        word in action for word in ('설득', '협박', '속인다', '거짓말')
    ):
        return False
    if not any(person['id'] == 'client' for person in contacts(state)):
        return False
    return target_id == 'client' or (
        target_id == 'gm' and bool(HANDOVER_REQUEST.search(action) or REPORT_REQUEST.search(action))
    )



def run(session, question, selector):
    handover = bool(HANDOVER_REQUEST.search(question))
    report = bool(REPORT_REQUEST.search(question))
    options = []
    if report:
        ids = []
        visited = session['state']['visited']
        remaining = [place for place in RECORD_OFFICES
                     if not any(place in old or (place == RECORD_OFFICES[0] and '보스턴 글로브' in old)
                                for old in visited)]
        text = '지금까지 조사한 경과는 들었습니다. 집에서 확인한 내용은 수첩에 남겨두세요.'
        if remaining:
            place = remaining[0]
            text += f' 집에서 조사가 막혔다면 이제 {place}에서 그 집의 과거 기록을 확인해 보시죠.'
            options = [f'이동: {place}']
        else:
            text += ' 이미 살펴본 기록과 현장 메모를 대조해 아직 확인하지 못한 점을 찾아보시죠.'
            options = ['지금까지 확인한 단서와 남은 의문을 정리한다.']
    elif handover:
        ids = ['payment']
        text = _handover(session)
    else:
        schema = {'type':'object', 'properties':{
            'fact_ids':{'type':'array','items':{'type':'string','enum':list(FACTS)},'maxItems':3},
            'correct_previous_claim':{'type':'boolean'}},
            'required':['fact_ids','correct_previous_claim'],'additionalProperties':False}
        context = {'question':question, 'allowed_facts':FACTS,
                   'last_exchange':session.get('history', [])[-4:],
                   'previous_fact_ids':session.get('client_fact_ids', [])}
        answer = selector([{'role':'system','content':'''Select ONLY fact IDs answering the user's latest question.
    You are the landlord in an opening TRPG briefing, NOT an omniscient GM.
    Earlier dialogue is context for pronouns, NOT a source of truth. Correct unsupported claims about
    deaths, dates, inhabitants' motives or supernatural causes. Never defend an earlier hallucination.
    If asked what an unsupported prior statement means, set correct_previous_claim true and select
    limits/corbitt/commission as appropriate. If asked for more information, select relevant not-yet-covered
    facts or limits. Do not repeat the invitation to agree to the job instead of answering the question.
    Output the requested JSON. Do not generate dialogue or any new names, dates or events.'''},
            {'role':'user','content':json.dumps(context,ensure_ascii=False)}], json_mode=schema)
        try:
            selected = json.loads(answer.get('content', '{}'))
        except (ValueError, TypeError):
            selected = {}
        if not isinstance(selected, dict):
            selected = {}
        ids = selected.get('fact_ids', []) if isinstance(selected, dict) else []
        if not isinstance(ids, list):
            ids = []
        ids = list(dict.fromkeys(v for v in ids if isinstance(v,str) and v in FACTS))[:3] or ['unknown']
        prefix = '앞서 제가 한 설명은 부정확했습니다. ' if selected.get('correct_previous_claim') is True else ''
        text = prefix + '\n\n'.join(FACTS[v] for v in ids)
    message = {'speaker_id':'client','speaker':'의뢰인','kind':'dialogue','text':text}
    reply = '[의뢰인] ' + text
    session['reply'] = reply
    session['reply_messages'] = [message]
    session['last_rolls'] = []
    session['client_fact_ids'] = list(dict.fromkeys(session.get('client_fact_ids', []) + ids))
    session['state']['scene'] = '의뢰인의 사무실에서 의뢰인과 대화 중입니다.'
    session['state']['scene_location'] = session['state']['location']
    if handover or report:
        session['state']['last_action'] = {'request': question, 'status': 'resolved', 'result': text, 'options': options}
    else:
        session['state'].pop('last_action', None)
    session['history'] = (session['history'] + [{'role':'user','content':question},
                           {'role':'assistant','content':reply}])[-12:]
    session['turn'] += 1
    return {'session_id':session['session_id'],'reply':reply,'messages':[message],
            'state':session['state'],'rolls':[],'trace':['의뢰인의 공개 지식 확인'],
            'contacts':contacts(session['state']),'suggestions':suggestions(session['state'])}
