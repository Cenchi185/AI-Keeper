"""The landlord's opening knowledge, separated from GM-only scenario secrets.

Paraphrased from the public introduction, physical PDF page 19 of the supplied
50-page Quick-Start. The model selects facts; it cannot author new facts here.
"""
import json
from .conversation import contacts
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


def applicable(state, target_id, action):
    return (target_id == 'client' and state.get('location') == '의뢰인의 사무실'
            and not any(word in action for word in ('설득', '위협', '협박', '속인다', '거짓말', '공격')))


def run(session, question, selector):
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
    session['state'].pop('last_action', None)
    session['history'] = (session['history'] + [{'role':'user','content':question},
                           {'role':'assistant','content':reply}])[-12:]
    session['turn'] += 1
    return {'session_id':session['session_id'],'reply':reply,'messages':[message],
            'state':session['state'],'rolls':[],'trace':['의뢰인의 공개 지식 확인'],
            'contacts':contacts(session['state']),'suggestions':suggestions(session['state'])}
