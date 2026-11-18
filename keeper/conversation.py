"""Public speaker routing. Unknown or absent NPCs can never be addressed."""
import hashlib
import re


SPEECH_SCHEMA = {
    'type': 'object', 'properties': {'narration': {'type': 'string'}, 'dialogue': {'type': 'string'}},
    'required': ['narration', 'dialogue'], 'additionalProperties': False,
}


def parse_speech(value, state, target_id):
    """Assign speaker ids in code, never by interpreting names inside prose."""
    if isinstance(value, dict) and ('narration' in value or 'dialogue' in value):
        narration, dialogue = value.get('narration', ''), value.get('dialogue', '')
        if not isinstance(narration, str) or not isinstance(dialogue, str):
            raise ValueError('해설과 대사는 각각 문자열이어야 합니다.')
        messages = [{'speaker_id': 'gm', 'text': narration}]
        if dialogue.strip():
            if target_id == 'gm':
                raise ValueError('GM 응답에 NPC 대사가 섞였습니다.')
            select_contact(state, target_id)
            messages.append({'speaker_id': target_id, 'text': dialogue})
        value = {'messages': messages}
    return speaker_messages(value, state, target_id)


def speech_issues(messages, state, target_id):
    problems = []
    npc_names = [p['name'] for p in contacts(state) if p['id'] != 'gm']
    for message in messages:
        text = message['text']
        if re.search(r'physical observations|actual rolls only|selected NPC.s (?:actual words|direct answer)', text, re.I):
            problems.append('응답에 해설 또는 대사 작성 지시문이 포함됨')
        if message['speaker_id'] == 'gm':
            if re.search(r'["“「][^"”」\n]{8,}["”」]', text):
                problems.append('GM 해설에 직접 인용 대사가 있음')
            if re.search(r'(?:라고|다고|냐고|자고)\s*(?:말|설명|이야기|경고|전했|했)', text):
                problems.append('GM이 NPC 답변을 간접화법으로 대신 전달함')
            for name in npc_names:
                if re.search(re.escape(name) + r'(?:은|는|이|가).*(?:설명합|설명했|이야기했|이야기합|말했|말합|대답했|대답합)', text):
                    problems.append('GM 해설이 NPC의 답변 내용을 대신 서술함')
        else:
            if re.search(r'(?:도움을 받는다|자료를 조사한다|질문한다|이동한다|선택한다)[.!。]?$', text.strip()):
                problems.append('NPC 대사 자리에 행동 지시가 있음')
            if re.search(re.escape(message['speaker']) + r'(?:은|는|이|가).*(?:말|설명|바라|웃|고개|어깨)', text):
                problems.append('NPC 대사에 3인칭 무대 지시가 있음')
    if target_id != 'gm' and any(p['id'] == target_id for p in contacts(state)):
        if not any(m['speaker_id'] == target_id for m in messages):
            problems.append('선택한 NPC의 직접 답변이 없음')
    return list(dict.fromkeys(problems))


def contacts(state):
    return [{'id': 'gm', 'name': 'GM', 'role': '해설·힌트', 'available': True},
            *[dict(p) for p in state.get('participants', []) if p.get('available') is True]]


def select_contact(state, target_id):
    for person in contacts(state):
        if person['id'] == target_id:
            return person
    raise ValueError('지금 대화할 수 없는 인물입니다. 현재 장면의 대화 상대를 다시 선택하세요.')


def update_participants(before, after, review, evidence):
    moved = before.get('location') != after.get('location')
    # A scene change clears old contacts even if the model omits this field.
    if 'participants' not in review:
        after['participants'] = [] if moved else before.get('participants', [])
        return
    candidates = review['participants']
    if not isinstance(candidates, list):
        raise ValueError('대화 상대 정보는 목록이어야 합니다.')
    texts = '\n'.join(doc.get('text', '') for doc in evidence)
    known = {p['id']: p for p in before.get('participants', [])}
    verified = []
    for p in candidates[:8]:
        if not isinstance(p, dict) or p.get('present') is not True:
            continue
        previous = known.get(p.get('id'))
        if previous and not moved:
            verified.append(previous)
            continue
        canonical = p.get('canonical_name', '')
        quote = p.get('source_quote', '')
        name = p.get('name', '')
        if not all(isinstance(v, str) for v in (canonical, quote, name)):
            continue
        # Exact source citation, not a made-up name or a name from player instructions.
        if not (2 <= len(canonical) <= 80 and 2 <= len(name) <= 50
                and 10 <= len(quote) <= 500 and quote in texts and canonical.casefold() in quote.casefold()):
            continue
        identifier = 'npc-' + hashlib.sha256(canonical.casefold().encode()).hexdigest()[:12]
        # Display the source name, rather than trusting an invented translated identity.
        verified.append({'id': identifier, 'name': canonical, 'role': '현재 장면의 인물', 'available': True})
    after['participants'] = list({p['id']: p for p in verified}.values())


def speaker_messages(value, state, target_id='gm'):
    allowed = {p['id']: p['name'] for p in contacts(state)}
    result = []
    if isinstance(value, dict):
        value = value.get('messages', [])
    if not isinstance(value, list):
        raise ValueError('대화 응답 형식이 올바르지 않습니다.')
    for item in value[:8]:
        if not isinstance(item, dict):
            continue
        identifier = item.get('speaker_id', 'gm')
        text = item.get('text', '')
        if identifier not in allowed or (identifier != 'gm' and identifier != target_id):
            continue
        if isinstance(text, str) and text.strip():
            result.append({'speaker_id': identifier, 'speaker': allowed[identifier],
                           'kind': 'narration' if identifier == 'gm' else 'dialogue', 'text': text.strip()[:3000]})
    if not result:
        raise ValueError('공개 가능한 대화가 없습니다.')
    return result
