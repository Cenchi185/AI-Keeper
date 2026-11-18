"""Detect failed turn generation; do not confuse repeated prose with progress."""
import re
from difflib import SequenceMatcher


class TurnResolutionError(ValueError):
    """Safe player-facing message for an uncommitted failed generation."""


def normalize(text):
    return re.sub(r'[\W_]+', '', text.casefold())


def repeats_paragraphs(messages):
    paragraphs = []
    for message in messages:
        for p in re.split(r'\n\s*\n', message['text']):
            text = normalize(p)
            if len(text) >= 40:
                if any(SequenceMatcher(None, text, old).ratio() > .82 for old in paragraphs):
                    return True
                paragraphs.append(text)
    return False


def repeats_previous(messages, previous):
    text = normalize(' '.join(m['text'] for m in messages))
    old = normalize(previous)
    return bool(len(text) >= 40 and len(old) >= 40 and SequenceMatcher(None, text, old).ratio() > .78)


def movement_requested(action):
    action = re.sub(
        r'(?:내려가는|올라가는|들어가는|이동하는|가는)\s*(?:출입구|경로|통로|계단|문|길)',
        '', action,
    )
    action = re.sub(
        r'(?:들어가|나가|내려가|올라가|내려오|올라오|돌아가|가|오르|이동하|방문하|입장하)지\s*'
        r'(?:않[가-힣]*|말[가-힣]*)'
        r'|(?:안|못)\s+(?:들어가|나가|내려가|올라가|돌아가|가|오르|이동하|방문하|입장하)[가-힣]*',
        '', action,
    )
    return bool(re.search(
        r'오르(?:기|다|겠|자)|(?:올라|내려)\s*(?:[가갈간]|오|온)|내려오|들어[가갈간]|입장|이동|나[가갈간]|돌아[가갈간]|찾아[가갈간]|향(?:한다|할게)|'
        r'방문(?:한다|할게|해\s*볼게)|(?:으?로|에)\s*(?:가는|가겠|간다|갈게|가\s*볼게)',
        action,
    ))

def hostile_action(action):
    action = re.sub(r'(?:공격|폭행|제압|위협)하지\s*(?:않[가-힣]*|말[가-힣]*)', '', action)
    action = re.sub(r'(?:안|못)\s+(?:공격|폭행|제압|위협)[가-힣]*', '', action)
    return bool(re.search(r'공격|폭행|때린|때리|주먹으로 치|덮친|위협|깽판|난동|책상을 엎|기물을 부', action))
