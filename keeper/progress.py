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
    return any(word in action for word in ('들어가', '들어갈', '들어간', '입장', '이동', '나간', '나갈', '나가', '돌아간', '돌아갈'))
