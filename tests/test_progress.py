import copy
import json
import unittest
from unittest.mock import patch

from keeper import agent, game
from keeper.progress import repeats_paragraphs, repeats_previous, TurnResolutionError


class ProgressTests(unittest.TestCase):
    def test_duplicate_paragraphs_and_previous_answer_are_detected(self):
        paragraph = '문양을 바라보지만 그 의미는 아직 알 수 없습니다. 더 많은 정보가 있어야 한다는 생각이 듭니다. 같은 자리에서 문양을 살피고 있습니다.'
        self.assertTrue(repeats_paragraphs([{'text': paragraph + '\n\n' + paragraph}]))
        self.assertTrue(repeats_previous([{'text': paragraph}], '[GM] ' + paragraph))
        self.assertFalse(repeats_previous([{'text': '문을 열고 집 안으로 들어갔습니다.'}], paragraph))

    def test_missing_scene_does_not_reuse_previous_scene(self):
        state = game.initial_state('test')
        state['scene'] = 'STALE_SCENE_SENTINEL'
        with patch.object(agent, 'chat', return_value={'content': '{}'}):
            reviewed = agent._review(state, '문을 살핀다', [], [], [], '')
        self.assertNotEqual(reviewed.get('scene'), 'STALE_SCENE_SENTINEL')
        self.assertEqual(state['scene'], 'STALE_SCENE_SENTINEL')

    def test_ignored_movement_gets_one_tool_repair_then_commits_real_location(self):
        session = game.new_session('test')
        session['history'] = [{'role': 'assistant', 'content': 'UNSUPPORTED_PRIOR_STORY_SENTINEL'}]
        tool_calls = []

        def fake_chat(messages, tools=None, json_mode=False):
            if tools:
                tool_calls.append(copy.deepcopy(messages))
                if len(tool_calls) == 1:
                    return {'role': 'assistant', 'content': '이전 설명을 반복한다.'}
                if len(tool_calls) == 2:
                    return {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {
                        'name': 'propose_transition', 'arguments': {'location': '집 안', 'reason': '입구로 이동'}}}]}
                return {'role': 'assistant', 'content': '입구로 이동 요청을 처리한다.'}
            if messages[0]['content'].startswith('Validate'):
                return {'content': json.dumps({'location': '집 안', 'scene': '입구 안에 섰다.',
                    'action_result': '문을 열고 입구 안으로 이동했다.', 'action_status': 'resolved',
                    'next_options': ['주변을 살펴본다.']})}
            return {'content': json.dumps({'messages': [{'speaker_id': 'gm', 'text': '당신은 입구 안으로 들어왔습니다.'}]})}

        with patch.object(agent, '_evidence', return_value=[{'text': 'An accessible entrance.'}]), \
                patch.object(agent, 'chat', side_effect=fake_chat):
            response = agent.run_turn(session, '집으로 들어간다.')
        self.assertEqual(response['state']['location'], '집 안')
        self.assertEqual(len(tool_calls), 3)
        self.assertNotIn('UNSUPPORTED_PRIOR_STORY_SENTINEL', json.dumps(tool_calls))

    def test_unresolved_movement_never_commits_a_fake_turn(self):
        session = game.new_session('test')
        before = copy.deepcopy(session)
        with patch.object(agent, '_evidence', return_value=[{'text': 'evidence'}]), \
                patch.object(agent, 'chat', return_value={'role': 'assistant', 'content': 'unchanged'}), \
                patch.object(agent, '_review', return_value=copy.deepcopy(session['state'])):
            with self.assertRaises(TurnResolutionError):
                agent.run_turn(session, '집으로 들어간다.')
        self.assertEqual(session, before)


if __name__ == '__main__':
    unittest.main()
