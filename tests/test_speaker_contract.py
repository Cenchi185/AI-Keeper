import copy
import json
import unittest
from unittest.mock import patch

from keeper import agent, game
from keeper.conversation import parse_speech, speech_issues
from keeper.progress import TurnResolutionError


class SpeakerContractTests(unittest.TestCase):
    def setUp(self):
        self.state = game.initial_state('test')

    def test_code_assigns_speakers_to_separate_fields(self):
        result = parse_speech({'narration': '의뢰인이 수첩을 내려놓습니다.',
                               'dialogue': '제가 아는 범위에서 말씀드리지요.'}, self.state, 'client')
        self.assertEqual([m['speaker_id'] for m in result], ['gm', 'client'])
        self.assertEqual(speech_issues(result, self.state, 'client'), [])

    def test_direct_and_indirect_npc_answers_in_gm_are_rejected(self):
        for text in ['의뢰인은 말했습니다. "그 사건에 대해서는 아는 것이 없습니다."',
                     '의뢰인은 그 사건에 대해 잘 모른다고 말했다.',
                     '의뢰인은 가족의 비극을 자세히 설명합니다.']:
            messages = [{'speaker_id': 'gm', 'speaker': 'GM', 'text': text},
                        {'speaker_id': 'client', 'speaker': '의뢰인', 'text': '잘 모르겠습니다.'}]
            with self.subTest(text=text):
                self.assertTrue(speech_issues(messages, self.state, 'client'))

    def test_npc_stage_direction_and_missing_reply_are_rejected(self):
        for text in ['자료 조사에 필요한 도움을 받는다', '의뢰인은 당신을 바라보며 말했다.']:
            messages = [{'speaker_id': 'client', 'speaker': '의뢰인', 'text': text}]
            self.assertTrue(speech_issues(messages, self.state, 'client'))
        self.assertTrue(speech_issues([{'speaker_id': 'gm', 'speaker': 'GM', 'text': '방이 조용합니다.'}], self.state, 'client'))

    def test_gm_cannot_receive_npc_dialogue(self):
        with self.assertRaises(ValueError):
            parse_speech({'narration': '', 'dialogue': '제가 말씀드리겠습니다.'}, self.state, 'gm')

    def test_mixed_response_is_repaired_before_commit(self):
        session = game.new_session('test')
        bad = {'messages': [{'speaker_id': 'gm', 'text': '의뢰인은 말했습니다. "모든 사실을 제가 알고 있습니다."'},
                            {'speaker_id': 'client', 'text': '자료 조사에 필요한 도움을 받는다'}]}
        good = {'narration': '의뢰인이 잠시 생각에 잠깁니다.', 'dialogue': '그 부분은 저도 잘 모르겠습니다.'}
        outputs = iter([bad, good])

        def model(messages, tools=None, json_mode=False):
            return {'content': 'draft'} if tools else {'content': json.dumps(next(outputs))}

        with patch.object(agent, '_evidence', return_value=[{'text': 'evidence'}]), \
                patch.object(agent, '_review', return_value=copy.deepcopy(self.state)), \
                patch.object(agent, 'chat', side_effect=model):
            result = agent.run_turn(session, '알고 계신가요?', 'client')
        self.assertEqual(result['messages'][0]['text'], good['dialogue'])
        self.assertEqual(result['messages'][0]['speaker_id'], 'client')
        self.assertEqual(session['turn'], 1)

    def test_persistent_mixing_does_not_modify_session(self):
        session = game.new_session('test')
        before = copy.deepcopy(session)
        bad = {'narration': '의뢰인은 비극적인 사건을 자세히 설명합니다.',
               'dialogue': '자료 조사에 필요한 도움을 받는다'}

        def model(messages, tools=None, json_mode=False):
            return {'content': 'draft'} if tools else {'content': json.dumps(bad)}

        with patch.object(agent, '_evidence', return_value=[{'text': 'evidence'}]), \
                patch.object(agent, '_review', return_value=copy.deepcopy(self.state)), \
                patch.object(agent, 'chat', side_effect=model):
            with self.assertRaises(TurnResolutionError):
                agent.run_turn(session, '설명해주세요.', 'client')
        self.assertEqual(session, before)


if __name__ == '__main__':
    unittest.main()
