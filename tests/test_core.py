"""Offline behavior checks. No Ollama model or copyrighted PDF is required."""
import copy
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from keeper import agent, game, server


class DiceTests(unittest.TestCase):
    def test_percentile_boundaries_and_hard_difficulty(self):
        state = game.initial_state('테스트')
        cases = [(1, 'critical', True), (12, 'extreme', True),
                 (13, 'hard', True), (30, 'hard', True),
                 (31, 'regular', False), (60, 'regular', False),
                 (61, 'failure', False), (100, 'fumble', False)]
        for roll, level, success in cases:
            with self.subTest(roll=roll), patch.object(game.secrets, 'randbelow', return_value=roll-1):
                result = game.roll_check(state, '관찰', 'hard')
                self.assertEqual((result['target'], result['level'], result['success']),
                                 (30, level, success))

    def test_low_skill_fumbles_start_at_96(self):
        state = game.initial_state('테스트')
        for roll, expected in [(95, 'failure'), (96, 'fumble'), (99, 'fumble')]:
            with patch.object(game.secrets, 'randbelow', return_value=roll-1):
                self.assertEqual(game.roll_check(state, '회피')['level'], expected)

    def test_dice_validate_inputs_and_sum_real_rolls(self):
        for count, sides in [(True, 6), (1, False), (0, 6), (11, 6), (1, 1), (1, 101)]:
            with self.subTest(count=count, sides=sides), self.assertRaises(ValueError):
                game.roll_dice(count, sides)
        with patch.object(game.secrets, 'randbelow', side_effect=[0, 5]):
            self.assertEqual(game.roll_dice(2, 6), {'dice': [1, 6], 'total': 7})


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data_patch = patch.object(game, 'DATA', Path(self.temp.name))
        self.data_patch.start()
        self.addCleanup(self.data_patch.stop)

    def test_round_trip_and_same_id_update(self):
        session = game.new_session('한국어 조사자')
        game.save(session)
        self.assertEqual(game.load(session['session_id']), session)
        session['state']['clues'].append('발견한 기록')
        session['turn'] = 1
        game.save(session)
        self.assertEqual(game.load(session['session_id']), session)

    def test_invalid_and_missing_save_ids_are_distinct(self):
        with self.assertRaises(ValueError):
            game.load('../sessions.sqlite3')
        with self.assertRaises(KeyError):
            game.load(str(uuid.uuid4()))


class AgentTests(unittest.TestCase):
    def test_review_blocks_failed_discoveries_and_unproposed_movement(self):
        state = game.initial_state('테스트')
        review = {'location': '다른 장소', 'clues': ['새 단서'], 'items': ['열쇠'], 'scene': '불확실하다.'}
        proposals = [{'clue': '새 단서', 'item': '열쇠', 'reason': '조사'}]
        with patch.object(agent, 'chat', return_value={'content': json.dumps(review)}):
            result = agent._review(state, '조사한다', [], proposals, [{'success': False}], '')
        self.assertEqual(result['location'], state['location'])
        self.assertEqual(result['clues'], [])
        self.assertEqual(result['inventory'], state['inventory'])
        self.assertEqual(state, game.initial_state('테스트'))

    def test_review_damage_never_exceeds_proposal_or_turn_cap(self):
        state = game.initial_state('테스트')
        for proposed_hp, proposed_san, expected_hp, expected_san in [(2, 3, 10, 57), (999, 999, 0, 50), (-5, -5, 12, 60)]:
            with self.subTest(hp=proposed_hp), patch.object(agent, 'chat', return_value={
                    'content': json.dumps({'hp_loss': 999, 'sanity_loss': 999, 'scene': '위험한 상황이다.'})}):
                result = agent._review(state, '행동', [], [{'hp_loss': proposed_hp,
                                       'sanity_loss': proposed_san, 'reason': '위험'}], [], '')
                self.assertEqual((result['hp'], result['sanity']), (expected_hp, expected_san))

    def test_final_narrator_receives_only_reviewed_scene_not_raw_secrets(self):
        session = game.new_session('테스트')
        hidden = 'RAW_HIDDEN_EVIDENCE_SENTINEL'
        draft = 'UNREVIEWED_DRAFT_SENTINEL'
        calls = []

        def fake_chat(messages, tools=None, json_mode=False):
            calls.append(copy.deepcopy(messages))
            if tools:
                return {'role': 'assistant', 'content': draft}
            if json_mode and messages[0]['content'].startswith('Validate'):
                return {'content': json.dumps({'scene': '책상 위에 서류가 놓여 있다.',
                                               'clues': ['허가되지 않은 단서'], 'hp_loss': 12})}
            return {'content': '책상 위의 서류가 눈에 들어옵니다. 무엇을 하시겠습니까?'}

        with patch.object(agent, '_evidence', return_value=[{'text': hidden}]), \
                patch.object(agent, 'chat', side_effect=fake_chat):
            result = agent.run_turn(session, '책상을 살펴본다')
        self.assertIn(hidden, json.dumps(calls[0]))
        final_input = json.dumps(calls[-1], ensure_ascii=False)
        self.assertNotIn(hidden, final_input)
        self.assertNotIn(draft, final_input)
        self.assertIn('책상 위에 서류', final_input)
        self.assertEqual(result['state']['hp'], 12)
        self.assertEqual(result['state']['clues'], [])  # No proposal, no persistent grant.
        self.assertNotIn(hidden, json.dumps(result))
        self.assertEqual(session['turn'], 1)

    def test_empty_narration_does_not_commit_state_or_history(self):
        session = game.new_session('테스트')
        before = copy.deepcopy(session)
        responses = [{'content': 'draft'}, {'content': json.dumps({'scene': 'visible'})},
                     {'content': '<think>private reasoning</think>'}]
        with patch.object(agent, '_evidence', return_value=[{'text': 'evidence'}]), \
                patch.object(agent, 'chat', side_effect=responses), self.assertRaises(ValueError):
            agent.run_turn(session, '살펴본다')
        self.assertEqual(session, before)


class ApiTests(PersistenceTests):
    def setUp(self):
        super().setUp()
        self.token_patch = patch.object(server, 'TOKEN', 'test-token')
        self.token_patch.start()
        self.addCleanup(self.token_patch.stop)
        self.client = TestClient(server.app)
        self.addCleanup(self.client.close)
        self.headers = {'Authorization': 'Bearer test-token'}

    def test_all_game_endpoints_require_authorization(self):
        requests = [('GET', '/api/status', None), ('POST', '/api/start', {'name': 'A'}),
                    ('POST', '/api/turn', {'session_id': str(uuid.uuid4()), 'message': 'look'}),
                    ('GET', '/api/session/' + str(uuid.uuid4()), None)]
        for method, url, body in requests:
            with self.subTest(url=url):
                self.assertEqual(self.client.request(method, url, json=body).status_code, 401)
        with patch.object(server.rag, 'document_count', return_value=4):
            response = self.client.get('/api/status', headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['ready'])

    def test_failed_turn_retains_existing_saved_state_and_releases_lock(self):
        session = game.new_session('테스트')
        game.save(session)

        def fail_after_mutation(loaded, action, target_id='gm'):
            loaded['state']['hp'] = 0
            loaded['history'].append({'role': 'assistant', 'content': 'not committed'})
            raise ValueError('invalid model response')

        with patch.object(server, 'run_turn', side_effect=fail_after_mutation):
            response = self.client.post('/api/turn', headers=self.headers, json={
                'session_id': session['session_id'], 'message': '조사한다'})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(game.load(session['session_id']), session)
        self.assertFalse(server.LOCK.locked())

    def test_start_provides_intro_without_model_or_index(self):
        with patch.object(server.rag, 'document_count', return_value=0), \
                patch.object(server, 'run_turn', side_effect=ValueError('invalid response')), \
                patch.object(server, 'save') as save:
            response = self.client.post('/api/start', headers=self.headers, json={'name': '테스트'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('1920', response.json()['reply'])
        self.assertTrue(response.json()['suggestions'])
        save.assert_called_once()
        self.assertFalse(server.LOCK.locked())

    def test_help_never_changes_saved_game_or_calls_model(self):
        session = game.new_session('도움말 테스트')
        session['turn'] = 7
        game.save(session)
        with patch.object(server, 'run_turn') as run, patch.object(server, 'save') as save:
            response = self.client.post('/api/help', headers=self.headers, json={
                'session_id': session['session_id'], 'message': '이성 수치가 뭐야?'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['mode'], 'help')
        self.assertIn(str(session['state']['sanity']), response.json()['reply'])
        self.assertEqual(game.load(session['session_id']), session)
        run.assert_not_called()
        save.assert_not_called()

    def test_help_requires_authentication(self):
        response = self.client.post('/api/help', json={
            'session_id': str(uuid.uuid4()), 'message': '도움말'})
        self.assertEqual(response.status_code, 401)

    def test_legacy_state_gets_display_metadata_without_reset(self):
        session = game.new_session('기존 조사자')
        for key in ('occupation', 'background', 'character_source', 'goal', 'max_sanity'):
            session['state'].pop(key)
        session['state']['hp'] = 3
        session['state']['location'] = '이전에 방문한 장소'
        session['state']['clues'] = ['이미 발견한 단서']
        game.save(session)
        loaded = game.load(session['session_id'])['state']
        self.assertEqual(loaded['hp'], 3)
        self.assertEqual(loaded['location'], '이전에 방문한 장소')
        self.assertEqual(loaded['clues'], ['이미 발견한 단서'])
        self.assertIn('character_source', loaded)

    def test_public_save_response_excludes_hidden_history(self):
        session = game.new_session('테스트')
        session['history'] = [{'role': 'system', 'content': 'PRIVATE_HISTORY_SENTINEL'}]
        game.save(session)
        response = self.client.get('/api/session/' + session['session_id'], headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('PRIVATE_HISTORY_SENTINEL', response.text)
        self.assertNotIn('history', response.json())


if __name__ == '__main__':
    unittest.main()
