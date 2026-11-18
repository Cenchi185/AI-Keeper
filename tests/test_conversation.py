import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from keeper import agent, game, server
from keeper.conversation import contacts, select_contact, update_participants, speaker_messages


class ContactTests(unittest.TestCase):
    def test_only_available_people_can_be_selected(self):
        state = game.initial_state('test')
        self.assertEqual([p['id'] for p in contacts(state)], ['gm', 'client'])
        state['participants'][0]['available'] = False
        with self.assertRaises(ValueError):
            select_contact(state, 'client')
        self.assertEqual(select_contact(state, 'gm')['name'], 'GM')

    def test_scene_change_removes_old_contacts(self):
        before = game.initial_state('test')
        after = copy.deepcopy(before)
        after['location'] = 'empty street'
        update_participants(before, after, {}, [])
        self.assertEqual(after['participants'], [])

    def test_new_people_require_exact_source_support(self):
        before = game.initial_state('test')
        after = copy.deepcopy(before)
        evidence = [{'text': 'Officer Example is standing beside the front desk.'}]
        good = {'canonical_name': 'Officer Example', 'name': 'invented translation',
                'present': True, 'source_quote': evidence[0]['text']}
        fake = {**good, 'canonical_name': 'Fabricated Person'}
        absent = {**good, 'present': False}
        update_participants(before, after, {'participants': [good, fake, absent]}, evidence)
        self.assertEqual(len(after['participants']), 1)
        self.assertEqual(after['participants'][0]['name'], 'Officer Example')

    def test_unselected_or_unknown_speakers_cannot_be_labeled_as_dialogue(self):
        state = game.initial_state('test')
        result = speaker_messages({'messages': [
            {'speaker_id': 'gm', 'text': 'narration'},
            {'speaker_id': 'client', 'text': 'speech'},
            {'speaker_id': 'invented', 'text': 'fake'}]}, state, 'gm')
        self.assertEqual([m['speaker_id'] for m in result], ['gm'])
        dialogue = speaker_messages([{'speaker_id': 'client', 'text': 'speech'}], state, 'client')
        self.assertEqual(dialogue[0]['speaker'], '의뢰인')
        self.assertEqual(dialogue[0]['kind'], 'dialogue')


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = patch.object(game, 'DATA', Path(self.temp.name))
        self.data.start()
        self.addCleanup(self.data.stop)
        self.token = patch.object(server, 'TOKEN', '')
        self.token.start()
        self.addCleanup(self.token.stop)
        self.client = TestClient(server.app)
        self.addCleanup(self.client.close)
        self.session = game.new_session('test')
        game.save(self.session)

    def test_stale_contact_is_rejected_before_model_call(self):
        with patch.object(server, 'run_turn') as run:
            response = self.client.post('/api/turn', json={
                'session_id': self.session['session_id'], 'message': 'hello', 'target_id': 'absent'})
        self.assertEqual(response.status_code, 409)
        run.assert_not_called()

    def test_npc_target_reaches_agent(self):
        def result(session, message, target_id):
            self.assertEqual(target_id, 'client')
            return {'reply': 'response', 'state': session['state']}
        with patch.object(server.client_dialogue, 'applicable', return_value=False), patch.object(server, 'run_turn', side_effect=result):
            response = self.client.post('/api/turn', json={
                'session_id': self.session['session_id'], 'message': 'hello', 'target_id': 'client'})
        self.assertEqual(response.status_code, 200)

    def test_gm_handover_action_updates_saved_inventory(self):
        session_id = self.session['session_id']
        with patch.object(server, 'run_turn') as generic_turn:
            response = self.client.post('/api/turn', json={
                'session_id': session_id, 'target_id': 'gm',
                'message': '의뢰를 수락하고 열쇠와 주소를 받는다.',
            })
        self.assertEqual(response.status_code, 200, response.text)
        generic_turn.assert_not_called()
        inventory = response.json()['state']['inventory']
        self.assertIn('코빗 하우스 열쇠', inventory)
        self.assertIn('코빗 하우스 주소 메모', inventory)
        self.assertIn('조사 선금 20달러', inventory)
        self.assertEqual(game.load(session_id)['state']['inventory'], inventory)

    def test_final_report_requires_visit_and_persists_ending(self):
        sid = self.session['session_id']
        report = {'session_id': sid, 'target_id': 'client',
                  'message': '조사를 마치고 집에는 아무 문제가 없다고 의뢰인에게 보고한다.'}
        response = self.client.post('/api/turn', json=report)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('ending', response.json()['state'])
        visited = game.load(sid)
        visited['state']['visited'].append('코빗 하우스 내부')
        game.save(visited)
        response = self.client.post('/api/turn', json=report)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['state']['ending'], 'unresolved_report')
        self.assertIn('엔딩', response.json()['reply'])
        self.assertEqual(response.json()['suggestions'], [])
        restored = self.client.get('/api/session/' + sid).json()
        self.assertEqual(restored['state']['ending'], 'unresolved_report')
        self.assertEqual(restored['reply'], response.json()['reply'])
        self.assertEqual(self.client.post('/api/turn', json=report).status_code, 409)

    def test_gm_question_reads_only_public_context_and_never_saves(self):
        self.session['history'] = [{'role': 'system', 'content': 'PRIVATE_HISTORY_SENTINEL'}]
        game.save(self.session)
        with patch.object(agent, 'chat', return_value={'content': '공개된 내용을 확인하세요.'}) as model, \
                patch.object(server, 'save') as save:
            response = self.client.post('/api/gm', json={
                'session_id': self.session['session_id'], 'message': '힌트 좀 줄래?', 'target_id': 'gm'})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('PRIVATE_HISTORY_SENTINEL', json.dumps(model.call_args.args))
        save.assert_not_called()
        self.assertEqual(game.load(self.session['session_id']), self.session)

    def test_load_retains_structured_speaker_messages(self):
        self.session['reply_messages'] = [{'speaker_id': 'client', 'speaker': '의뢰인',
                                           'kind': 'dialogue', 'text': '안녕하세요.'}]
        game.save(self.session)
        response = self.client.get('/api/session/' + self.session['session_id']).json()
        self.assertEqual(response['messages'], self.session['reply_messages'])
        self.assertIn('gm', [p['id'] for p in response['contacts']])

    def test_gm_request_with_npc_target_is_rejected(self):
        response = self.client.post('/api/gm', json={
            'session_id': self.session['session_id'], 'message': 'hint', 'target_id': 'client'})
        self.assertEqual(response.status_code, 422)


if __name__ == '__main__':
    unittest.main()
