import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from keeper import agent, game, rolls, server


class ConsentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = patch.object(game, 'DATA', Path(self.temp.name))
        self.data.start(); self.addCleanup(self.data.stop)
        self.token = patch.object(server, 'TOKEN', '')
        self.token.start(); self.addCleanup(self.token.stop)
        self.client = TestClient(server.app); self.addCleanup(self.client.close)
        self.session = game.new_session('test')

    def pending(self):
        result = rolls.request_roll(self.session, '관찰한다', 'gm', 'skill',
                                    {'skill':'관찰','difficulty':'hard'}, [])
        game.save(self.session)
        return result['roll_request']['id']

    def test_agent_pauses_before_any_random_draw(self):
        before = copy.deepcopy(self.session)
        message = {'role':'assistant','tool_calls':[{'function':{'name':'check_skill',
                   'arguments':{'skill':'관찰','difficulty':'hard'}}}]}
        with patch.object(agent, '_evidence', return_value=[{'text':'evidence'}]), \
                patch.object(agent, 'chat', return_value=message), \
                patch.object(game.secrets, 'randbelow') as random:
            response = agent.run_turn(self.session, '관찰한다')
        random.assert_not_called()
        self.assertEqual(response['status'], 'awaiting_roll')
        self.assertEqual(response['roll_request']['target'], 30)
        self.assertEqual(response['roll_request']['base'], 60)
        self.assertEqual(self.session['state'], before['state'])
        self.assertEqual(self.session['turn'], before['turn'])

    def test_cancel_never_rolls_or_changes_state(self):
        id = self.pending()
        with patch.object(rolls, 'roll_check') as roll, patch.object(server, 'run_turn') as run:
            response = self.client.post('/api/roll', json={'session_id':self.session['session_id'],
                                       'request_id':id,'confirm':False})
        self.assertEqual(response.status_code, 200)
        roll.assert_not_called();run.assert_not_called()
        loaded = game.load(self.session['session_id'])
        self.assertNotIn('pending_roll', loaded)
        self.assertEqual(loaded['state'], self.session['state'])

    def test_model_failure_preserves_roll_and_retry_does_not_reroll(self):
        id = self.pending()
        with patch.object(game.secrets, 'randbelow', return_value=41) as random, \
                patch.object(server, 'run_turn', side_effect=ValueError('model format')):
            for _ in range(2):
                response = self.client.post('/api/roll', json={'session_id':self.session['session_id'],
                                           'request_id':id,'confirm':True})
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()['rolls'][0]['roll'], 42)
                self.assertFalse(response.json()['rolls'][0]['success'])
        self.assertEqual(random.call_count, 1)
        self.assertEqual(game.load(self.session['session_id'])['state'], self.session['state'])
        loaded = self.client.get('/api/session/'+self.session['session_id']).json()
        self.assertTrue(loaded['roll_request']['already_rolled'])
        self.assertEqual(loaded['rolls'][0]['roll'],42)

    def test_pending_request_blocks_unrelated_actions(self):
        self.pending()
        with patch.object(server, 'run_turn') as run:
            response = self.client.post('/api/turn', json={'session_id':self.session['session_id'],
                                       'message':'이동한다','target_id':'gm'})
        self.assertEqual(response.status_code, 409)
        run.assert_not_called()

    def test_raw_dice_also_require_confirmation(self):
        message = {'role':'assistant','tool_calls':[{'function':{'name':'roll_dice',
                   'arguments':{'count':2,'sides':6}}}]}
        with patch.object(agent, '_evidence', return_value=[{'text':'evidence'}]), \
                patch.object(agent, 'chat', return_value=message), \
                patch.object(game.secrets, 'randbelow') as random:
            result = agent.run_turn(self.session, '위험한 행동을 한다')
        random.assert_not_called()
        self.assertEqual(result['roll_request']['kind'],'dice')


if __name__ == '__main__':
    unittest.main()
