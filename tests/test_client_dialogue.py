import json
import unittest
from unittest.mock import Mock
from keeper import game, client_dialogue


class ClientKnowledgeTests(unittest.TestCase):
    def test_model_cannot_invent_dialogue_facts(self):
        session = game.new_session('test')
        model = Mock(return_value={'content':json.dumps({'fact_ids':['commission','SECRET_EVIL_PLAN'],
                             'correct_previous_claim':False,'dialogue':'INVENTED_STORY'})})
        result = client_dialogue.run(session, '의뢰 내용은 무엇인가요?', model)
        self.assertEqual(result['messages'][0]['text'],client_dialogue.FACTS['commission'])
        self.assertNotIn('INVENTED_STORY',result['reply'])
        self.assertEqual(result['rolls'],[])

    def test_followup_context_is_available_but_not_authoritative(self):
        session = game.new_session('test')
        session['history']=[{'role':'assistant','content':'그 사람은 떠나버립니다.'}]
        model = Mock(return_value={'content':json.dumps({'fact_ids':['limits'],'correct_previous_claim':True})})
        result = client_dialogue.run(session, '떠난다는 게 무슨 뜻이죠?', model)
        self.assertIn('부정확',result['reply'])
        self.assertIn('떠나버립니다',json.dumps(model.call_args.args,ensure_ascii=False))
        self.assertNotIn('조사에 동의',result['reply'])

    def test_npc_does_not_receive_hidden_scenario_or_saved_clues(self):
        session=game.new_session('test')
        session['state']['clues']=['HIDDEN_SCENARIO_SENTINEL']
        model=Mock(return_value={'content':'{"fact_ids":["unknown"],"correct_previous_claim":false}'})
        client_dialogue.run(session,'추가 정보가 있나요?',model)
        self.assertNotIn('HIDDEN_SCENARIO_SENTINEL',json.dumps(model.call_args.args))

    def test_profile_does_not_apply_to_other_npcs_or_risky_actions(self):
        state=game.initial_state('test')
        self.assertTrue(client_dialogue.applicable(state,'client','사건에 대해 묻는다'))
        self.assertFalse(client_dialogue.applicable(state,'client','의뢰인을 위협한다'))
        self.assertFalse(client_dialogue.applicable(state,'someone-else','사건에 대해 묻는다'))

    def test_request_for_key_and_address_delivers_the_package_once(self):
        session = game.new_session('test')
        model = Mock()
        result = client_dialogue.run(session, '열쇠와 주소를 받는다.', model)
        self.assertIn('함께 드리죠', result['reply'])
        self.assertIn('함께 드리죠', result['state']['last_action']['result'])
        model.assert_not_called()
        repeat = client_dialogue.run(session, '열쇠와 주소를 주세요.', model)
        self.assertIn('아까 드렸습니다', repeat['reply'])
        for item in client_dialogue.HANDOVER_ITEMS:
            self.assertEqual(repeat['state']['inventory'].count(item), 1)

    def test_payment_question_does_not_deliver_the_package(self):
        session = game.new_session('test')
        model = Mock(return_value={'content': '{"fact_ids":["payment"],"correct_previous_claim":false}'})
        result = client_dialogue.run(session, '선금은 얼마인가요?', model)
        self.assertIn('지급하겠습니다', result['reply'])
        self.assertFalse(set(client_dialogue.HANDOVER_ITEMS) & set(result['state']['inventory']))

    def test_report_after_return_keeps_progress_and_suggests_unvisited_records(self):
        session = game.new_session('test')
        state = session['state']
        state['visited'].extend(['보스턴 글로브', '코빗 하우스'])
        state['clues'].append('확인한 현장 단서')
        model = Mock()
        for action in ('의뢰인에게 조사 상황을 이야기한다.', '의뢰인에게 조사 상황을 전달한다.'):
            with self.subTest(action=action):
                self.assertTrue(client_dialogue.applicable(state, 'gm', action))
                result = client_dialogue.run(session, action, model)
                self.assertIn('경과는 들었습니다', result['reply'])
                self.assertEqual(result['suggestions'], ['이동: 중앙 도서관'])
        model.assert_not_called()
        self.assertIn('중앙 도서관', result['reply'])
        self.assertEqual(state['clues'], ['확인한 현장 단서'])
        self.assertIn('코빗 하우스', state['visited'])

    def test_invalid_selector_output_uses_known_unknown_reply(self):
        session=game.new_session('test')
        result=client_dialogue.run(session,'?',Mock(return_value={'content':'[]'}))
        self.assertEqual(result['messages'][0]['text'],client_dialogue.FACTS['unknown'])


if __name__=='__main__': unittest.main()
