import copy
import json
import unittest
from unittest.mock import patch

from keeper import agent, game
from keeper.progress import movement_requested, repeats_paragraphs, repeats_previous, TurnResolutionError


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
                        'name': 'propose_transition', 'arguments': {'location': '2층', 'reason': '계단을 올라감'}}}]}
                return {'role': 'assistant', 'content': '입구로 이동 요청을 처리한다.'}
            if messages[0]['content'].startswith('Validate'):
                return {'content': json.dumps({'location': '2층', 'scene': '계단 위에 섰다.',
                    'action_result': '문을 열고 입구 안으로 이동했다.', 'action_status': 'resolved',
                    'next_options': ['주변을 살펴본다.']})}
            return {'content': json.dumps({'messages': [{'speaker_id': 'gm', 'text': '당신은 입구 안으로 들어왔습니다.'}]})}

        with patch.object(agent, '_evidence', return_value=[{'text': 'An accessible entrance.'}]), \
                patch.object(agent, 'chat', side_effect=fake_chat):
            response = agent.run_turn(session, '계단을 따라 올라가기')
        self.assertEqual(response['state']['location'], '2층')
        self.assertEqual(len(tool_calls), 3)
        self.assertNotIn('UNSUPPORTED_PRIOR_STORY_SENTINEL', json.dumps(tool_calls))

    def test_newspaper_research_cannot_silently_move_the_investigator(self):
        session = game.new_session('test')
        before = copy.deepcopy(session)
        responses = [
            {'role': 'assistant', 'content': '조사자가 신문사에 도착했다.'},
            {'content': json.dumps({'location': '보스턴 글로브 신문사',
                                    'scene_location': '보스턴 글로브 신문사',
                                    'scene': '편집장이 조사자를 맞이했다.'})},
        ]
        with patch.object(agent, '_evidence', return_value=[{'text': 'evidence'}]), \
                patch.object(agent, 'chat', side_effect=responses):
            with self.assertRaisesRegex(TurnResolutionError, '의뢰인의 사무실'):
                agent.run_turn(session, '신문사 기사 정보를 더 자세히 조사하기')
        self.assertEqual(session, before)

    def test_distinguishes_travel_from_calling_or_research(self):
        self.assertTrue(movement_requested('계단 위로 올라가기'))
        self.assertTrue(movement_requested('계단을 따라 올라가기'))
        self.assertTrue(movement_requested('계단을 내려간다'))
        self.assertFalse(movement_requested('계단의 상태를 살펴본다'))
        self.assertFalse(movement_requested('지하실로 내려가는 출입구를 찾는다.'))
        self.assertTrue(movement_requested('지하실로 내려가는 계단을 따라 내려간다.'))
        self.assertTrue(movement_requested('보스턴 글로브 신문사로 갈게'))
        self.assertTrue(movement_requested('보스턴 글로브 신문사를 방문할게'))
        self.assertFalse(movement_requested('보스턴 글로브 신문사에 전화해서 알아보기'))
        self.assertFalse(movement_requested('신문사 내부 분위기를 관찰해볼게'))

    def test_new_conversation_does_not_resubmit_previous_stealth_action(self):
        session = game.new_session('test')
        session['history'] = [
            {'role': 'user', 'content': '은밀 행동을 통해 지하실에 몰래 접근해보자'},
            {'role': 'assistant', 'content': '은밀행동 판정에 실패했다.'},
        ]
        state = session['state']
        state.update(scene_location=state['location'], scene='신문 기사가 펼쳐져 있다.',
                     last_action={'request': session['history'][0]['content'], 'result': '기사를 받았다.'})
        state['clues'] = ['코빗 하우스의 옛 주소']
        action = '기자에게 대화를 시도하다'
        responses = [
            {'role': 'assistant', 'content': '기자에게 말을 건다.'},
            {'content': json.dumps({'scene_location': session['state']['location'],
                                    'action_status': 'resolved', 'action_result': '기자에게 말을 건다.'})},
            {'content': json.dumps({'narration': '기자에게 말을 겁니다.', 'dialogue': ''})},
        ]
        with patch.object(agent, '_evidence', return_value=[{'text': 'evidence'}]), \
                patch.object(agent, 'chat', side_effect=responses) as chat:
            response = agent.run_turn(session, action)
        messages = chat.call_args_list[0].args[0]
        self.assertEqual([m['content'] for m in messages if m['role'] == 'user'], [action])
        self.assertEqual(response['rolls'], [])
        context = json.loads(messages[1]['content'])
        self.assertEqual(context['continuity']['current_scene'], '신문 기사가 펼쳐져 있다.')
        self.assertEqual(context['continuity']['recent_outcome'], {'result': '기사를 받았다.'})
        self.assertEqual(context['state']['clues'], ['코빗 하우스의 옛 주소'])
        self.assertNotIn(session['history'][0]['content'], json.dumps(messages, ensure_ascii=False))


    def test_historical_research_does_not_invent_a_document_in_the_room(self):
        session = game.new_session('test')
        state = session['state']
        state.update(location='어두운 방', scene_location='어두운 방', scene='방에 유리병이 있다.')
        responses = [
            {'role': 'assistant', 'content': '가문의 역사가 기록된 책을 읽는다.'},
            {'content': json.dumps({'scene_location': '어두운 방', 'action_status': 'resolved',
                                    'action_result': '책에서 가문의 역사를 확인했다.',
                                    'research_source': {'carried_item': '가문의 역사책',
                                                        'local_quote': 'This book lies open on the table.'}})},
        ]
        with patch.object(agent, '_evidence', return_value=[{'text': 'A glass bottle is visible.'}]), \
                patch.object(agent, 'chat', side_effect=responses) as model:
            result = agent.run_turn(session, '코빗 가문의 역사에 대한 자료를 조사한다')
        self.assertEqual(model.call_count, 2)
        self.assertEqual(result['state']['location'], '어두운 방')
        self.assertEqual(result['state']['scene'], '방에 유리병이 있다.')
        self.assertEqual(result['state']['clues'], [])
        self.assertIn('자료는 아직 확보하지 못했습니다', result['reply'])
        self.assertNotIn('가문의 역사를 확인했다', result['reply'])

    def test_climb_uses_real_consent_and_failed_check_does_not_move(self):
        session = game.new_session('test')
        state = session['state']
        state['location'] = '계단실'
        state['skills'].pop('오르기')
        game.upgrade_state(state)
        action = '계단 위로 올라가기'
        tool_call = {'role': 'assistant', 'tool_calls': [{'function': {
            'name': 'check_skill', 'arguments': {'skill': '오르기', 'difficulty': 'regular'},
        }}]}
        with patch.object(agent, '_evidence', return_value=[{'text': 'Damaged steps.'}]), \
                patch.object(agent, 'chat', return_value=tool_call):
            response = agent.run_turn(session, action)
        self.assertEqual(response['roll_request']['skill'], '오르기')
        self.assertEqual(response['roll_request']['target'], 20)
        review = {'location': '2층', 'scene_location': '계단실', 'action_status': 'blocked',
                  'action_result': '계단을 오르지 못했다.'}
        with patch.object(agent, 'chat', return_value={'content': json.dumps(review)}):
            after = agent._review(state, action, [], [{'location': '2층'}], [{'success': False}], '')
        self.assertEqual(after['location'], '계단실')
        self.assertNotIn('2층', after['visited'])

    def test_gm_hint_replaces_chinese_with_korean_actions(self):
        session = game.new_session('test')
        session['state']['location'] = '코빗의 방'
        before = copy.deepcopy(session)
        with patch.object(agent, 'chat', return_value={'content': '지금은 继续查看其他纸张.'}):
            result = agent.gm_question(session, '지금 뭘 할 수 있어?')
        self.assertIn('코빗의 방', result['reply'])
        self.assertIn('들어온 길로 돌아가', result['reply'])
        self.assertNotIn('继续', result['reply'])
        self.assertEqual(session, before)

    def test_office_return_recovers_legacy_save_without_model(self):
        session = game.new_session('test')
        state = session['state']
        state.update(location='책상 아래의 문을 통해 이동한 방', participants=[])
        state['visited'].extend(['코빗 하우스', state['location']])
        with patch.object(agent, 'chat') as model:
            result = agent.run_turn(session, '사무실로 돌아간다.')
        model.assert_not_called()
        self.assertEqual(result['state']['location'], '의뢰인의 사무실')
        self.assertIn('client', [person['id'] for person in result['contacts']])

    def test_echoed_speech_template_falls_back_to_reviewed_korean_result(self):
        session = game.new_session('test')
        template = {'content': json.dumps({
            'narration': 'physical observations / gestures / actual rolls only', 'dialogue': '',
        })}
        result_text = '단서를 보여줬지만 기록 열람 허가는 아직 받지 못했습니다.'
        responses = [
            {'role': 'assistant', 'content': '단서를 보여준다.'},
            {'content': json.dumps({'action_status': 'resolved', 'action_result': result_text})},
            template, template,
        ]
        with patch.object(agent, '_evidence', return_value=[{'text': 'evidence'}]), \
                patch.object(agent, 'chat', side_effect=responses):
            result = agent.run_turn(session, '코빗 하우스에서 얻은 단서를 보여준다.')
        self.assertEqual(result['messages'][0]['text'], result_text)
        self.assertNotIn('physical observations', session['reply'])

    def test_newspaper_basement_hint_exits_building_before_other_venues(self):
        session = game.new_session('test')
        session['state'].update(location='지하실의 모르그',
                                visited=['코빗의 방', '보스턴 글로브 신문사', '지하실의 모르그'])
        with patch.object(agent, 'chat') as model:
            result = agent.gm_question(session, '지금 뭘 할 수 있어?')
        model.assert_not_called()
        self.assertIn('신문사 1층', result['reply'])
        self.assertIn('신문사 밖', result['reply'])
        self.assertNotIn('코빗의 방', result['reply'])
        self.assertEqual(result['state']['location'], '지하실의 모르그')

    def test_recording_invented_house_entrance_symbol_corrects_the_scene(self):
        session = game.new_session('test')
        state = session['state']
        state.update(location='코빗 하우스 내부', scene_location='코빗 하우스 내부',
                     scene='벽에 새겨진 문양이 보인다.')
        with patch.object(agent, 'chat') as model:
            result = agent.run_turn(session, '문양의 형태를 기록하기')
        model.assert_not_called()
        self.assertEqual(result['state']['location'], '코빗 하우스 내부')
        self.assertEqual(result['state']['clues'], [])
        self.assertIn('기록할 문양은 확인되지 않았습니다', result['reply'])
        self.assertNotIn('문양', result['state']['scene'])

    def test_unidentified_door_does_not_open_or_move_on_unverified_key_claim(self):
        session = game.new_session('test')
        state = session['state']
        state.update(location='코빗 하우스 내부', scene_location='코빗 하우스 내부')
        state['inventory'].append('코빗 하우스 열쇠')
        responses = [
            {'role': 'assistant', 'tool_calls': [{'function': {'name': 'propose_transition',
                'arguments': {'location': '어두운 방', 'reason': '열쇠 사용'}}}]},
            {'role': 'assistant', 'content': '어두운 방을 살펴본다.'},
            {'content': json.dumps({'door_result': 'opened', 'location': '어두운 방',
                                    'scene_location': '어두운 방', 'action_status': 'resolved',
                                    'door_source': 'The key fits an unidentified interior door.'})},
        ]
        with patch.object(agent, '_evidence', return_value=[{'text': 'The house has a front entrance.'}]), \
                patch.object(agent, 'chat', side_effect=responses) as model:
            result = agent.run_turn(session, '열쇠를 사용해 닫힌 문의 방을 열어보기')
        self.assertEqual(model.call_count, 3)
        self.assertEqual(result['state']['location'], '코빗 하우스 내부')
        self.assertIn('열쇠가 그 문에 맞는지 확인할 수 없습니다', result['reply'])

    def test_opening_supported_door_is_not_the_same_as_entering(self):
        state = game.initial_state('test')
        review = {'door_result': 'opened', 'location': '다음 방', 'scene_location': '다음 방'}
        proposals = [{'location': '다음 방'}]
        with patch.object(agent, 'chat', return_value={'content': json.dumps(review)}):
            opened = agent._review(state, '문을 열어보기', [], proposals, [], '')
            entered = agent._review(state, '문을 열고 들어간다', [], proposals, [], '')
        self.assertEqual(opened['location'], state['location'])
        self.assertIn('문을 열었습니다', opened['last_action']['result'])
        self.assertEqual(entered['location'], '다음 방')
        self.assertIn('문을 열고 들어갔습니다', entered['last_action']['result'])

    def test_invented_room_recovery_keeps_other_clues_and_requires_explicit_return(self):
        session = game.new_session('test')
        state = session['state']
        state['location'] = '코빗 하우스 내부 - 다른 방'
        state['visited'].extend(['코빗 하우스 내부', state['location']])
        state['clues'] = ['신문사의 기사', agent.UNVERIFIED_HOUSE_CLUE]
        with patch.object(agent, 'chat') as model:
            hint = agent.gm_question(session, '지금 뭘 할 수 있어?')
            blocked = agent.run_turn(session, '흐릿한 벽면을 만져본다')
            returned = agent.run_turn(session, agent.HOUSE_RETREAT)
        model.assert_not_called()
        self.assertIn(agent.HOUSE_RETREAT, hint['reply'])
        self.assertEqual(blocked['state']['location'], state['location'])
        self.assertEqual(returned['state']['location'], '코빗 하우스 내부')
        self.assertEqual(returned['state']['clues'], ['신문사의 기사'])

    def test_touching_wall_cannot_commit_a_proposed_move(self):
        state = game.initial_state('test')
        review = {'location': '다른 방', 'scene_location': state['location'],
                  'action_status': 'resolved', 'action_result': '벽을 만졌다.'}
        with patch.object(agent, 'chat', return_value={'content': json.dumps(review)}):
            after = agent._review(state, '흐릿한 벽면을 만져본다', [], [{'location': '다른 방'}], [], '')
        self.assertEqual(after['location'], state['location'])
        self.assertNotIn('다른 방', after['visited'])

    def test_named_front_door_uses_awarded_key_and_respects_requested_entry(self):
        session = game.new_session('test')
        session['state']['location'] = '코빗 하우스'
        session['state']['inventory'].append('코빗 하우스 열쇠')
        with patch.object(agent, 'chat') as model:
            opened = agent.run_turn(session, '정문을 열어본다.')
            entered = agent.run_turn(session, '정문을 지나 집 안으로 들어간다.')
        model.assert_not_called()
        self.assertEqual(opened['state']['location'], '코빗 하우스')
        self.assertIn('열쇠로 정문을 열었습니다', opened['reply'])
        self.assertEqual(entered['state']['location'], '코빗 하우스 내부')

    def test_house_arrival_uses_public_exterior_instead_of_invented_props(self):
        state = game.initial_state('test')
        after = dict(state, location='코빗 하우스', scene='정원에서 낡은 간판이 흔들린다.')
        text = agent._house_exterior_result(state, after, '코빗 하우스로 간다.')
        self.assertIn('양옆에는 더 높은 사무실 건물', text)
        self.assertNotIn('간판', after['scene'])
        self.assertEqual(after['last_action']['result'], text)

    def test_house_interior_cannot_observe_basement_before_entering(self):
        session = game.new_session('test')
        session['state']['location'] = '코빗 하우스 내부'
        with patch.object(agent, 'chat') as model:
            found = agent.run_turn(session, '지하실로 내려가는 출입구를 찾는다.')
            observed = agent.run_turn(session, '지하실을 손전등으로 살펴본다.')
        model.assert_not_called()
        self.assertIn('문을 찾았습니다', found['reply'])
        self.assertIn('아직 지하실에 들어가지 않았습니다', observed['reply'])
        self.assertEqual(observed['state']['location'], '코빗 하우스 내부')

    def test_basement_descent_requires_consent_and_pushed_failure_costs_hp(self):
        session = game.new_session('test')
        session['state']['location'] = '코빗 하우스 내부'
        descent = '지하실 문을 열고 지하실로 내려간다.'
        pushing = '위험을 감수하고 지하실 계단을 다시 내려간다.'
        failed = {'id': 'skill', 'kind': 'skill', 'result': {
            'skill': '오르기', 'success': False}}
        self.assertEqual(agent.run_turn(session, descent)['roll_request']['skill'], '오르기')
        first = agent.run_turn(session, descent, approved_rolls=[failed])
        self.assertEqual(first['state']['location'], '코빗 하우스 내부')
        self.assertNotIn('roll_request', agent.run_turn(session, descent))
        self.assertEqual(agent.run_turn(session, pushing)['roll_request']['skill'], '오르기')
        damage_request = agent.run_turn(session, pushing, approved_rolls=[failed])
        self.assertEqual(damage_request['roll_request']['kind'], 'dice')
        damage = {'id': 'dice', 'kind': 'dice', 'result': {'total': 3, 'dice': [3]}}
        result = agent.run_turn(session, pushing, approved_rolls=[failed, damage])
        self.assertEqual(result['state']['location'], '코빗 하우스 지하실')
        self.assertEqual(result['state']['hp'], 9)

    def test_solution_requires_public_observation_confrontation_and_successful_roll(self):
        session = game.new_session('test')
        session['state']['visited'].append('코빗 하우스 내부')
        report = '사건을 해결했다고 의뢰인에게 보고한다.'
        premature = agent.run_turn(session, report)
        self.assertNotIn('ending', premature['state'])
        session['state']['location'] = '코빗 하우스 지하실'
        action = '달려드는 인물에게 맞서 싸운다.'
        early = agent.run_turn(session, action)
        self.assertNotIn('roll_request', early)
        observation = agent.run_turn(session, '나무 판재가 붙은 벽을 살펴본다.')
        self.assertIn('빈 듯한 소리', observation['reply'])
        reveal = agent.run_turn(session, '속이 빈 부분의 판재를 들춰본다.')
        self.assertIn('달려듭니다', reveal['reply'])
        self.assertEqual(agent.run_turn(session, action)['roll_request']['skill'], '근접전')
        retry = agent.run_turn(session, '자세를 바꿔 코빗에게 다시 맞선다.')
        self.assertEqual(retry['roll_request']['skill'], '근접전')
        approved = {'id': 'melee', 'kind': 'skill', 'result': {'success': True}}
        victory = agent.run_turn(session, action, approved_rolls=[approved])
        self.assertNotIn('ending', victory['state'])
        agent.run_turn(session, '사무실로 돌아간다.')
        final = agent.run_turn(session, report, 'client')
        self.assertEqual(final['state']['ending'], 'case_solved')
        self.assertIn('의뢰 보너스 30달러', final['state']['inventory'])

    def test_house_exterior_suggested_route_reaches_rear_and_returns(self):
        session = game.new_session('test')
        session['state']['location'] = '코빗 하우스'
        with patch.object(agent, 'chat') as model:
            rear = agent.run_turn(session, '집 옆을 따라 뒤쪽으로 가본다.')
            front = agent.run_turn(session, '정문 앞으로 돌아간다.')
        model.assert_not_called()
        self.assertEqual(rear['state']['location'], '코빗 하우스 뒤편')
        self.assertEqual(front['state']['location'], '코빗 하우스')

    def test_front_door_does_not_disclose_basement_before_any_observation(self):
        session = game.new_session('test')
        state = session['state']
        state['location'] = '코빗 하우스 정문'
        state['inventory'].append('코빗 하우스 열쇠')
        entered = agent.run_turn(session, '정문을 열고 집 안으로 들어간다.')
        self.assertNotIn('지하실', json.dumps(entered, ensure_ascii=False))
        looked = agent.run_turn(session, '현관 주변을 살펴 어떤 공간으로 이어지는지 확인한다.')
        self.assertIn('닫힌 문', looked['reply'])
        self.assertNotIn('지하실', looked['reply'])
        closer = agent.run_turn(session, '닫힌 문을 가까이 살펴본다.')
        self.assertNotIn('지하실', closer['reply'])
        opened = agent.run_turn(session, '닫힌 문을 열어 어디로 이어지는지 확인한다.')
        self.assertIn('아래로 이어지는 낡은 계단', opened['reply'])
        self.assertEqual(opened['state']['location'], '코빗 하우스 내부')

    def test_globe_access_and_library_record_are_separate_sources(self):
        session = game.new_session('test')
        agent.run_turn(session, '보스턴 글로브 신문사로 간다.')
        denied = agent.run_turn(session, '기록실로 이동한다.')
        self.assertEqual(denied['state']['location'], '보스턴 글로브 신문사')
        self.assertEqual(denied['state']['clues'], [])
        success = [{'id': 'roll', 'kind': 'skill', 'result': {'success': True}}]
        agent.run_turn(session, '편집장을 설득해 기사 열람을 요청한다.', approved_rolls=success)
        agent.run_turn(session, '기록실로 이동한다.')
        globe = agent.run_turn(session, '코빗 하우스 관련 기사를 찾는다.')
        self.assertEqual(globe['state']['clues'], [agent.GLOBE_CLUE])
        agent.run_turn(session, '신문사 출입구로 돌아간다.')
        agent.run_turn(session, '중앙 도서관으로 간다.')
        failed = [{'id': 'roll', 'kind': 'skill', 'result': {'success': False}}]
        searched = agent.run_turn(session, '코빗 하우스에 관한 오래된 기록을 찾는다.', approved_rolls=failed)
        self.assertEqual(searched['state']['clues'], [agent.GLOBE_CLUE])
        library = agent.run_turn(session, '도서관에서 반나절을 더 들여 오래된 기록을 찾는다.',
                                 approved_rolls=success)
        self.assertEqual(library['state']['library_half_days'], 2)
        self.assertEqual(library['state']['clues'], [agent.GLOBE_CLUE, agent.LIBRARY_CLUE])

    def test_negative_movement_does_not_cancel_other_affirmative_clause(self):
        self.assertFalse(movement_requested('정문을 열되 집 안으로 들어가지 않는다.'))
        self.assertFalse(movement_requested('지하실 문을 열되 계단을 내려가지 않는다.'))
        self.assertTrue(movement_requested('집 안으로 들어가지 않고 정문 앞으로 돌아간다.'))

    def test_opening_without_entering_keeps_front_and_basement_positions(self):
        session = game.new_session('test')
        session['state']['location'] = '코빗 하우스'
        session['state']['inventory'].append('코빗 하우스 열쇠')
        front = agent.run_turn(session, '정문을 열되 집 안으로 들어가지 않는다.')
        self.assertEqual(front['state']['location'], '코빗 하우스')
        self.assertTrue(front['state']['front_door_open'])
        agent.run_turn(session, '정문을 지나 집 안으로 들어간다.')
        agent.run_turn(session, '지하실로 내려가는 출입구를 찾는다.')
        cellar = agent.run_turn(session, '지하실 문을 열되 계단을 내려가지 않는다.')
        self.assertEqual(cellar['state']['location'], '코빗 하우스 내부')
        self.assertTrue(cellar['state']['basement_door_open'])
        self.assertNotIn('roll_request', cellar)

    def test_remote_key_use_and_unauthorized_reading_do_not_reveal_state(self):
        session = game.new_session('test')
        with patch.object(agent, 'chat') as model:
            key = agent.run_turn(session, '사무실에서 코빗 하우스 열쇠를 사용해 지하실 문을 연다.')
            agent.run_turn(session, '보스턴 글로브 신문사로 간다.')
            blocked = agent.run_turn(session, '기록실 안의 기사철을 여기서 훔쳐 읽는다.')
        model.assert_not_called()
        self.assertEqual(key['state']['location'], '의뢰인의 사무실')
        self.assertIn('현장에서만', key['reply'])
        self.assertEqual(blocked['state']['location'], '보스턴 글로브 신문사')
        self.assertEqual(blocked['state']['clues'], [])
        self.assertIn('허가', blocked['reply'])

    def test_landlord_attack_suspends_briefing_until_apology(self):
        session = game.new_session('test')
        before = session['state']['hp']
        attack = agent.run_turn(session, '아무 이유 없이 의뢰인을 주먹으로 공격한다.', 'client')
        self.assertTrue(attack['state']['client_agitated'])
        self.assertEqual(attack['state']['hp'], before)
        blocked = agent.run_turn(session, '의뢰를 수락하고 열쇠와 주소를 받는다.', 'client')
        self.assertNotIn('코빗 하우스 열쇠', blocked['state']['inventory'])
        agent.run_turn(session, '의뢰인에게 사과한다.', 'client')
        self.assertNotIn('client_agitated', session['state'])

    def test_newspaper_disruption_revokes_access_until_apology(self):
        session = game.new_session('test')
        session['state'].update(location='보스턴 글로브 신문사', globe_access=True)
        ejected = agent.run_turn(session, '신문사에서 소리를 지르고 책상을 엎으며 깽판을 친다.')
        self.assertEqual(ejected['state']['location'], '보스턴 글로브 신문사 밖')
        self.assertFalse(ejected['state']['globe_access'])
        agent.run_turn(session, '중앙 도서관으로 간다.')
        blocked = agent.run_turn(session, '보스턴 글로브 신문사로 간다.')
        self.assertEqual(blocked['state']['location'], '보스턴 글로브 신문사 밖')
        agent.run_turn(session, '신문사 직원에게 소란을 피운 일을 사과한다.')
        self.assertFalse(session['state']['globe_access'])
        self.assertNotIn('globe_disruption', session['state'])

    def test_no_attack_dialogue_does_not_request_melee(self):
        session = game.new_session('test')
        state = session['state']
        state.update(location='코빗 하우스 지하실', basement_discovery='confrontation')
        result = agent.run_turn(session, '공격하지 않고 두 손을 들어 코빗에게 왜 떠나지 않았는지 묻는다.')
        self.assertIn('대답하지 않고', result['reply'])
        self.assertEqual(result['rolls'], [])
        self.assertEqual(result['state']['hp'], state['hp'])
        self.assertEqual(result['state']['basement_discovery'], 'confrontation')

    def test_repeated_corbitt_dialogue_remains_noncombat(self):
        session = game.new_session('test')
        session['state'].update(location='코빗 하우스 지하실', basement_discovery='confrontation')
        before_hp = session['state']['hp']
        responses = []
        for action in ('공격하지 않고 이유를 묻는다.', '싸울 뜻이 없다고 말하고 대화를 제안한다.',
                       '다시 휴전을 제안한다.', '한 번 더 대화를 제안한다.'):
            result = agent.run_turn(session, action)
            self.assertNotIn('roll_request', result)
            self.assertEqual(result['rolls'], [])
            self.assertEqual(result['state']['hp'], before_hp)
            self.assertEqual(result['state']['clues'], [])
            self.assertEqual(result['state']['basement_discovery'], 'confrontation')
            responses.append(result['reply'])
        self.assertEqual(len(set(responses)), 3)
        self.assertEqual(session['state']['corbitt_talk_attempts'], 4)

    def test_attack_on_verified_contact_does_not_become_model_generated_combat(self):
        session = game.new_session('test')
        session['state'].update(location='다른 조사 장소', participants=[
            {'id': 'witness', 'name': '목격자', 'available': True}])
        with patch.object(agent, 'chat') as model:
            result = agent.run_turn(session, '목격자를 공격한다.', 'witness')
        model.assert_not_called()
        self.assertNotIn('roll_request', result)
        self.assertIn('전투를 처리하지 않습니다', result['reply'])

    def test_building_ignition_attempt_preserves_world_state_in_each_venue(self):
        cases = (
            ('코빗 하우스', '코빗의 집에 불을 붙인다.'),
            ('보스턴 글로브 신문사', '신문사에 불붙인다.'),
            ('중앙 도서관', '도서관에 불을 지른다.'),
        )
        for location, action in cases:
            with self.subTest(location=location):
                session = game.new_session('test')
                session['state']['location'] = location
                before = copy.deepcopy(session['state'])
                with patch.object(agent, 'chat') as model:
                    response = agent.run_turn(session, action)
                model.assert_not_called()
                self.assertEqual(response['rolls'], [])
                self.assertIn('불은 나지 않았고', response['reply'])
                self.assertEqual({k: v for k, v in response['state'].items() if k != 'last_action'}, before)
        self.assertFalse(agent._building_fire_requested('촛불을 켠다.'))
        self.assertFalse(agent._building_fire_requested('도서관에 불을 붙이지 않는다.'))

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
