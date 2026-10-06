import json
import io
import urllib.error
import random
import tempfile
import unittest
from unittest.mock import patch
from engine import new_game, apply, public, llm_context, local_intent, validate_intent, NPC_ROOMS
from app import Store
from llm import Zen, GatewayError

class Fixed:
    def __init__(self,values):self.values=iter(values)
    def randint(self,a,b):return next(self.values)

class GameTests(unittest.TestCase):
    def test_all_sources_solvable_and_no_truth_eliminated(self):
        for _ in range(40):
            g=new_game()
            for room in range(8):
                apply(g,{'kind':'move','place':room})
                for _ in range(4):
                    apply(g,{'kind':'prepare'})
                    apply(g,{'kind':'check','target':None,'approach':'compassion'},'이곳의 기록을 조사한다')
                    apply(g,{'kind':'roll'},rng=Fixed([6,6,6]))
            self.assertEqual(len(g['clues']),25)
            self.assertEqual(len({c['id'] for c in g['clues']}),25)
            self.assertTrue(all(c['index']!=g['truth'][c['category']] for c in g['clues']))
            apply(g,{'kind':'accuse',**g['truth']})
            self.assertTrue(g['won'])
    def test_success_levels_equality_and_cost(self):
        for values,expected,count in [([4,4,4],0,0),([5,4,4],1,1),([5,6,4],2,1),([5,5,5],3,2)]:
            g=new_game(role=0,stat=4);apply(g,{'kind':'move','place':2});apply(g,{'kind':'prepare'})
            apply(g,{'kind':'check','target':None,'approach':'compassion'},'記録を調査')
            start=g['turn'];apply(g,{'kind':'roll'},rng=Fixed(values))
            rolls=[m for m in g['messages'] if m['kind']=='roll']
            self.assertEqual(rolls[-1]['successes'],expected);self.assertEqual(len(g['clues']),count)
            self.assertEqual(g['turn']-start,2 if expected==1 else 1)
        g=new_game(stat=4);apply(g,{'kind':'check','approach':'brutality','target':None},'力');apply(g,{'kind':'roll'},rng=Fixed([4,3]))
        self.assertEqual([m for m in g['messages'] if m['kind']=='roll'][-1]['successes'],1)
    def test_pending_cannot_be_bypassed_and_cancel(self):
        g=new_game();apply(g,{'kind':'check','target':None},'調査');before=public(g)
        apply(g,{'kind':'move','place':7});self.assertEqual(g['room'],before['room']);self.assertTrue(g['pending'])
        apply(g,{'kind':'cancel'});self.assertIsNone(g['pending'])
    def test_npc_presence_and_memory(self):
        g=new_game(role=0);apply(g,{'kind':'check','target':10},'車掌');self.assertIsNone(g['pending'])
        apply(g,{'kind':'find','target':10},'차장을 찾아가 자정의 행적을 묻는다')
        self.assertEqual(g['room'],NPC_ROOMS[10]);self.assertEqual(g['pending']['target'],10)
        self.assertEqual(g['npc_memory']['10'][0]['who'],'player')
    def test_public_and_llm_have_no_private_state(self):
        g=new_game();marker='PRIVATE_SENTINEL';g['all_clues'][0]['text']=marker
        for p in (public(g),llm_context(g)):
            self.assertNotIn('truth',p);self.assertNotIn('all_clues',p);self.assertNotIn(marker,json.dumps(p))
    def test_wrong_guess_and_incomplete_guess(self):
        g=new_game();apply(g,{'kind':'accuse','suspect':0});self.assertFalse(g['ended']);self.assertEqual(g['wrong'],0)
        wrong={**g['truth'],'suspect':(g['truth']['suspect']+1)%12}
        apply(g,{'kind':'accuse',**wrong});self.assertFalse(g['ended']);apply(g,{'kind':'accuse',**wrong});self.assertTrue(g['ended']);self.assertFalse(g['won'])
    def test_offline_intents(self):
        g=new_game()
        for text,kind in [('주방 칸으로 이동한다','move'),('차장을 찾아간다','find'),('차장의 알리바이를 묻는다','check'),('서류를 조사한다','check'),('질문을 정리한다','prepare')]:self.assertEqual(local_intent(g,text)['kind'],kind)
        with self.assertRaises(ValueError):validate_intent({'kind':'move','place':800},g)
    def test_persistence_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            store=Store(d+'/state.sqlite');g=new_game();apply(g,{'kind':'move','place':3});apply(g,{'kind':'check','target':None},'조사');store.save(g)
            loaded=Store(d+'/state.sqlite').get(g['id']);self.assertEqual(loaded,g)
    def test_llm_json_validation_and_fallback(self):
        g=new_game();z=Zen('fake')
        with patch.object(z,'request',return_value='{"kind":"move","place":3}'):
            intent,notice=z.intent(g,'주방에 가자');self.assertEqual(intent['place'],3);self.assertIsNone(notice)
        with patch.object(z,'request',return_value='{"kind":"move","place":99}'):
            intent,notice=z.intent(g,'주방 칸으로 이동한다');self.assertEqual(intent['place'],3);self.assertTrue(notice)
        with patch.object(z,'request',side_effect=GatewayError('연결 실패')):
            result,notice=z.narrate(g,'안녕','기본 응답');self.assertEqual(result,'기본 응답');self.assertEqual(notice,'연결 실패')
    def test_gateway_error_diagnostic_and_credential_redaction(self):
        cases=[
            (401,{},'인증 실패'),
            (402,{},'크레딧'),
            (403,{'error':{'code':'model_access_denied','message':'Model is disabled for this workspace'}},'model_access_denied'),
            (404,{},'모델 ID'),
            (429,{},'한도'),
        ]
        for status,payload,expected in cases:
            exc=urllib.error.HTTPError('https://opencode.ai/zen/v1/responses',status,'denied',{},io.BytesIO(json.dumps(payload).encode()))
            with self.subTest(status=status),patch('urllib.request.urlopen',side_effect=exc):
                with self.assertRaises(GatewayError) as caught:Zen('private-test-key').request('system','user')
                self.assertIn(expected,str(caught.exception));self.assertNotIn('private-test-key',str(caught.exception))
        payload={'error':{'message':'Rejected credential: private-test-key'}}
        exc=urllib.error.HTTPError('https://opencode.ai/zen/v1/responses',403,'denied',{},io.BytesIO(json.dumps(payload).encode()))
        with patch('urllib.request.urlopen',side_effect=exc):
            with self.assertRaises(GatewayError) as caught:Zen('private-test-key').request('system','user')
            self.assertIn('[키 숨김]',str(caught.exception));self.assertNotIn('private-test-key',str(caught.exception))
    def test_console_key_routes_luna_to_console_responses(self):
        class Response:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,*args):return json.dumps({'output':[{'content':[{'type':'output_text','text':'ok'}]}]}).encode()
        z=Zen('oc_sk_example-secret-key','gpt-6-luna',connection='auto',session_id='game-abc')
        with patch('urllib.request.urlopen',return_value=Response()) as call:
            self.assertEqual(z.request('system','user'),'ok')
        req=call.call_args.args[0]
        self.assertEqual(req.full_url,'https://opencode.ai/inference/openai/v1/responses')
        self.assertNotIn('X-opencode-session',req.headers)
        self.assertEqual(req.headers['Authorization'],'Bearer oc_sk_example-secret-key')
        self.assertEqual(json.loads(req.data)['model'],'gpt-6-luna')
    def test_openai_key_routes_to_official_api(self):
        class Response:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,*args):return json.dumps({'output':[{'content':[{'type':'output_text','text':'ok'}]}]}).encode()
        z=Zen('sk-test-secret','gpt-6-luna',connection='openai')
        with patch('urllib.request.urlopen',return_value=Response()) as call:
            self.assertEqual(z.request('system','user'),'ok')
        req=call.call_args.args[0]
        self.assertEqual(req.full_url,'https://api.openai.com/v1/responses')
        self.assertEqual(req.headers['Authorization'],'Bearer sk-test-secret')
        self.assertEqual(json.loads(req.data)['model'],'gpt-6-luna')
        self.assertEqual(z.service,'openai')
    def test_console_and_zen_keys_route_to_correct_services(self):
        self.assertEqual(Zen('oc_sk_test','kimi-k3',protocol='chat',connection='auto',session_id='s').endpoint_family(),'chat')
        self.assertEqual(Zen('oc_sk_test','minimax-m3',connection='auto',session_id='s').endpoint_family(),'chat')
        self.assertEqual(Zen('oc_sk_test','gpt-6-luna',protocol='chat').endpoint_family(),'responses')
        self.assertEqual(Zen('oc_sk_test','claude-sonnet-5').endpoint_family(),'messages')
        self.assertEqual(Zen('other-key','gpt-6-luna',connection='auto').service,'zen')
        self.assertEqual(Zen('oc_sk_key','gpt-6-luna',connection='auto').service,'console')
    def test_api_payload_and_response_families(self):
        class Response:
            def __init__(self,payload):self.payload=json.dumps(payload).encode();self.offset=0
            def read(self,*args):return self.payload
            def __enter__(self):return self
            def __exit__(self,*args):pass
        for protocol,model,payload in [('responses','gpt-6-luna',{'output':[{'content':[{'type':'output_text','text':'ok'}]}]}),('chat','kimi-k3',{'choices':[{'message':{'content':'ok'}}]})]:
            with patch('urllib.request.urlopen',return_value=Response(payload)) as request:
                self.assertEqual(Zen('fake',model=model,protocol=protocol).request('system','user'),'ok')
                req=request.call_args.args[0];self.assertEqual(req.headers['Authorization'],'Bearer fake')
                self.assertNotIn('fake',req.data.decode());self.assertTrue(req.full_url.startswith('https://opencode.ai/zen/v1/'));self.assertEqual(req.headers['User-agent'],'ornate-express-gm/1.2')

if __name__=='__main__':unittest.main()
