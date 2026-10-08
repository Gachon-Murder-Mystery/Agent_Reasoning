"""OpenAI and OpenCode adapters: validated action proposals and public-only narration."""
import json
import re
import urllib.request
import urllib.error
# from engine import llm_context, local_intent, validate_intent, PLACES

class GatewayError(Exception):pass

class Zen:
    def __init__(self,key='',model='gpt-6-luna',protocol='responses',connection='auto',session_id='',temperature=0.0):
        self.key=key.strip();self.model=model.strip() or 'gpt-6-luna';self.protocol=protocol
        self.connection=connection;self.session_id=str(session_id)[:100];self.temperature=temperature
        if protocol not in ('responses','chat'):raise ValueError('지원하지 않는 API 방식입니다.')
        if connection not in ('auto','console','zen','openai'):raise ValueError('연결 서비스를 확인하세요.')
    @property
    def service(self):
        if self.connection=='openai':return 'openai'
        if self.connection=='auto':return 'console' if self.key.startswith('oc_sk_') else 'zen'
        return self.connection
    def endpoint_family(self):
        if self.model.startswith(('gpt-','grok-','muse-spark-')):return 'responses'
        if self.model.startswith(('kimi-','glm-','deepseek-','mimo-','longcat-','hy','space-bunny','minimax-','nemotron','ling-')):return 'chat'
        if self.model.startswith(('claude-','qwen3.8-flash','qwen3.7-plus','qwen3.6-plus','qwen3.5-plus')):return 'messages'
        return self.protocol
    def request(self,system,user,budget=700):
        family=self.endpoint_family()
        if family=='messages':raise GatewayError('이 모델은 Anthropic Messages API 방식입니다. 현재 데모는 Responses와 Chat Completions를 지원합니다.')
        if family=='responses':
            endpoint='responses';body={'model':self.model,'instructions':system,'input':user,'max_output_tokens':budget,'temperature':self.temperature}
        else:
            endpoint='chat/completions';body={'model':self.model,'messages':[{'role':'system','content':system},{'role':'user','content':user}],'max_tokens':budget,'temperature':self.temperature}
        roots={'console':'https://opencode.ai/inference/openai/v1/','zen':'https://opencode.ai/zen/v1/','openai':'https://api.openai.com/v1/'}
        root=roots[self.service]
        headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json','User-Agent':'ornate-express-gm/1.2'}
        req=urllib.request.Request(root+endpoint,data=json.dumps(body).encode(),headers=headers,method='POST')
        try:
            with urllib.request.urlopen(req,timeout=30) as res:data=json.load(res)
            if not isinstance(data,dict):raise GatewayError('제공자 응답 형식이 올바르지 않습니다.')
            if endpoint=='responses':
                text='\n'.join(p.get('text','') for o in data.get('output',[]) for p in o.get('content',[]) if p.get('type')=='output_text')
            else:text=data.get('choices',[{}])[0].get('message',{}).get('content','')
            if not isinstance(text,str) or not text.strip():raise GatewayError('모델에서 응답 문장을 받지 못했습니다.')
            return text.strip(), self._usage(data)
        except urllib.error.HTTPError as e:
            reasons={
                400:'요청 형식을 확인하세요.',
                401:'API 키 인증 실패입니다. 선택한 제공자의 API 키가 유효한지 확인하세요.',
                402:'선택한 API 제공자의 결제 또는 크레딧 설정을 확인하세요.',
                403:'요청이 거부되었습니다. API 키의 권한과 모델 접근을 확인하세요.',
                404:'모델 ID 또는 API 방식을 확인하세요.',
                429:'요청 한도를 초과했습니다. 잠시 뒤 다시 시도하세요.',
            }
            detail=''
            try:
                payload=json.loads(e.read(4096).decode('utf-8','replace'))
                err=payload.get('error',{}) if isinstance(payload,dict) else {}
                if isinstance(err,dict):
                    code=err.get('code') or err.get('type')
                    message=err.get('message')
                elif isinstance(err,str):code=None;message=err
                else:code=None;message=None
                parts=[]
                if isinstance(code,str):parts.append(code[:100])
                if isinstance(message,str):parts.append(message[:220])
                detail=' — '+' / '.join(parts) if parts else ''
            except (ValueError,TypeError,AttributeError):
                detail=''
            # Never echo the submitted credential if a gateway unexpectedly repeats it.
            detail=detail.replace(self.key,'[키 숨김]') if self.key else detail
            detail=re.sub(r'(?i)bearer\s+[^\s,;]+','Bearer [숨김]',detail)
            detail=re.sub(r'(?i)(oc_sk_[a-z0-9_-]{8,}|sk-[a-z0-9_-]{8,}|oc-[a-z0-9_-]{8,})','[키 숨김]',detail)
            provider={'openai':'OpenAI','console':'OpenCode Console','zen':'OpenCode Zen'}[self.service]
            reason=reasons.get(e.code,f'{provider} 응답 오류 (HTTP {e.code})')
            raise GatewayError(reason+detail) from None
        except (urllib.error.URLError,TimeoutError,OSError):raise GatewayError('연결 실패 또는 응답 시간 초과') from None
        except (ValueError,KeyError,IndexError,TypeError,AttributeError):raise GatewayError('제공자 응답 형식을 읽지 못했습니다.') from None
    @staticmethod
    def _usage(data):
        """Providers use different key names, so check both."""
        u = data.get('usage')
        if not isinstance(u, dict):
            return None
        return {
            'input': u.get('input_tokens') or u.get('prompt_tokens'),
            'output': u.get('output_tokens') or u.get('completion_tokens'),
            'total': u.get('total_tokens'),
        }
