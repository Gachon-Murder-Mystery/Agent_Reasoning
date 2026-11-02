"""OpenAI and OpenCode adapters: validated action proposals and public-only narration.

Token accounting lives here because the engine ranks correct deductions by the
submitted `output_tokens`, fewest first. Every call is added to a cumulative
counter the player runtime reads when it submits `accuse`.
"""
import json
import math
import re
import threading
import time
import urllib.request
import urllib.error


class GatewayError(Exception):
    """code: auth | rate_limit | truncated | empty | transport | format | http"""
    def __init__(self, message, code='http', retry_after=None):
        super().__init__(message)
        self.code = code
        self.retry_after = retry_after


class Zen:
    def __init__(self,key='',model='gpt-6-luna',protocol='responses',connection='auto',session_id='',temperature=0.0,timeout=20):
        self.key=key.strip();self.model=model.strip() or 'gpt-6-luna';self.protocol=protocol
        self.connection=connection;self.session_id=str(session_id)[:100];self.temperature=temperature
        self.timeout=timeout
        if protocol not in ('responses','chat'):raise ValueError('지원하지 않는 API 방식입니다.')
        if connection not in ('auto','console','zen','openai'):raise ValueError('연결 서비스를 확인하세요.')
        # Players run in parallel threads; one client may be shared.
        self._lock=threading.Lock()
        self.calls=0;self.input_tokens=0;self.output_tokens=0;self.estimated_calls=0;self.elapsed=0.0
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
    def usage_totals(self):
        with self._lock:
            return {'calls':self.calls,'input':self.input_tokens,'output':self.output_tokens,
                    'estimated_calls':self.estimated_calls,'elapsed':round(self.elapsed,2)}
    def request(self,system,user,budget=700,timeout=None,json_only=False):
        """Return (text, usage). `timeout` must stay well inside the engine lease."""
        family=self.endpoint_family()
        if family=='messages':raise GatewayError('이 모델은 Anthropic Messages API 방식입니다. 현재 데모는 Responses와 Chat Completions를 지원합니다.','format')
        if family=='responses':
            endpoint='responses';body={'model':self.model,'instructions':system,'input':user,'max_output_tokens':budget,'temperature':self.temperature}
            if json_only:body['text']={'format':{'type':'json_object'}}
        else:
            endpoint='chat/completions';body={'model':self.model,'messages':[{'role':'system','content':system},{'role':'user','content':user}],'max_tokens':budget,'temperature':self.temperature}
            if json_only:body['response_format']={'type':'json_object'}
        roots={'console':'https://opencode.ai/inference/openai/v1/','zen':'https://opencode.ai/zen/v1/','openai':'https://api.openai.com/v1/'}
        root=roots[self.service]
        headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json','User-Agent':'ornate-express-gm/1.3'}
        if self.session_id:headers['X-Session-Id']=self.session_id
        req=urllib.request.Request(root+endpoint,data=json.dumps(body).encode(),headers=headers,method='POST')
        started=time.time()
        try:
            with urllib.request.urlopen(req,timeout=timeout or self.timeout) as res:data=json.load(res)
            if not isinstance(data,dict):raise GatewayError('제공자 응답 형식이 올바르지 않습니다.','format')
            if endpoint=='responses':
                text='\n'.join(p.get('text','') for o in data.get('output',[]) for p in o.get('content',[]) if p.get('type')=='output_text')
                cut=data.get('status')=='incomplete'
            else:
                choice=data.get('choices',[{}])[0]
                text=choice.get('message',{}).get('content','')
                cut=choice.get('finish_reason')=='length'
            usage=self._usage(data,system,user,text if isinstance(text,str) else '')
            self._account(usage,time.time()-started)
            # A truncated answer is never valid JSON, so say why: the caller
            # can raise the budget instead of re-sending the same prompt.
            if cut:raise GatewayError('출력이 토큰 상한에서 잘렸습니다.','truncated')
            if not isinstance(text,str) or not text.strip():raise GatewayError('모델에서 응답 문장을 받지 못했습니다.','empty')
            return text.strip(), usage
        except urllib.error.HTTPError as e:
            self._account(None,time.time()-started)
            reasons={
                400:'요청 형식을 확인하세요.',
                401:'API 키 인증 실패입니다. 선택한 제공자의 API 키가 유효한지 확인하세요.',
                402:'선택한 API 제공자의 결제 또는 크레딧 설정을 확인하세요.',
                403:'요청이 거부되었습니다. API 키의 권한과 모델 접근을 확인하세요.',
                404:'모델 ID 또는 API 방식을 확인하세요.',
                429:'요청 한도를 초과했습니다. 잠시 뒤 다시 시도하세요.',
            }
            codes={401:'auth',403:'auth',429:'rate_limit',400:'format',404:'format'}
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
            after=e.headers.get('Retry-After') if e.headers else None
            try:after=float(after) if after is not None else None
            except (TypeError,ValueError):after=None
            raise GatewayError(reason+detail,codes.get(e.code,'http'),after) from None
        except GatewayError:raise
        except (urllib.error.URLError,TimeoutError,OSError):
            self._account(None,time.time()-started)
            raise GatewayError('연결 실패 또는 응답 시간 초과','transport') from None
        except (ValueError,KeyError,IndexError,TypeError,AttributeError):
            self._account(None,time.time()-started)
            raise GatewayError('제공자 응답 형식을 읽지 못했습니다.','format') from None
    def _account(self,usage,elapsed):
        with self._lock:
            self.calls+=1;self.elapsed+=elapsed
            if not usage:return
            self.input_tokens+=usage.get('input') or 0
            self.output_tokens+=usage.get('output') or 0
            self.estimated_calls+=int(bool(usage.get('estimated')))
    @staticmethod
    def _usage(data,system,user,text):
        """Providers use different key names, and some omit usage entirely.

        Deduction rank goes to the fewest output tokens, so a missing count is
        estimated upward rather than dropped: silently counting 0 would hand
        this agent an unearned rank.
        """
        u=data.get('usage')
        if not isinstance(u,dict):
            return {'input':math.ceil(len(system+user)/2),'output':max(1,math.ceil(len(text)/2)),
                    'total':None,'reasoning':0,'estimated':True}
        out=u.get('output_tokens') or u.get('completion_tokens')
        details=u.get('output_tokens_details') or u.get('completion_tokens_details') or {}
        reasoning=details.get('reasoning_tokens') or 0 if isinstance(details,dict) else 0
        return {
            'input':u.get('input_tokens') or u.get('prompt_tokens'),
            'output':out,                 # reasoning tokens are already inside this
            'total':u.get('total_tokens'),
            'reasoning':reasoning,
            'estimated':False,
        }
