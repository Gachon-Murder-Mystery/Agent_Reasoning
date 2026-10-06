"""Authoritative game state. No API keys or private solution sent to an LLM."""
import random
import secrets

SUSPECTS=['알렉스 누아르 (탐정)','케이시 스칼렛 (배우)','에이버리 재스퍼 (사업가)','로언 앰버 (사기꾼)','라일리 올리브 (교수)','다코타 네이비 (대령)','케이시 더블루 (의사)','바이올렛 모레티 (상속인)','캐머런 펄 (기술자)','캉디드 블랑슈 (요리사)','퀸 슬레이트 (차장)','파커 에보니 (기자)']
WEAPONS=['골동품 단검','유리 조각','드라이버','만년필','편지칼','식칼','총검','머리핀']
PLACES=['라운지 칸','침대 칸','식당 칸','주방 칸','수하물 칸','금고 칸','전망 칸','기관실']
CAT={'suspect':SUSPECTS,'weapon':WEAPONS,'place':PLACES}
JOBS=['탐정','배우','사업가','사기꾼','교수','대령','의사','상속인','기술자','요리사','차장','기자']
NPC_ROOMS=[0,1,5,4,6,0,1,5,7,3,2,6]
PERSONALITIES=['짧고 신중하게 말을 고른다','불안한 웃음 뒤에 감정을 감춘다','손익을 따지며 조심스럽게 답한다','농담으로 질문을 돌린다','관찰한 내용을 차분히 설명한다','단정하고 직설적으로 말한다','상태를 꼼꼼히 살피며 말한다','품위를 지키며 날카롭게 되묻는다','손끝에 묻은 기름을 닦으며 생각한다','거친 말투 속에 걱정이 묻어난다','예의를 지키며 승객을 진정시킨다','질문을 메모하며 말을 이어간다']
BACKGROUNDS=['숨겨진 목적을 가진 비번 수사관','스캔들을 피해 도망치는 젊은 배우','수상한 거래에 관여하는 자본가','사기 전력이 있는 협잡꾼','음산한 평판을 가진 과학자','엄격한 성격의 퇴역 장교','비밀스러운 이중생활을 하는 유명 외과의사','많은 비밀을 간직한 부유한 미망인','기묘한 연구 프로젝트를 진행하는 발명가','불같은 성격의 요리사','친절하지만 미스터리한 과거를 가진 열차 직원','세부사항을 놓치지 않는 예리한 기자']
ROOMS=[
 '벨벳 의자와 식어가는 차. 창문에 비친 승객들은 서로의 눈을 피합니다.',
 '좁은 복도에 황동 객실 번호가 늘어서 있습니다. 문틈 사이로 불빛이 새어 나옵니다.',
 '하얀 식탁보 위의 은식기가 열차의 진동에 부딪힙니다. 차장이 승객들을 모으고 있습니다.',
 '희미한 증기와 버터 냄새. 도마와 조리 도구가 정돈되어 있습니다.',
 '가죽 트렁크들이 선반에 빼곡합니다. 바닥을 타고 열차의 진동이 전해집니다.',
 '금속 문 옆에 보관 장부가 놓여 있습니다. 피해자의 재산도 이곳에 있다고 합니다.',
 '커다란 유리창 너머로 황무지가 펼쳐집니다. 창가 의자 사이에 작은 테이블이 놓여 있습니다.',
 '피스톤의 굉음 때문에 목소리를 높여야 합니다. 계기판의 바늘이 흔들립니다.'
]

def add(g,kind,text,**extra):
    e={'kind':kind,'text':text,'seq':len(g['messages'])+1,**extra};g['messages'].append(e);return e

def new_game(name='여행자',role=0,stat=4,goal='사건 해결'):
    if not 0<=role<12 or stat not in range(2,6):raise ValueError('캐릭터 설정을 확인하세요.')
    name=str(name).strip()[:30] or '여행자'
    g={'version':2,'id':secrets.token_urlsafe(24),'name':name,'role':role,'stat':stat,'goal':str(goal)[:120],
       'truth':{k:secrets.randbelow(len(v)) for k,v in CAT.items()},'room':0,'target':None,'turn':0,
       'prepared':False,'pending':None,'ended':False,'won':False,'wrong':0,'messages':[],'clues':[],
       'all_clues':[],'visited':[0],'npc_memory':{str(i):[] for i in range(12)}}
    # Every negative has a stable physical source, also known by NPCs at that source.
    n=0
    for k,vals in CAT.items():
        for i,item in enumerate(vals):
            if i==g['truth'][k]:continue
            room=(n*3+2)%8;n+=1
            if k=='suspect':text=f'사건 시각의 기록을 대조하면 {display(g,k,i)}의 알리바이가 확인됩니다. 범인 후보에서 제외합니다.'
            elif k=='weapon':text=f'{item}의 보관 기록과 물증이 일치합니다. 살인에 사용된 흉기는 아닙니다.'
            else:text=f'현장 자료를 대조하면 {item}은(는) 실제 범행 장소가 아님을 확인할 수 있습니다.'
            g['all_clues'].append({'id':f'{k}:{i}','category':k,'index':i,'source_room':room,'text':text})
    add(g,'gm',f'오네이트 익스프레스에 오신 것을 환영합니다, {name} 님. 당신은 {JOBS[role]}입니다. 저녁 식사 자리에서 재력가 모건 골드는 금고에 보관된 자신의 재산을 자랑했고, 승객들의 표정은 싸늘해졌습니다.\n\n자정이 지나자 날카로운 비명이 복도를 가릅니다. 모건 골드가 숨진 채 발견되었습니다. 몸통에는 여러 자창이 있고, 시신을 다른 장소에서 옮긴 흔적이 보입니다. 차장 퀸이 말합니다. “열차 안에 범인이 있습니다. 도와주시겠습니까?”\n\n지금 당신은 라운지 칸에 서 있습니다. 누구에게 말을 걸거나, 어디부터 살펴보겠습니까?')
    return g

def display(g,k,i):
    return f"{g['name']} ({JOBS[i]})" if k=='suspect' and i==g['role'] else CAT[k][i]

def public(g):
    removed={c['id'] for c in g['clues']}
    return {'id':g['id'],'name':g['name'],'job':JOBS[g['role']],'stat':g['stat'],'goal':g['goal'],
      'location':PLACES[g['room']],'room':g['room'],'target':display(g,'suspect',g['target']) if g['target'] is not None else None,
      'turn':g['turn'],'prepared':g['prepared'],'pending':g['pending'],'ended':g['ended'],'won':g['won'],'wrong':g['wrong'],
      'messages':g['messages'],'clues':g['clues'],
      'npcs':[{'id':i,'name':display(g,'suspect',i),'location':PLACES[NPC_ROOMS[i]],'here':NPC_ROOMS[i]==g['room'],'personality':PERSONALITIES[i],'background':BACKGROUNDS[i]} for i in range(12) if i!=g['role']],
      'board':{k:[{'id':i,'name':display(g,k,i),'excluded':f'{k}:{i}' in removed} for i in range(len(v))] for k,v in CAT.items()},
      'places':[{'id':i,'name':v,'visited':i in g['visited']} for i,v in enumerate(PLACES)]}

def llm_context(g):
    # Explicit whitelist; do not serialize the entire game.
    p=public(g)
    return {k:p[k] for k in ('name','job','goal','stat','location','target','prepared','pending','clues','npcs','board')} | {'recent_dialogue':g['messages'][-12:], 'npc_memory':g['npc_memory'].get(str(g['target']),[])[-5:]}

def match(text, items):
    found=[]
    for i,v in enumerate(items):
        tokens=[v,v.split(' (')[0]]
        if '(' in v:tokens.append(v.split('(')[1].strip(')'))
        if any(t in text for t in tokens):found.append(i)
    return found

def local_intent(g,text):
    stripped=text.strip();rooms=match(text,PLACES);people=match(text,[display(g,'suspect',i) for i in range(12)])
    weapons=match(text,WEAPONS)
    if stripped in ('취소','판정 취소','그만'):return {'kind':'cancel'}
    if stripped in ('굴려','굴린다','주사위','주사위 굴리기','판정','판정 진행','네','좋아','진행해'):return {'kind':'roll'} if g['pending'] else {'kind':'talk'}
    if any(v in text for v in ('범인은','범인으로','지목','범행 장소는')):
        return {'kind':'accuse','suspect':people[0] if len(people)==1 else -1,'weapon':weapons[0] if len(weapons)==1 else -1,'place':rooms[0] if len(rooms)==1 else -1}
    if any(v in text for v in ('지도','어디에 있','누가 있','주변','둘러','현재 위치','목록')):return {'kind':'look'}
    if rooms and any(v in text for v in ('이동','가자','간다','가볼','들어','찾아가','향해')):return {'kind':'move','place':rooms[0]}
    if people and any(v in text for v in ('찾아가','찾아간','찾아갈','만나러')):return {'kind':'find','target':people[0]}
    if any(v in text for v in ('준비','정리','메모','검토')):return {'kind':'prepare'}
    approach='brutality' if any(v in text for v in ('협박','위협','강경','강제로','힘으로','부순','직감')) else 'compassion'
    if any(v in text for v in ('위험','뛰어내','훔친','훔치','부순','열차를 멈','강제로 문')):
        return {'kind':'check','target':None,'approach':approach,'purpose':'risk'}
    if any(v in text for v in ('조사','살펴','살핀','뒤져','수색','검사','심문','알리바이','자정','단서','설득','협박','증거','범행','사건 당시','기록을','확인','묻','물어')):
        physical=any(v in text for v in ('서류','시신','바닥','소지품','현장','상처','수색','방을','장부','조사'))
        return {'kind':'check','target':people[0] if people else None if physical else g['target'],'approach':approach}
    return {'kind':'talk','target':people[0] if people else g['target']}

def validate_intent(raw,g):
    if not isinstance(raw,dict):raise ValueError('해석 결과 형식 오류')
    kind=raw.get('kind')
    if kind not in ('look','move','find','talk','prepare','check','accuse','roll','cancel'):raise ValueError('지원하지 않는 행동')
    result={'kind':kind}
    for field,limit in [('target',12),('suspect',12),('weapon',8),('place',8)]:
        v=raw.get(field)
        if v is not None:
            if not isinstance(v,int) or isinstance(v,bool) or not 0<=v<limit:raise ValueError('대상 지정 오류')
            result[field]=v
    if raw.get('purpose') in ('investigate','risk'):result['purpose']=raw['purpose']
    if raw.get('approach') in ('compassion','brutality'):result['approach']=raw['approach']
    return result

def apply(g,intent,text='',rng=None):
    if g['ended']:return '사건은 끝났습니다. 기록을 저장하거나 새 여행을 시작하세요.'
    kind=intent['kind']
    if kind=='cancel':g['pending']=None;return '판정을 취소했습니다. 다른 방법을 선택해도 좋습니다.'
    if g['pending'] and kind not in ('roll',):
        return '먼저 기다리고 있는 판정을 진행하거나 “취소”라고 말해주세요. 접근을 바꾸고 싶다면 판정을 취소한 뒤 새 행동을 설명하면 됩니다.'
    if kind=='look':
        here=[display(g,'suspect',i) for i in range(12) if i!=g['role'] and NPC_ROOMS[i]==g['room']]
        return f"{PLACES[g['room']]}\n{ROOMS[g['room']]}\n이곳에 있는 인물: {', '.join(here) or '지금은 아무도 없습니다.'}\n이곳의 자료를 조사하거나, 다른 칸으로 이동할 수 있습니다."
    if kind=='move':
        room=intent.get('place')
        if room is None:return '어느 칸으로 이동할까요?'
        g['room']=room;g['target']=None;g['turn']+=1
        if room not in g['visited']:g['visited'].append(room)
        return apply(g,{'kind':'look'})
    if kind=='find':
        target=intent.get('target')
        if target is None:return '누구를 찾고 있나요?'
        if target==g['role']:return '그 역할은 당신의 캐릭터입니다. 다른 승객에게 말을 걸어보세요.'
        result=apply(g,{'kind':'move','place':NPC_ROOMS[target]});g['target']=target
        if any(w in text for w in ('묻','물어','심문','알리바이','설득','협박')):
            approach='brutality' if any(w in text for w in ('협박','위협','강경')) else 'compassion'
            return result+'\n'+apply(g,{'kind':'check','target':target,'approach':approach},text)
        return result+f'\n{display(g,"suspect",target)}에게 다가갑니다.'
    if kind=='prepare':
        g['prepared']=True;g['turn']+=1
        return '당신은 관찰 내용과 질문을 정리합니다. 다음 판정에 준비 보너스 +1d를 받습니다. 준비는 한 번만 적용됩니다.'
    if kind in ('check','talk'):
        target=intent.get('target',g['target'])
        if target==g['role']:return '그 인물은 당신입니다. 자신의 소지품을 조사하려면 대상을 빼고 행동을 적어주세요.'
        if target is not None:
            if NPC_ROOMS[target]!=g['room']:return f'{display(g,"suspect",target)}은(는) {PLACES[NPC_ROOMS[target]]}에 있습니다. 먼저 찾아가시겠습니까?'
            g['target']=target
            memory=g['npc_memory'][str(target)];memory.append({'who':'player','text':text[:600]});del memory[:-20]
        if kind=='talk':
            if target is not None:return f'{display(g,"suspect",target)}은(는) {PERSONALITIES[target]}. “모건 골드를 알았지만, 제가 그를 죽인 것은 아닙니다.” 당신의 다음 말을 기다립니다. 아직 새로운 단서는 확인되지 않았습니다.'
            return '차창에 어둠이 스쳐 지나갑니다. 무엇을 하고 싶은지 말해주세요. 이동, 승객과의 대화, 주변 조사 모두 가능합니다.'
        approach=intent.get('approach','compassion')
        job=JOBS[g['role']]
        expert=(target is not None and job in ('탐정','배우','사기꾼','기자')) or (target is None and ((job=='탐정') or (job=='의사' and any(w in text for w in ('시신','상처','의학'))) or (job=='기술자' and g['room']==7) or (job=='요리사' and g['room']==3) or (job=='사업가' and g['room']==5)))
        if any(w in text for w in ('시신','상처')) and g['room']!=0:
            return '모건 골드의 시신은 라운지 칸에 있습니다. 먼저 그곳으로 이동해서 살펴보세요.'
        if any(w in text for w in ('훔치','훔친','금고를 열')) and g['room']!=5:
            return '열차의 금고는 금고 칸에 있습니다. 먼저 그곳으로 이동해주세요.'
        if any(w in text for w in ('뛰어내','열차를 멈')):
            return '열차는 추운 황무지를 빠르게 달리고 있습니다. 지금 그 행동은 위험합니다. 차장이나 기술자를 찾아 안전한 방법부터 알아보시겠습니까?'
        g['pending']={'purpose':intent.get('purpose','investigate'),'action':text[:600],'target':target,'room':g['room'],'approach':approach,'dice':1+int(g['prepared'])+int(expert),'prepared':g['prepared'],'expert':expert}
        mode='강경함' if approach=='brutality' else '공감';condition='미만' if approach=='brutality' else '초과'
        return f"{mode} 판정을 요청합니다. {g['pending']['dice']}d6을 굴려 {g['stat']} {condition}인 주사위마다 성공 1개입니다. 같은 숫자는 실패입니다.\n준비: {'+1d' if g['prepared'] else '없음'} · 전문성: {'+1d' if expert else '없음'}\n주사위를 굴리겠습니까?"
    if kind=='roll':
        p=g['pending']
        if not p:return '지금은 기다리는 판정이 없습니다. 먼저 행동을 설명해주세요.'
        rng=rng or random.SystemRandom();dice=[rng.randint(1,6) for _ in range(p['dice'])]
        successes=sum(v<g['stat'] if p['approach']=='brutality' else v>g['stat'] for v in dice)
        known={x['id'] for x in g['clues']}
        pool=[c for c in g['all_clues'] if c['source_room']==p['room'] and c['id'] not in known] if p.get('purpose')!='risk' else []
        amount=2 if successes>=3 else int(successes>0);got=pool[:amount]
        source=display(g,'suspect',p['target']) if p['target'] is not None else PLACES[p['room']]
        for c in got:g['clues'].append({k:c[k] for k in ('id','category','index','text')}|{'source':source,'turn':g['turn']+1})
        g['turn']+=1;g['pending']=None;g['prepared']=False
        if successes==0:out='접근이 통하지 않습니다. 열차의 흔들림과 주변의 경계 때문에 조사를 중단합니다. 단서는 얻지 못했습니다. 방법을 바꾸거나 준비한 뒤 다시 시도할 수 있습니다.'
        elif p.get('purpose')=='risk':out='당신이 시도한 행동에 성공합니다. 사건의 단서는 추가되지 않았습니다.'
        elif not got:out='조사는 성공했지만 이곳에서 확인할 수 있는 단서는 이미 모두 기록했습니다. 다른 칸을 조사해보세요.'
        else:out='\n'.join(c['text'] for c in got)
        if successes==1:g['turn']+=1;out+='\n대가: 자료를 확보하고 상황을 수습하느라 시간이 더 걸렸습니다. 수사 경과 +1.'
        if successes>=3:g['prepared']=True;out+='\n추가 효과: 조사 요령을 파악했습니다. 다음 판정에 준비 +1d를 받습니다.'
        add(g,'roll',f"주사위 {dice} → 성공 {successes}개",dice=dice,successes=successes)
        for c in got:add(g,'clue',c['text'],source=source)
        return out
    if kind=='accuse':
        guess={k:intent.get(k) for k in CAT}
        if any(guess[k] is None or not 0<=guess[k]<len(CAT[k]) for k in CAT):return '최종 지목에는 범인, 흉기, 실제 범행 장소가 모두 필요합니다. “범인은 [이름], 흉기는 [이름], 장소는 [칸]”처럼 말해주세요.'
        # Only a complete accusation reveals whether the whole triple matches.
        if guess==g['truth']:
            g['ended']=True;g['won']=True
            return '세 가지 추리가 모두 맞습니다. '+', '.join(display(g,k,guess[k]) for k in CAT)+' — 이것이 사건의 진실입니다.\n승객들 앞에서 당신의 증거를 제시하자 침묵이 내려앉습니다. 도착 전 범인을 붙잡았습니다. 오네이트 익스프레스의 사건은 해결되었습니다.'
        g['wrong']+=1
        if g['wrong']>=2:
            g['ended']=True;return '두 번째 지목도 진실과 다릅니다. 수사팀은 승객들의 신뢰를 잃고, 사건은 미제로 남습니다. 새 여행에서 다시 도전할 수 있습니다.'
        return '지목한 조합은 진실과 다릅니다. 차장이 마지막 한 번의 기회를 줍니다. 다시 증거를 검토해보세요.'
    return '어떻게 행동할지 조금 더 구체적으로 알려주세요.'
