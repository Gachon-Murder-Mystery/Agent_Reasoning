"""
agent/fake_engine.py

A가 만들 멀티 에이전트 엔진의 대역(stand-in).
목적 1: A의 엔진이 나오기 전에 B(에이전트 추론) 코드를 먼저 짜기 위함.
목적 2: 회의에서 A에게 "이 인터페이스로 주세요"를 코드로 보여주기 위함.

공개 인터페이스
    get_view(agent_id)          -> dict   지금 이 에이전트가 볼 수 있는 것 전부
    legal_actions(agent_id)     -> list   지금 할 수 있는 행동 목록(사전 제공)
    step(agent_id, action)      -> dict   행동 실행 결과 / 거부 이유

설계 원칙
  - get_view는 정답과 타 에이전트의 단서를 절대 포함하지 않는다. (가설 H1의 전제)
  - 단서는 clues_by_agent[aid]로 분리 저장한다. (데모 board 함정 회피)
  - 단서는 선점(先占) 방식. 한 에이전트가 얻으면 사라진다. (정보 비대칭 확보)
  - 거부는 한국어 문장이 아니라 reason 코드로 돌려준다. (재시도/지표 집계용)
  - 오지목은 '본인만 탈락'. 남은 에이전트가 과제를 이어간다. (C의 3→2→1 연속성)
"""

import copy
import random

N_AGENTS = 3
N_LOCATIONS = 8
CHECK_P = 0.7
CHECK_P_PREPARED = 0.9
MAX_TURNS = 200          # 교착 감지용 안전망. 정상 판은 20~40턴에 끝난다.

# 탈락자가 쥐고 있던 미공유 단서의 처리 정책
#   "respawn" : 원래 칸으로 되돌려 다시 획득 가능 (권장 - 교착 방지)
#   "lost"    : 영구 소실 (숨기다 죽으면 팀이 못 풀 수도 있음)
#   "reveal"  : 전부 공개 (보류 유인이 사라지므로 비권장)
DROP_POLICY = "respawn"

SUSPECTS = [
    "차장 반 데르 린", "배우 비올라 베가", "의사 모로", "사업가 칼스텐",
    "백작부인 아르노", "기관사 두메니", "요리사 피셰", "가정교사 랑",
    "군인 소콜로프", "기자 베넷", "짐꾼 유리", "승무원 미나",
]
WEAPONS = [
    "은촛대", "넥타이핀", "모르핀", "쇠지렛대",
    "피아노선", "얼음송곳", "권총", "독주(毒酒)",
]
LOCATIONS = [
    "0호차 기관실", "1호차 침대칸", "2호차 식당칸", "3호차 살롱",
    "4호차 수하물칸", "5호차 전망칸", "6호차 승무원실", "7호차 후미 전망대",
]
NPCS = [(i, i % N_LOCATIONS) for i in range(11)]

PERSONAL_GOALS = {
    0: "자신이 획득한 '흉기' 단서는 최소 2턴 보유한 뒤 공유한다.",
    1: "타 에이전트보다 먼저 ACCUSE를 제출한다.",
    2: "'차장 반 데르 린'을 용의에서 제외시키는 단서는 공유하지 않는다.",
}


class FakeEngine:
    # ---------------------------------------------------------------- 초기화
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.turn = 0
        self.cur = 0
        self.over = False
        self.result = None

        self._answer = {
            "suspect": self.rng.randrange(len(SUSPECTS)),
            "weapon": self.rng.randrange(len(WEAPONS)),
            "location": self.rng.randrange(len(LOCATIONS)),
        }

        self.clues = self._make_clues()
        self.pos = {a: 0 for a in range(N_AGENTS)}
        self.prepared = {a: False for a in range(N_AGENTS)}
        self.clues_by_agent = {a: [] for a in range(N_AGENTS)}
        self.inbox = {a: [] for a in range(N_AGENTS)}
        self.public_log = []
        self.events = []
        self.budget = {"game": 1, "comm": 1}

        # 생존 상태. 오지목하면 본인만 탈락한다.
        self.alive = {a: True for a in range(N_AGENTS)}
        self.eliminated_agents = []       # [(agent_id, turn, guess)]
        self.winner = None

    def _make_clues(self):
        """
        25개 배제 단서. 용의자 11 + 흉기 7 + 장소 7 = 25 (완전 소거 가능).
        via는 '그 칸에 실제로 있는 NPC' 중에서만 고른다.
        (칸과 NPC가 어긋나면 영원히 얻을 수 없는 단서가 생긴다.)
        """
        clues, cid = {}, 0
        specs = [("suspect", SUSPECTS), ("weapon", WEAPONS), ("location", LOCATIONS)]
        for kind, names in specs:
            wrong = [i for i in range(len(names)) if i != self._answer[kind]]
            for idx in wrong:
                key = f"{kind}:{idx}"
                loc = cid % N_LOCATIONS
                here = [n for n, l in NPCS if l == loc]
                clues[key] = {
                    "id": key,
                    "kind": kind,
                    "excludes": idx,
                    "text": f"{names[idx]}은(는) 이 사건과 무관함이 확인되었다.",
                    "loc": loc,
                    "via": None if (cid % 2 == 0 or not here) else here[cid % len(here)],
                }
                cid += 1
        return clues

    # ---------------------------------------------------------------- 내부 헬퍼
    def _npcs_here(self, aid):
        return [n for n, loc in NPCS if loc == self.pos[aid]]

    def _claimed(self):
        taken = set()
        for lst in self.clues_by_agent.values():
            taken.update(lst)
        return taken

    def _available(self, aid, via):
        """선점 방식: 누군가 이미 얻은 단서는 더 이상 나오지 않는다."""
        taken = self._claimed()
        return [c for c in self.clues.values()
                if c["loc"] == self.pos[aid] and c["via"] == via and c["id"] not in taken]

    def _roll(self, aid):
        p = CHECK_P_PREPARED if self.prepared[aid] else CHECK_P
        self.prepared[aid] = False
        return self.rng.random() < p

    def _advance(self):
        """다음 생존 에이전트에게 차례를 넘긴다. 전원 탈락이면 종료."""
        self.budget = {"game": 1, "comm": 1}
        if not any(self.alive.values()):
            self.over, self.result = True, "all_eliminated"
            return
        for _ in range(N_AGENTS):
            self.cur = (self.cur + 1) % N_AGENTS
            if self.cur == 0:
                self.turn += 1
                if self.turn >= MAX_TURNS:
                    self.over, self.result = True, "timeout"
                    return
            if self.alive[self.cur]:
                return
        # 여기 도달하면 생존자가 없다는 뜻
        self.over, self.result = True, "all_eliminated"

    def _release_clues(self, aid):
        """
        탈락자가 쥐고 있던 단서 처리. DROP_POLICY로 결정한다.
        공유한 단서는 이미 public_log에 있으므로 영향받지 않는다.
        """
        shared = {e.get("clue_id") for e in self.public_log
                  if e.get("from") == aid and e.get("mode") == "SHARE_FULL"}
        held = list(self.clues_by_agent[aid])
        unshared = [c for c in held if c not in shared]

        if DROP_POLICY == "respawn":
            # 미공유분은 원래 칸으로 돌아간다 (소유 해제 = 재획득 가능)
            self.clues_by_agent[aid] = [c for c in held if c in shared]
        elif DROP_POLICY == "reveal":
            for cid in unshared:
                c = self.clues[cid]
                self.public_log.append({
                    "from": aid, "turn": self.turn, "mode": "SHARE_FULL",
                    "clue_id": cid, "text": c["text"], "kind": c["kind"],
                    "excludes": c["excludes"], "by": "elimination_drop",
                })
        # "lost"는 아무것도 하지 않는다 (소유한 채로 사라짐)

        self.events.append({"turn": self.turn, "agent": aid,
                            "action": {"type": "CLUE_DROP"}, "ok": True,
                            "reason": None,
                            "payload": {"policy": DROP_POLICY,
                                        "unshared": unshared}})

    def _log(self, aid, action, ok, reason=None, payload=None):
        self.events.append({"turn": self.turn, "agent": aid,
                            "action": copy.deepcopy(action),
                            "ok": ok, "reason": reason, "payload": payload})

    # ---------------------------------------------------------------- get_view
    def get_view(self, aid):
        return copy.deepcopy({
            "agent_id": aid,
            "turn": self.turn,
            "my_turn": (aid == self.cur and self.alive[aid] and not self.over),
            "alive": self.alive,                                  # 공개 정보
            "am_i_alive": self.alive[aid],
            "eliminated_agents": [e[0] for e in self.eliminated_agents],
            "budget": self.budget if aid == self.cur else {"game": 0, "comm": 0},
            "location": self.pos[aid],
            "location_name": LOCATIONS[self.pos[aid]],
            "npcs_here": self._npcs_here(aid),
            "prepared": self.prepared[aid],
            "my_clues": [self.clues[c] for c in self.clues_by_agent[aid]],
            "personal_goal": PERSONAL_GOALS[aid],
            "public_log": self.public_log,
            "inbox": self.inbox[aid],
            "catalog": {"suspects": SUSPECTS, "weapons": WEAPONS, "locations": LOCATIONS},
            "game_over": self.over,
        })
        # 주의: 정답, 타 에이전트 단서, 전체 clues, 소거판(board)은 넣지 않는다.

    # ----------------------------------------------------------- legal_actions
    def legal_actions(self, aid):
        if self.over or aid != self.cur or not self.alive[aid]:
            return []
        acts = []
        if self.budget["game"] > 0:
            here = self.pos[aid]
            for nxt in (here - 1, here + 1):
                if 0 <= nxt < N_LOCATIONS:
                    acts.append({"type": "MOVE", "target": nxt})
            if self._available(aid, None):
                acts.append({"type": "INVESTIGATE"})
            for npc in self._npcs_here(aid):
                if self._available(aid, npc):
                    acts.append({"type": "INTERROGATE", "target": npc})
            if not self.prepared[aid]:
                acts.append({"type": "PREPARE"})
            acts.append({"type": "ACCUSE", "suspect": None, "weapon": None,
                         "location": None})
        if self.budget["comm"] > 0:
            for c in self.clues_by_agent[aid]:
                acts.append({"type": "SHARE_FULL", "clue_id": c})
                acts.append({"type": "SHARE_PARTIAL", "clue_id": c})
            acts.append({"type": "WITHHOLD"})
            for other in range(N_AGENTS):
                if other != aid and self.alive[other]:      # 죽은 상대에겐 못 보냄
                    acts.append({"type": "REQUEST", "to": other, "topic": "weapon"})
        acts.append({"type": "END_TURN"})
        return copy.deepcopy(acts)

    # ---------------------------------------------------------------- step
    def step(self, aid, action):
        t = action.get("type")
        if self.over:
            return self._fail(aid, action, "game_over")
        if not self.alive[aid]:
            return self._fail(aid, action, "agent_eliminated")
        if aid != self.cur:
            return self._fail(aid, action, "not_your_turn")

        game_acts = {"MOVE", "INVESTIGATE", "INTERROGATE", "PREPARE", "ACCUSE"}
        comm_acts = {"SHARE_FULL", "SHARE_PARTIAL", "WITHHOLD", "REQUEST"}
        if t in game_acts and self.budget["game"] <= 0:
            return self._fail(aid, action, "no_game_action_left")
        if t in comm_acts and self.budget["comm"] <= 0:
            return self._fail(aid, action, "no_comm_action_left")

        if t == "END_TURN":
            self._log(aid, action, True)
            self._advance()
            return {"ok": True, "observation": {"type": "turn_ended"}}

        if t == "MOVE":
            tgt = action.get("target")
            if tgt is None or not (0 <= tgt < N_LOCATIONS) or abs(tgt - self.pos[aid]) != 1:
                return self._fail(aid, action, "target_not_adjacent")
            self.pos[aid] = tgt
            return self._ok(aid, action, {"type": "moved", "location": tgt})

        if t in ("INVESTIGATE", "INTERROGATE"):
            via = action.get("target") if t == "INTERROGATE" else None
            if t == "INTERROGATE" and via not in self._npcs_here(aid):
                return self._fail(aid, action, "target_not_here")
            pool = self._available(aid, via)
            if not pool:
                return self._fail(aid, action, "no_clue_here")
            if not self._roll(aid):
                return self._ok(aid, action, {"type": "check_failed"})
            got = pool[0]
            self.clues_by_agent[aid].append(got["id"])
            return self._ok(aid, action, {"type": "clue_gained", "clue": got})

        if t == "PREPARE":
            self.prepared[aid] = True
            return self._ok(aid, action, {"type": "prepared"})

        if t == "ACCUSE":
            guess = {k: action.get(k) for k in ("suspect", "weapon", "location")}
            if any(v is None for v in guess.values()):
                return self._fail(aid, action, "incomplete_accusation")

            if guess == self._answer:
                # 정답: 게임 종료
                self.over = True
                self.result = "win"
                self.winner = aid
                obs = {"type": "verdict", "correct": True, "accuser": aid,
                       "eliminated": False}
                self._log(aid, action, True, payload=obs)
                return {"ok": True, "observation": copy.deepcopy(obs)}

            # 오지목: 본인만 탈락. 남은 에이전트는 계속 진행한다.
            self.alive[aid] = False
            self.eliminated_agents.append((aid, self.turn, guess))
            self.public_log.append({"from": aid, "turn": self.turn,
                                    "mode": "ELIMINATED", "guess": guess})
            obs = {"type": "verdict", "correct": False, "accuser": aid,
                   "eliminated": True}
            self._log(aid, action, True, payload=obs)
            self._release_clues(aid)
            self._advance()
            return {"ok": True, "observation": copy.deepcopy(obs)}

        if t in ("SHARE_FULL", "SHARE_PARTIAL"):
            cid = action.get("clue_id")
            if cid not in self.clues_by_agent[aid]:
                return self._fail(aid, action, "clue_not_owned")
            c = self.clues[cid]
            entry = {"from": aid, "turn": self.turn, "mode": t, "clue_id": cid}
            if t == "SHARE_FULL":
                entry["text"] = c["text"]
                entry["kind"] = c["kind"]
                entry["excludes"] = c["excludes"]
            else:
                entry["kind"] = c["kind"]
            self.public_log.append(entry)
            return self._ok(aid, action, {"type": "shared", "mode": t, "clue_id": cid})

        if t == "WITHHOLD":
            return self._ok(aid, action, {"type": "withheld"})

        if t == "REQUEST":
            to = action.get("to")
            if to not in range(N_AGENTS) or to == aid:
                return self._fail(aid, action, "bad_recipient")
            if not self.alive[to]:
                return self._fail(aid, action, "recipient_eliminated")
            self.inbox[to].append({"from": aid, "turn": self.turn,
                                   "topic": action.get("topic")})
            return self._ok(aid, action, {"type": "requested", "to": to,
                                          "topic": action.get("topic")})

        return self._fail(aid, action, "unknown_action")

    # ---------------------------------------------------------------- 응답 생성
    def _ok(self, aid, action, obs):
        t = action.get("type")
        if t in {"MOVE", "INVESTIGATE", "INTERROGATE", "PREPARE", "ACCUSE"}:
            self.budget["game"] -= 1
        else:
            self.budget["comm"] -= 1
        self._log(aid, action, True, payload=obs)
        if self.budget["game"] <= 0 and self.budget["comm"] <= 0 and not self.over:
            self._advance()
        return {"ok": True, "observation": copy.deepcopy(obs)}

    def _fail(self, aid, action, reason):
        self._log(aid, action, False, reason=reason)
        return {"ok": False, "reason": reason}


# -------------------------------------------------------------- 모듈 레벨 API
_engine = FakeEngine(seed=0)


def reset(seed=0):
    global _engine
    _engine = FakeEngine(seed=seed)
    return _engine


def get_view(agent_id):
    return _engine.get_view(agent_id)


def legal_actions(agent_id):
    return _engine.legal_actions(agent_id)


def step(agent_id, action):
    return _engine.step(agent_id, action)


def current_agent():
    return _engine.cur


def is_over():
    return _engine.over


def alive():
    return dict(_engine.alive)


# -------------------------------------------------------------- 스모크 테스트
if __name__ == "__main__":
    reset(seed=42)
    rng = random.Random(1)
    guard = 0
    while not is_over() and guard < 5000:
        guard += 1
        aid = current_agent()
        acts = legal_actions(aid)
        if not acts:
            break
        # 가끔 무작위 지목을 섞어 탈락 경로를 시험한다
        if rng.random() < 0.01:
            pick = {"type": "ACCUSE",
                    "suspect": rng.randrange(len(SUSPECTS)),
                    "weapon": rng.randrange(len(WEAPONS)),
                    "location": rng.randrange(len(LOCATIONS))}
        else:
            pick = rng.choice([a for a in acts if a["type"] != "ACCUSE"] or acts)
        step(aid, pick)

    print("turns:", _engine.turn, "result:", _engine.result,
          "winner:", _engine.winner)
    print("alive:", _engine.alive)
    print("eliminated:", _engine.eliminated_agents)
    for a in range(N_AGENTS):
        v = get_view(a)
        print(f"agent {a}: alive={v['am_i_alive']} loc={v['location']} "
              f"clues={[c['id'] for c in v['my_clues']]}")
    sets = [set(c["id"] for c in get_view(a)["my_clues"]) for a in range(N_AGENTS)]
    print("단서 집합이 서로 다름:", len({frozenset(s) for s in sets}) > 1)
    print("중복 소유 없음:", len(sets[0] | sets[1] | sets[2]) ==
          len(sets[0]) + len(sets[1]) + len(sets[2]))
    print("정답 누출 여부:", "_answer" in get_view(0))
