"""
agent/schema.py

1단계: 형식 고정. (에이전트 껍대기)
여기서 정한 세 가지 형식은 2~7단계 전부가 참조한다.

  1) BeliefState    : 에이전트의 신념 (JSON 직렬화 필수 - C의 강제 종료/인계용)
  2) Action         : 행동 형식
  3) DecisionRecord : 의사결정 로그 한 줄 (가설 H1·H2 측정의 원천 데이터)
"""

import copy
import json
import math

SCHEMA_VERSION = "b-0.1"

try:
    from fake_engine import SUSPECTS, WEAPONS, LOCATIONS
    N_SUSPECT, N_WEAPON, N_LOCATION = len(SUSPECTS), len(WEAPONS), len(LOCATIONS)
except Exception:
    N_SUSPECT, N_WEAPON, N_LOCATION = 12, 8, 8

CATEGORIES = {"suspect": N_SUSPECT, "weapon": N_WEAPON, "location": N_LOCATION}


# =============================================================== 2) Action
GAME_ACTIONS = ("MOVE", "INVESTIGATE", "INTERROGATE", "PREPARE", "ACCUSE")
COMM_ACTIONS = ("SHARE_FULL", "SHARE_PARTIAL", "WITHHOLD", "REQUEST")
META_ACTIONS = ("END_TURN",)
ALL_ACTIONS = GAME_ACTIONS + COMM_ACTIONS + META_ACTIONS

ACTION_FIELDS = {
    "MOVE": ("target",),
    "INVESTIGATE": (),
    "INTERROGATE": ("target",),
    "PREPARE": (),
    "ACCUSE": ("suspect", "weapon", "location"),
    "SHARE_FULL": ("clue_id",),
    "SHARE_PARTIAL": ("clue_id",),
    "WITHHOLD": (),
    "REQUEST": ("to", "topic"),
    "END_TURN": (),
}

# 주장성 행동: 근거를 요구한다. (H2 분모)
ASSERTIVE_ACTIONS = ("ACCUSE", "SHARE_FULL", "SHARE_PARTIAL", "WITHHOLD")


def make_action(type_, **kw):
    a = {"type": type_}
    a.update(kw)
    ok, reason = validate_action(a)
    if not ok:
        raise ValueError(f"invalid action {a}: {reason}")
    return a


def validate_action(action):
    """형식만 검사한다. 규칙상 가능한지는 엔진이 판단한다."""
    if not isinstance(action, dict):
        return False, "not_a_dict"
    t = action.get("type")
    if t not in ALL_ACTIONS:
        return False, "unknown_action_type"
    for f in ACTION_FIELDS[t]:
        if f not in action or action[f] is None:
            return False, f"missing_field:{f}"
    return True, None


def action_kind(action):
    t = action.get("type")
    if t in GAME_ACTIONS:
        return "game"
    if t in COMM_ACTIONS:
        return "comm"
    return "meta"


# ========================================================== 1) BeliefState
class BeliefState:
    """
    conf[cat][i]      : 후보 i가 정답일 확신도 (합 1.0)
    evidence[cat][i]  : 그 후보의 확신도를 움직인 단서 ID 목록
    eliminated[cat]   : 배제가 확정된 후보 인덱스 집합
    known_clues       : 내가 직접 보유한 단서 ID 집합
    heard_clues       : 동료에게서 들은 단서 ID 집합
    shared_clues      : 내가 외부로 내보낸 단서 ID 집합 (H1 추적용)
    peer_model        : 동료가 안다고 내가 추정하는 단서 ID 목록
    open_requests     : 주고받은 정보 요청
    """

    def __init__(self, agent_id):
        self.schema_version = SCHEMA_VERSION
        self.agent_id = agent_id
        self.turn = 0
        self.conf = {c: [1.0 / n] * n for c, n in CATEGORIES.items()}
        self.evidence = {c: {i: [] for i in range(n)} for c, n in CATEGORIES.items()}
        self.eliminated = {c: set() for c in CATEGORIES}
        self.known_clues = set()
        self.heard_clues = set()
        self.shared_clues = set()
        self.peer_model = {}
        self.open_requests = []
        self.notes = []

    # ------------------------------------------------------------ 갱신
    def apply_clue(self, clue, source="own"):
        """
        배제 단서 하나를 반영한다.
        clue: {"id","kind","excludes"} (SHARE_PARTIAL은 excludes가 없어 무시)
        반환: 분포가 실제로 바뀌었는지
        """
        cid = clue.get("id")
        cat = clue.get("kind")
        idx = clue.get("excludes")
        if source == "own":
            self.known_clues.add(cid)
        else:
            self.heard_clues.add(cid)
        if cat not in CATEGORIES or idx is None:
            return False
        if idx in self.eliminated[cat]:
            if cid not in self.evidence[cat][idx]:
                self.evidence[cat][idx].append(cid)
            return False
        self.eliminated[cat].add(idx)
        self.evidence[cat][idx].append(cid)
        self.conf[cat][idx] = 0.0
        self._renormalize(cat)
        return True

    def _renormalize(self, cat):
        alive = [i for i in range(CATEGORIES[cat]) if i not in self.eliminated[cat]]
        if not alive:
            return
        p = 1.0 / len(alive)
        for i in range(CATEGORIES[cat]):
            self.conf[cat][i] = 0.0 if i in self.eliminated[cat] else p

    def note_shared(self, clue_id):
        self.shared_clues.add(clue_id)

    def update_peer_model(self, peer_id, clue_ids):
        key = str(peer_id)
        cur = set(self.peer_model.get(key, []))
        cur.update(clue_ids)
        self.peer_model[key] = sorted(cur)

    # ------------------------------------------------------------ 조회
    def top(self, cat, k=3):
        pairs = [(i, self.conf[cat][i]) for i in range(CATEGORIES[cat])]
        pairs.sort(key=lambda x: -x[1])
        return pairs[:k]

    def best_guess(self):
        return {c: self.top(c, 1)[0][0] for c in CATEGORIES}

    def is_resolved(self, cat):
        return len(self.eliminated[cat]) >= CATEGORIES[cat] - 1

    def all_resolved(self):
        return all(self.is_resolved(c) for c in CATEGORIES)

    def entropy(self, cat):
        h = 0.0
        for p in self.conf[cat]:
            if p > 0:
                h -= p * math.log(p, 2)
        return h

    def total_entropy(self):
        return sum(self.entropy(c) for c in CATEGORIES)

    def evidence_for(self, cat, idx):
        return list(self.evidence[cat][idx])

    def summary(self):
        return {
            "turn": self.turn,
            "top": {c: self.top(c, 2) for c in CATEGORIES},
            "n_eliminated": {c: len(self.eliminated[c]) for c in CATEGORIES},
            "entropy": round(self.total_entropy(), 3),
            "n_known": len(self.known_clues),
            "n_heard": len(self.heard_clues),
        }

    # ------------------------------------------------- 직렬화 (C 연동용)
    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "agent_id": self.agent_id,
            "turn": self.turn,
            "conf": copy.deepcopy(self.conf),
            "evidence": {c: {str(i): list(v) for i, v in d.items()}
                         for c, d in self.evidence.items()},
            "eliminated": {c: sorted(s) for c, s in self.eliminated.items()},
            "known_clues": sorted(self.known_clues),
            "heard_clues": sorted(self.heard_clues),
            "shared_clues": sorted(self.shared_clues),
            "peer_model": copy.deepcopy(self.peer_model),
            "open_requests": copy.deepcopy(self.open_requests),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, d):
        b = cls(d["agent_id"])
        b.schema_version = d.get("schema_version", SCHEMA_VERSION)
        b.turn = d.get("turn", 0)
        b.conf = {c: list(v) for c, v in d["conf"].items()}
        b.evidence = {c: {int(i): list(v) for i, v in sub.items()}
                      for c, sub in d["evidence"].items()}
        b.eliminated = {c: set(v) for c, v in d["eliminated"].items()}
        b.known_clues = set(d.get("known_clues", []))
        b.heard_clues = set(d.get("heard_clues", []))
        b.shared_clues = set(d.get("shared_clues", []))
        b.peer_model = copy.deepcopy(d.get("peer_model", {}))
        b.open_requests = copy.deepcopy(d.get("open_requests", []))
        b.notes = list(d.get("notes", []))
        return b

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, s):
        return cls.from_dict(json.loads(s))


# ====================================================== 3) DecisionRecord
UTILITY_TERMS = ("team", "private", "info_gain", "risk", "cost")


class DecisionRecord:
    """
    선택 1건 = 1줄. JSONL로 저장한다.
    H1은 private_info_out / leak_flag로, H2는 evidence_refs / contradiction으로 센다.
    """

    def __init__(self, turn, agent_id, phase):
        self.schema_version = SCHEMA_VERSION
        self.turn = turn
        self.agent_id = agent_id
        self.phase = phase                 # "game" | "comm"
        self.belief_summary = None
        self.n_legal = 0
        self.scored = []
        self.chosen = None
        self.engine_ok = None
        self.engine_reason = None
        self.evidence_refs = []
        self.private_info_out = []
        self.leak_flag = False             # 보유하지 않은 정보를 발화 (H1 위반)
        self.contradiction = False         # 배제된 후보를 지목 (H2 위반)
        self.goal_conflict = False
        self.policy = "rule"               # "rule" | "llm" | "freeform"
        self.llm_raw = None
        self.note = ""

    def score(self, action, utility, terms):
        self.scored.append({"action": copy.deepcopy(action),
                            "u": round(float(utility), 4),
                            "terms": {k: round(float(terms.get(k, 0.0)), 4)
                                      for k in UTILITY_TERMS}})

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "turn": self.turn,
            "agent_id": self.agent_id,
            "phase": self.phase,
            "policy": self.policy,
            "belief_summary": self.belief_summary,
            "n_legal": self.n_legal,
            "scored": self.scored,
            "chosen": self.chosen,
            "engine_ok": self.engine_ok,
            "engine_reason": self.engine_reason,
            "evidence_refs": list(self.evidence_refs),
            "private_info_out": list(self.private_info_out),
            "leak_flag": self.leak_flag,
            "contradiction": self.contradiction,
            "goal_conflict": self.goal_conflict,
            "llm_raw": self.llm_raw,
            "note": self.note,
        }

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)


class DecisionLog:
    """JSONL 기록기. C가 수집할 파일 하나로 통일한다."""

    def __init__(self, path=None):
        self.path = path
        self.records = []
        if path:
            open(path, "w", encoding="utf-8").close()

    def add(self, rec):
        self.records.append(rec)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(rec.to_json() + "\n")

    # ------------------------------------------------- 가설별 집계
    def metrics(self):
        n = len(self.records) or 1
        comm = [r for r in self.records if r.phase == "comm"]

        def typ(r):
            return r.chosen.get("type") if r.chosen else None

        shares = [r for r in comm if typ(r) in ("SHARE_FULL", "SHARE_PARTIAL")]
        withholds = [r for r in comm if typ(r) == "WITHHOLD"]
        requests = [r for r in comm if typ(r) == "REQUEST"]

        # H2는 '주장할 때 근거가 있었나'를 센다.
        # MOVE/INVESTIGATE 같은 정보 수집 행동은 애초에 근거를 가질 수 없으므로 제외.
        assertive = [r for r in self.records if typ(r) in ASSERTIVE_ACTIONS]
        na = len(assertive) or 1

        return {
            "n_decisions": len(self.records),
            # H1
            "leak_count": sum(1 for r in self.records if r.leak_flag),
            "leak_rate": round(sum(1 for r in self.records if r.leak_flag) / n, 4),
            # H2
            "n_assertive": len(assertive),
            "contradiction_count": sum(1 for r in self.records if r.contradiction),
            "no_evidence_rate": round(
                sum(1 for r in assertive if not r.evidence_refs) / na, 4),
            # 정보 정책 분포 (내부 문서 9번 지표)
            "share_full": sum(1 for r in shares if typ(r) == "SHARE_FULL"),
            "share_partial": sum(1 for r in shares if typ(r) == "SHARE_PARTIAL"),
            "withhold": len(withholds),
            "request": len(requests),
            "goal_conflict_count": sum(1 for r in self.records if r.goal_conflict),
            "engine_reject": sum(1 for r in self.records if r.engine_ok is False),
        }


# ============================================================ 자체 검증
if __name__ == "__main__":
    b = BeliefState(0)
    assert abs(sum(b.conf["suspect"]) - 1.0) < 1e-9
    h0 = b.total_entropy()

    changed = b.apply_clue({"id": "suspect:3", "kind": "suspect", "excludes": 3})
    assert changed and b.conf["suspect"][3] == 0.0
    assert b.evidence_for("suspect", 3) == ["suspect:3"]
    assert not b.apply_clue({"id": "suspect:3", "kind": "suspect", "excludes": 3})
    assert b.apply_clue({"id": "weapon:1", "kind": "weapon"}, source="heard") is False
    assert "weapon:1" in b.heard_clues

    for i in range(N_SUSPECT):
        if i != 7:
            b.apply_clue({"id": f"suspect:{i}", "kind": "suspect", "excludes": i})
    assert b.is_resolved("suspect") and b.best_guess()["suspect"] == 7

    restored = BeliefState.from_json(b.to_json())
    assert restored.to_json() == b.to_json()
    assert restored.eliminated["suspect"] == b.eliminated["suspect"]

    ok, why = validate_action({"type": "ACCUSE", "suspect": 1, "weapon": 2})
    assert not ok and why == "missing_field:location"
    assert action_kind(make_action("SHARE_FULL", clue_id="suspect:3")) == "comm"

    log = DecisionLog()
    r = DecisionRecord(turn=1, agent_id=0, phase="comm")
    r.belief_summary = b.summary()
    r.score(make_action("WITHHOLD"), 0.4, {"team": -0.2, "private": 0.6})
    r.score(make_action("SHARE_FULL", clue_id="suspect:3"), 0.3,
            {"team": 0.5, "private": -0.2})
    r.chosen = make_action("WITHHOLD")
    r.evidence_refs = ["suspect:3"]
    r.goal_conflict = True
    log.add(r)
    r2 = DecisionRecord(turn=1, agent_id=0, phase="game")
    r2.chosen = make_action("INVESTIGATE")
    log.add(r2)
    m = log.metrics()
    assert m["n_assertive"] == 1 and m["no_evidence_rate"] == 0.0

    print("entropy:", round(h0, 3), "->", round(b.total_entropy(), 3))
    print("best_guess:", b.best_guess())
    print("belief json bytes:", len(b.to_json()))
    print("metrics:", m)
    print("schema self-test passed")
