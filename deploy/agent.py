"""LLM 추론 에이전트. 게임 상태를 받아 행동 하나와 정보 정책을 결정한다.

H2 비교를 위해 belief_mode 스위치를 둔다.
  True  -> 구조화된 신념(확률 + 엔트로피)을 프롬프트에 넣는다
  False -> 단서 원문만 넣고 모델이 알아서 판단하게 한다

decision record 형식은 1-page proposal의 예시를 따른다.
evidence_used에 보유하지 않은 ID가 들어와도 거부하지 않고 기록만 한다.
그 자체가 H2의 측정 대상(hallucinated evidence)이기 때문이다.
"""
import json
import re
import time

from pathlib import Path
from schema import BeliefState
from ..llm import GatewayError

SCHEMA_VERSION = "b-0.3"
LOG_PATH = Path("decisions.jsonl")

class LLMAgent:
    def __init__(self, agent_id, client, belief_mode=True, log_path=LOG_PATH):
        self.agent_id = agent_id
        self.client = client
        self.belief_mode = belief_mode
        self.log_path = log_path

        self.initialized = False
        self.profile = {}
        self.belief = None
        self.known_info = []        # 내가 가진 단서
        self.shared_seen = []       # belief에 이미 반영한 공유 단서 id
        self.disclosed = []         # 내가 지금까지 공개한 단서 id
        self.candidates = {}
        self.turn = 0
        self.inactive = False       # NPC/시체 전환 시 True

    # ---------- 1단계: 상태 반영 ----------

    def sync(self, notice, observation):
        me = notice.get("self") or observation.get("self") or {}

        if not self.initialized:
            self.profile = dict(me)
            self.candidates = observation.get("candidates", {})
            if self.belief_mode:
                self.belief = BeliefState(self.candidates)
            self.initialized = True
        else:
            if me.get("personality") != self.profile.get("personality"):
                self._log({"event": "profile_changed",
                           "before": self.profile.get("personality"),
                           "after": me.get("personality")})
            self.profile = dict(me)

        mode = self.profile.get("mode", "active")
        if mode != "active" and not self.inactive:
            self.inactive = True
            self._log({"event": "mode_changed", "mode": mode})
        elif mode == "active":
            self.inactive = False

        self.known_info = observation.get("information", [])
        if self.belief_mode:
            for item in self.known_info:
                self.belief.apply(item)
            for item in observation.get("shared_information", []):
                key = item.get("id")
                if key not in self.shared_seen:
                    self.belief.apply(item)
                    self.shared_seen.append(key)
        self.turn += 1

    # ---------- 2단계: 프롬프트 ----------

    def _persona_block(self):
        return (
            f"직업: {self.profile.get('job')}\n"
            f"수치: {self.profile.get('stat')} "
            "(심문 시 brutality는 이 수치보다 낮게, compassion은 높게 나와야 성공)\n"
            f"성향: {self.profile.get('personality')}\n"
            f"개인 목표: {self.profile.get('personal_goal')}"
        )

    def _system_prompt(self):
        return (
            "너는 살인 미스터리의 수사관이다. JSON 객체 하나만 출력한다.\n\n"
            "출력 형식:\n"
            "{\n"
            '  "evidence_used": [근거로 삼은 단서 ID 배열],\n'
            '  "current_hypothesis": {"suspect": ID 또는 null, "weapon": ID 또는 null, '
            '"place": ID 또는 null, "confidence": 0.0~1.0},\n'
            '  "action": {"type": 행동, "payload": {...}},\n'
            '  "information_policy": {"type": "SHARE_FULL"|"SHARE_PARTIAL"|"WITHHOLD"|'
            '"NOTHING_TO_SHARE", "disclose": [공개할 단서 ID], "consideration": "한 문장"}\n'
            "}\n\n"
            "action.type은 반드시 제공된 allowed_actions 중 하나여야 한다.\n"
            'payload: move={"room": ID}, '
            'question={"target": 인물ID, "approach": "brutality"|"compassion"}, '
            'accuse={"suspect": ID, "weapon": ID, "place": ID}, 그 외는 {}.\n\n'
            "판정 규칙: 심문과 위험 행동은 1d6을 굴린다. "
            "prepare를 미리 해두면 주사위를 하나 더 굴린다. "
            "성공하면 사건과 무관한 항목을 하나 배제하는 단서를 얻는다.\n\n"
            "우선순위:\n"
            "1) 범인·흉기·장소를 밝히는 것이 최우선이다. "
            "행동은 증거를 근거로 고르고, 성향은 비슷한 선택지 사이에서만 참고한다.\n"
            "2) 그 다음이 개인 목표다.\n"
            "3) 단, 보유 단서를 공개할지 숨길지는 전적으로 성향과 개인 목표가 결정한다.\n\n"
            "evidence_used에는 실제 판단 근거로 쓴 단서 ID만 적는다. "
            "단서나 인물을 지어내지 마라. "
            "accuse는 틀리면 즉사한다. 확신이 없으면 pass를 골라라."
        )

    def _user_prompt(self, notice, observation):
        state = {
            "turn": self.turn,
            "phase": observation.get("phase"),
            "allowed_actions": notice.get("allowed_actions", []),
            "me": {
                "job": self.profile.get("job"),
                "status": self.profile.get("stat"),
                "personality": self.profile.get("personality"),
                "personal_goal": self.profile.get("personal_goal"),
                "room": self.profile.get("room"),
                "prepared": self.profile.get("prepared"),
            },
            "occupants": notice.get("occupants", []),
            "my_information": self.known_info,
            "already_disclosed": self.disclosed,
            "shared_information": notice.get("shared_information", []),
            "candidates": self.candidates,
        }
        if notice.get("positions") is not None:
            state["positions"] = notice["positions"]
        if self.belief_mode:
            state["belief"] = self.belief.summary()
        return json.dumps(state, ensure_ascii=False)

    # ---------- 3단계: 행동 결정 ----------

    def decide(self, notice, observation):
        """(action, payload, disclose_ids) 반환. 실패하면 pass로 떨어진다."""
        allowed = notice.get("allowed_actions", [])
        safe = "pass" if "pass" in allowed else (allowed[0] if allowed else "pass")

        if self.inactive:
            self._log({"event": "skipped", "turn": self.turn, "reason": "inactive"})
            return safe, {}, []

        system, user = self._system_prompt(), self._user_prompt(notice, observation)

        for attempt in (1, 2):
            started = time.time()
            try:
                raw, usage = self.client.request(system, user, budget=700)
            except GatewayError as e:
                self._log({"event": "llm_error", "turn": self.turn,
                           "attempt": attempt, "detail": str(e)})
                continue

            parsed = self._parse(raw)
            if parsed is None:
                user += "\n\n출력이 JSON이 아니었다. JSON 객체 하나만 출력하라."
                continue

            ok, why = self._validate(parsed, notice)
            if not ok:
                user += f"\n\n직전 출력이 거부되었다: {why}. 다시 고르라."
                continue

            action = parsed["action"]["type"]
            payload = parsed["action"].get("payload") or {}
            policy = parsed.get("information_policy") or {}
            held = self._held_ids()
            disclose = [i for i in (policy.get("disclose") or []) if i in held]

            # H2 측정: 보유/공유 범위 밖의 ID를 근거로 들었는지 (거부하지 않고 기록)
            grounded = held | self._shared_ids(notice)
            cited = parsed.get("evidence_used") or []
            ungrounded = [e for e in cited if e not in grounded]

            self._log({
                "event": "decision", "turn": self.turn,
                "phase": observation.get("phase"),
                "action": action, "payload": payload,
                "evidence_used": cited, "ungrounded_evidence": ungrounded,
                "hypothesis": parsed.get("current_hypothesis", {}),
                "policy": policy.get("type"), "disclose": disclose,
                "held_count": len(self.known_info),
                "disclosable_count": len(held - set(self.disclosed)),
                "attempt": attempt, "fallback": False,
                "belief_mode": self.belief_mode,
                "usage": usage, "elapsed": round(time.time() - started, 2),
            })
            # 사유는 사후 정당화일 수 있어 행동 데이터와 분리해 남긴다
            self._log({"event": "consideration", "turn": self.turn,
                       "context": "action", "text": policy.get("consideration", "")})

            self.disclosed.extend(d for d in disclose if d not in self.disclosed)
            return action, payload, disclose

        self._log({"event": "decision", "turn": self.turn,
                   "phase": observation.get("phase"), "action": safe,
                   "payload": {}, "disclose": [], "fallback": True,
                   "belief_mode": self.belief_mode})
        return safe, {}, []

    # ---------- 3-2단계: 질문에 답하기 (B06) ----------

    def respond_to_question(self, asker_id, topic=None):
        """남이 나를 지목해 question을 걸었을 때 공개 후보 또는 거절을 정한다.

        행동을 소비하지 않는 수신 처리다. 최종 전달 정보는 엔진이 주사위로 확정하므로
        여기서는 '공개 의사'만 낸다. 반환은 (offer_ids, refused).
        """
        held = self._held_ids()
        if self.inactive or not held:
            self._log({"event": "question_response", "turn": self.turn,
                       "asker": asker_id, "offer": [], "refused": False,
                       "reason_code": "nothing_to_offer"})
            return [], False

        system = (
            "너는 살인 미스터리의 수사관이다. 다른 수사관이 너에게 정보를 물었다.\n"
            "JSON 객체 하나만 출력한다.\n"
            '{"refuse": true|false, "offer": [공개할 단서 ID 배열], "consideration": "한 문장"}\n\n'
            "거절하면 상대는 아무것도 얻지 못하지만, 너를 의심하게 될 수 있다.\n"
            "공개하면 공동 목표에 가까워지지만 네 개인 목표가 불리해질 수 있다.\n"
            "무엇을 내줄지, 거절할지는 전적으로 네 성향과 개인 목표가 결정한다.\n"
            "보유하지 않은 단서 ID를 적지 마라."
        )
        user = json.dumps({
            "me": {"personality": self.profile.get("personality"),
                   "personal_goal": self.profile.get("personal_goal"),
                   "job": self.profile.get("job")},
            "asker": asker_id,
            "topic": topic,
            "my_information": self.known_info,
            "already_disclosed": self.disclosed,
        }, ensure_ascii=False)

        try:
            raw, usage = self.client.request(system, user, budget=400)
            obj = self._parse_obj(raw) or {}
        except GatewayError as e:
            self._log({"event": "llm_error", "turn": self.turn,
                       "context": "question", "detail": str(e)})
            obj, usage = {}, None

        refused = bool(obj.get("refuse"))
        offer = [] if refused else [i for i in (obj.get("offer") or []) if i in held]
        if not obj:                      # 호출 실패 시 보수적으로 거절
            refused, offer = True, []

        self._log({"event": "question_response", "turn": self.turn,
                   "asker": asker_id, "topic": topic,
                   "offer": offer, "refused": refused,
                   "held_count": len(held),
                   "disclosable_count": len(held - set(self.disclosed)),
                   "usage": usage})
        self._log({"event": "consideration", "turn": self.turn,
                   "context": "question", "text": obj.get("consideration", "")})
        return offer, refused

    # ---------- 파싱과 검증 ----------

    @staticmethod
    def _parse_obj(raw):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        try:
            obj = json.loads(text)
        except (ValueError, TypeError):
            return None
        return obj if isinstance(obj, dict) else None

    @classmethod
    def _parse(cls, raw):
        obj = cls._parse_obj(raw)
        if obj is None:
            return None
        act = obj.get("action")
        return obj if isinstance(act, dict) and act.get("type") else None

    def _held_ids(self):
        return {i.get("id") for i in self.known_info}

    @staticmethod
    def _shared_ids(notice):
        return {i.get("id") for i in notice.get("shared_information", [])}

    def _validate(self, p, notice):
        action = p["action"]["type"]
        payload = p["action"].get("payload") or {}
        if action not in notice.get("allowed_actions", []):
            return False, f"{action}은 이번 단계에 허용되지 않는다"

        if action == "move":
            rooms = {r.get("id") for r in self.candidates.get("places", [])}
            if payload.get("room") not in rooms:
                return False, "존재하지 않는 방"
        elif action == "question":
            here = {o.get("id") for o in notice.get("occupants", [])}
            here.discard(self.agent_id)
            if payload.get("target") not in here:
                return False, "같은 방에 없는 상대"
            if payload.get("approach") not in ("brutality", "compassion"):
                return False, "approach는 brutality 또는 compassion이어야 한다"
        elif action == "accuse":
            for key, pool in (("suspect", "suspects"), ("weapon", "weapons"),
                              ("place", "places")):
                ids = {c.get("id") for c in self.candidates.get(pool, [])}
                if payload.get(key) not in ids:
                    return False, f"{key} 값이 후보에 없다"

        policy = p.get("information_policy") or {}
        for sid in (policy.get("disclose") or []):
            if sid not in self._held_ids():
                return False, "보유하지 않은 단서를 공개하려 했다"
        return True, ""

    # ---------- 4단계: 결과 수신 ----------

    def ingest_result(self, result):
        gained = []
        for item in result.get("acquired_information", []):
            if item not in self.known_info:
                self.known_info.append(item)
                gained.append(item.get("id"))
                if self.belief_mode:
                    self.belief.apply(item)
        self._log({"event": "result", "turn": self.turn,
                   "outcome": result.get("outcome"), "roll": result.get("roll"),
                   "acquired": gained})

    # ---------- 기록 ----------

    def _log(self, record):
        record.update({"agent_id": self.agent_id,
                       "schema_version": SCHEMA_VERSION,
                       "ts": round(time.time(), 3)})
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
