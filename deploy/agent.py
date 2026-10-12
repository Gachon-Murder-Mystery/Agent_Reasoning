"""LLM reasoning agent. Takes game state and decides one action

A belief_mode switch exists for the H2 comparison.
  True  -> structured belief (probabilities + entropy) goes into the prompt
  False -> only raw clue text goes in; the model judges on its own

The decision record format follows the example in the 1-page proposal.
An unheld ID in evidence_used is recorded rather than rejected, because
that is exactly what H2 measures (hallucinated evidence).
"""
import json
import re
import time

from pathlib import Path
from schema import BeliefState
from llm import GatewayError

SCHEMA_VERSION = "b-0.3"
LOG_PATH = Path("decisions.jsonl")


class LLMAgent:
    def __init__(self, agent_id, client, belief_mode=True, log_path=LOG_PATH):
        self.agent_id = agent_id 
        self.client = client
        self.belief_mode = belief_mode
        self.log_path = Path(log_path)

        self.initialized = False
        self.profile = {}
        self.belief = None
        self.known_info = []        # clues I hold
        self.applied_seen = []      # held clue ids already folded into belief
        self.disclosed = []         # clue ids I have disclosed so far
        self.candidates = {}
        self.turn = 0 
        self.inactive = False       # True once turned into NPC/corpse

    # ---------- stage 1: state sync ----------

    def sync(self, notice, observation):
        me = notice.get("self") or observation.get("self") or {}

        if not self.initialized:
            self.profile = dict(me)
            self.initialized = True
        else:
            if me.get("personality") != self.profile.get("personality"):
                self._log({"event": "profile_changed",
                           "before": self.profile.get("personality"),
                           "after": me.get("personality")})
            self.profile = dict(me)

        # Candidate pools are fixed, but may arrive empty on the first turn,
        # so re-check every turn instead of reading them once.
        cands = observation.get("candidates") or notice.get("candidates")
        if cands:
            self.candidates = cands
        if self.belief_mode and self.belief is None and self.candidates:
            self.belief = BeliefState(self.candidates)

        mode = self.profile.get("mode", "active")
        if mode != "active" and not self.inactive:
            self.inactive = True
            self._log({"event": "mode_changed", "mode": mode})
        elif mode == "active":
            self.inactive = False

        self.known_info = observation.get("information") or []
        if self.belief_mode and self.belief is not None:
            held = {self._item_id(i) for i in self.known_info}
            for item in self.known_info + (observation.get("shared_information") or []):
                key = self._item_id(item)
                if key not in self.applied_seen:
                    self.belief.apply(item, source="own" if key in held else "heard")
                    self.applied_seen.append(key)
                    self._log({"event": "clue_seen", "turn": self.turn,
                               "clue_id": key,
                               "source": "own" if key in held else "shared"})
        self.turn += 1

    # ---------- stage 2: prompts ----------

    def _system_prompt(self):
        return (
            "너는 살인 미스터리의 수사관이다. JSON 객체 하나만 출력한다.\n\n"
            "출력 형식:\n"
            "{\n"
            '  "evidence_used": [근거로 삼은 단서 ID 배열],\n'
            '  "current_hypothesis": {"suspect": ID 또는 null, "weapon": ID 또는 null, '
            '"place": ID 또는 null, "confidence": 0.0~1.0},\n'
            '  "action": {"type": 행동, "payload": {...}},\n'
            '  "consideration": "한 문장"\n'
            "}\n\n"
            "action.type은 반드시 제공된 allowed_actions 중 하나여야 한다.\n"
            'payload: move={"room": 방 번호}, stay={}, prepare={}, pass={}, '
            'explore={"approach": "brutality"|"compassion"}, '
            'question={"target_id": 인물ID, "approach": "brutality"|"compassion"}, '
            'examine_corpse={"target_id": 시체ID, "approach": "brutality"|"compassion"}, '
            'accuse={"suspect": 번호, "weapon": 번호, "place": 번호}.\n'
            "방 번호와 accuse의 세 값은 문자열이 아니라 정수다.\n"
            "question은 같은 방의 살아있는 인물에게만, "
            "examine_corpse는 같은 방의 시체에게만 쓸 수 있다. "
            "occupants의 mode로 구분하라.\n\n"
            "판정 규칙: 심문과 위험 행동은 1d6을 굴린다. "
            "prepare를 미리 해두면 주사위를 하나 더 굴린다. "
            "brutality는 주사위가 내 수치보다 낮아야, compassion은 높아야 성공이다. "
            "성공하면 사건과 무관한 항목을 하나 배제하는 단서를 얻는다.\n\n"
            "우선순위:\n"
            "1) 범인·흉기·장소를 밝히는 것이 최우선이다. "
            "행동은 증거를 근거로 고르고, 성향은 비슷한 선택지 사이에서만 참고한다.\n"
            "2) 그 다음이 개인 목표다.\n\n"
            "evidence_used에는 실제 판단 근거로 쓴 단서 ID만 적는다. "
            "단서나 인물을 지어내지 마라. "
            "accuse는 틀리면 즉사한다. 확신이 없으면 pass를 골라라."
        )


    def _user_prompt(self, notice, observation):
        state = {
            "turn": self.turn,
            "phase": observation.get("phase") or notice.get("phase"),
            "allowed_actions": notice.get("allowed_actions", []),
            "me": {
                "job": (self.profile.get("job") or {}).get("name"),
                "status": self.profile.get("stat"),
                "personality": (self.profile.get("personality") or {}).get("name"),
                "personal_goal": (self.profile.get("personal_goal") or {}).get("name"),
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
        if self.belief_mode and self.belief is not None:
            state["belief"] = self.belief.summary()
        return json.dumps(state, ensure_ascii=False)


    # ---------- stage 3: action decision ----------

    def decide(self, notice, observation):
        """Return (action, payload); falls back to pass on failure."""
        allowed = notice.get("allowed_actions", [])
        safe = next((a for a in ("pass", "stay") if a in allowed), allowed[0] if allowed else "pass")

        if self.inactive:
            self._log({"event": "skipped", "turn": self.turn, "reason": "inactive"})
            return safe, {}

        system, user = self._system_prompt(), self._user_prompt(notice, observation)
        fail_reason = None

        for attempt in (1, 2):
            started = time.time()
            try:
                raw, usage = self.client.request(system, user, budget=700)
            except GatewayError as e:
                # The model was never reached, so there is nothing to re-ask.
                # Keep the same prompt, wait briefly, and retry.
                fail_reason = "gateway"
                self._log({"event": "llm_error", "turn": self.turn,
                           "context": "action", "attempt": attempt,
                           "detail": str(e),
                           "elapsed": round(time.time() - started, 2)})
                time.sleep(1.5 * attempt)
                continue

            parsed = self._parse(raw)
            if parsed is None:
                fail_reason = "invalid_output"
                user += "\n\n출력이 JSON이 아니었다. JSON 객체 하나만 출력하라."
                continue

            ok, why = self._validate(parsed, notice)
            if not ok:
                fail_reason = "invalid_output"
                user += f"\n\n직전 출력이 거부되었다: {why}. 다시 고르라."
                continue

            action = parsed["action"]["type"]
            payload = parsed["action"].get("payload") or {}
            held = self._held_ids()

            # H2 metric: cited IDs outside what I hold or have seen shared.
            # Recorded, never rejected.
            grounded = held | self._shared_ids(notice)
            cited = parsed.get("evidence_used") or []
            ungrounded = [e for e in cited if e not in grounded]

            self._log({
                "event": "decision", "turn": self.turn,
                "phase": observation.get("phase") or notice.get("phase"),
                "action": action, "payload": payload,
                "evidence_used": cited, "ungrounded_evidence": ungrounded,
                "hypothesis": parsed.get("current_hypothesis", {}),
                "held_count": len(self.known_info),
                "disclosable_count": len(held - set(self.disclosed)),
                "attempt": attempt, "fallback": False,
                "belief_mode": self.belief_mode,
                "usage": usage, "elapsed": round(time.time() - started, 2),
            })
            # Stated reasons may be post-hoc rationalisation, so they are
            # logged apart from the action data.
            self._log({"event": "consideration", "turn": self.turn,
                       "context": "action", "text": parsed.get("consideration", "")})

            return action, payload

        # Separating the two causes matters: "gateway" turns are lost to
        # transport and must be excluded from the H2 denominator, while
        # "invalid_output" turns are genuine rule-following failures.
        self._log({"event": "decision", "turn": self.turn,
                   "phase": observation.get("phase") or notice.get("phase"),
                   "action": safe, "payload": {},
                   "fallback": True, "fallback_reason": fail_reason or "unknown",
                   "belief_mode": self.belief_mode})
        return safe, {}

    # ---------- stage 3b: answering a question (B06) ----------

    def respond_to_question(self, asker_id, topic=None):
        """Decide what to offer, or refuse, when another agent questions me.

        This is a passive handler and consumes no action. The engine settles
        what actually transfers with a die roll, so only the intent to
        disclose is produced here. Returns (offer_ids, refused).
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
            "me": {"personality": (self.profile.get("personality") or {}).get("name"),
                   "personal_goal": (self.profile.get("personal_goal") or {}).get("name"),
                   "job": (self.profile.get("job") or {}).get("name")},
            "asker": asker_id,
            "topic": topic,
            "my_information": self.known_info,
            "already_disclosed": self.disclosed,
        }, ensure_ascii=False)

        obj, usage, failed = {}, None, False
        try:
            raw, usage = self.client.request(system, user, budget=400)
            obj = self._parse_obj(raw) or {}
        except GatewayError as e:
            self._log({"event": "llm_error", "turn": self.turn,
                       "context": "question", "detail": str(e)})
        if not obj:
            failed = True

        refused = bool(obj.get("refuse"))
        offer = [] if refused else [i for i in (obj.get("offer") or []) if i in held]
        if failed:
            # Refuse on call failure. This is a transport artefact, so it is
            # tagged to keep it out of the H1 withhold-rate numerator.
            refused, offer = True, []
        self._log({"event": "question_response", "turn": self.turn,
                   "asker": asker_id, "topic": topic,
                   "offer": offer, "refused": refused,
                   "reason_code": "call_failed" if failed else "model_choice",
                   "held_count": len(held),
                   "disclosable_count": len(held - set(self.disclosed)),
                   "usage": usage})
        self._log({"event": "consideration", "turn": self.turn,
                   "context": "question", "text": obj.get("consideration", "")})
        self.disclosed.extend(i for i in offer if i not in self.disclosed)
        return offer, refused

    # ---------- parsing and validation ----------

    @staticmethod
    def _item_id(item):
        """Clue items may be dicts or bare ID strings depending on the engine."""
        return item.get("id") if isinstance(item, dict) else item

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
        return {self._item_id(i) for i in self.known_info}

    @classmethod
    def _shared_ids(cls, notice):
        return {cls._item_id(i) for i in (notice.get("shared_information") or [])}

    def _pool_ids(self, pool):
        return {self._item_id(c) for c in (self.candidates.get(pool) or [])}

    def _validate(self, p, notice):
        action = p["action"]["type"]
        payload = p["action"].get("payload") or {}
        if action not in notice.get("allowed_actions", []):
            return False, f"{action}은 이번 단계에 허용되지 않는다"
        fields = {"move": {"room"}, "stay": set(), "pass": set(), "prepare": set(),
                  "explore": {"approach"}, "question": {"target_id", "approach"},
                  "examine_corpse": {"target_id", "approach"},
                  "accuse": {"suspect", "weapon", "place"}}
        if set(payload) - fields.get(action, set()):
            return False, "payload에 허용되지 않은 항목이 있다"
        
        if action == "move":
            if payload.get("room") not in self._pool_ids("place"):
                return False, "존재하지 않는 방"
        elif action in ("question", "examine_corpse"):
            here = {self._item_id(o): o.get("mode") for o in notice.get("occupants", [])}
            here.pop(self.agent_id, None)
            target = payload.get("target_id")
            if target not in here:
                return False, "같은 방에 없는 상대"
            if (action == "examine_corpse") != (here[target] == "corpse"):
                return False, "question은 살아있는 인물에게만, examine_corpse는 시체에게만 쓸 수 있다"
            if payload.get("approach") not in ("brutality", "compassion"):
                return False, "approach는 brutality 또는 compassion이어야 한다"
        elif action == "explore":
            if payload.get("approach") not in ("brutality", "compassion"):
                return False, "approach는 brutality 또는 compassion이어야 한다"
        elif action == "accuse":
            for key in ("suspect", "weapon", "place"):
                if payload.get(key) not in self._pool_ids(key):
                    return False, f"{key} 값이 후보에 없다"

        return True, ""

    # ---------- stage 4: result intake ----------

    def ingest_result(self, result):
        gained = []
        for item in result.get("acquired_information", []):
            if item not in self.known_info:
                self.known_info.append(item)
                gained.append(self._item_id(item))
        self._log({"event": "result", "turn": self.turn,
                   "outcome": result.get("outcome"), "roll": result.get("roll"),
                   "acquired": gained})

    # ---------- logging ----------

    def _log(self, record):
        record.update({"agent_id": self.agent_id,
                       "schema_version": SCHEMA_VERSION,
                       "ts": round(time.time(), 3)})
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
