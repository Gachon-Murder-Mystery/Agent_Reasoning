"""agent/schema.py

BeliefState only. Action and DecisionRecord are gone: contracts.py fixes the
action vocabulary and agent.py's _log already writes the decision rows, so a
second copy here would only drift out of step.

Clues in this game are exclusionary -- each one rules out exactly one id in
one category -- so belief stays uniform over the survivors and the useful
number is entropy. That is what the H2 prompt needs: it tells the model which
category is still worth investigating.

evidence[cat][id] keeps which clue killed which candidate. H2 compares the
evidence the model cites against what actually moved the belief, so that trail
has to exist.
"""
import copy
import json
import math

SCHEMA_VERSION = "b-0.3"
CATEGORIES = ("suspect", "weapon", "place")


class BeliefState:
    def __init__(self, candidates, agent_id=None):
        self.schema_version = SCHEMA_VERSION
        self.agent_id = agent_id
        self.pools = {c: [self._ident(i) for i in (candidates.get(c) or [])]
                      for c in CATEGORIES}
        self.names = {c: {self._ident(i): self._name(i)
                          for i in (candidates.get(c) or [])}
                      for c in CATEGORIES}
        self.eliminated = {c: set() for c in CATEGORIES}
        self.evidence = {c: {} for c in CATEGORIES}   # id -> [clue_id, ...]
        self.known_clues = set()    # clues I hold myself
        self.heard_clues = set()    # clues that reached me through sharing

    @staticmethod
    def _ident(item):
        return item.get("id") if isinstance(item, dict) else item

    @staticmethod
    def _name(item):
        return item.get("name") if isinstance(item, dict) else str(item)

    # ---------- update ----------

    def apply(self, clue, source="own"):
        """Fold in one exclusionary clue. Returns whether the belief moved."""
        if not isinstance(clue, dict):
            return False
        clue_id = clue.get("id")
        category = clue.get("category")
        excluded = clue.get("excluded_id")

        if clue_id is not None:
            (self.known_clues if source == "own" else self.heard_clues).add(clue_id)
        if category not in CATEGORIES or excluded is None:
            return False
        if excluded not in self.pools[category]:
            return False    # never trust an id the pool does not contain

        trail = self.evidence[category].setdefault(excluded, [])
        if clue_id is not None and clue_id not in trail:
            trail.append(clue_id)
        if excluded in self.eliminated[category]:
            return False    # already folded in; repeats stay harmless
        self.eliminated[category].add(excluded)
        return True

    # ---------- query ----------

    def remaining(self, category):
        return [i for i in self.pools[category] if i not in self.eliminated[category]]

    def entropy(self, category):
        n = len(self.remaining(category))
        return math.log2(n) if n else 0.0   # uniform, so entropy is log2(count)

    def total_entropy(self):
        return sum(self.entropy(c) for c in CATEGORIES)

    def settled(self, category):
        alive = self.remaining(category)
        return alive[0] if len(alive) == 1 else None

    def evidence_for(self, category, ident):
        return list(self.evidence[category].get(ident, []))

    def summary(self):
        """Goes straight into the prompt, so keep it small and readable."""
        out = {}
        for category in CATEGORIES:
            alive = self.remaining(category)
            share = round(1.0 / len(alive), 4) if alive else 0.0
            out[category] = {
                "remaining": [{"id": i, "name": self.names[category].get(i), "p": share}
                              for i in alive],
                "remaining_count": len(alive),
                "excluded_count": len(self.eliminated[category]),
                "entropy": round(self.entropy(category), 3),
                "settled": self.settled(category),
            }
        out["total_entropy"] = round(self.total_entropy(), 3)
        return out

    # ---------- serialisation (for C's takeover / forced termination) ----------

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "agent_id": self.agent_id,
            "pools": copy.deepcopy(self.pools),
            "names": {c: {str(k): v for k, v in d.items()} for c, d in self.names.items()},
            "eliminated": {c: sorted(s) for c, s in self.eliminated.items()},
            "evidence": {c: {str(k): list(v) for k, v in d.items()}
                         for c, d in self.evidence.items()},
            "known_clues": sorted(self.known_clues),
            "heard_clues": sorted(self.heard_clues),
        }

    @classmethod
    def from_dict(cls, d):
        b = cls({}, d.get("agent_id"))
        b.schema_version = d.get("schema_version", SCHEMA_VERSION)
        b.pools = {c: list(v) for c, v in d["pools"].items()}
        b.names = {c: {int(k): v for k, v in sub.items()} for c, sub in d["names"].items()}
        b.eliminated = {c: set(v) for c, v in d["eliminated"].items()}
        b.evidence = {c: {int(k): list(v) for k, v in sub.items()}
                      for c, sub in d["evidence"].items()}
        b.known_clues = set(d.get("known_clues", []))
        b.heard_clues = set(d.get("heard_clues", []))
        return b

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, s):
        return cls.from_dict(json.loads(s))


# ---------- self-test ----------

if __name__ == "__main__":
    pools = {"suspect": [{"id": i, "name": f"s{i}"} for i in range(12)],
             "weapon": [{"id": i, "name": f"w{i}"} for i in range(7)],
             "place": [{"id": i, "name": f"p{i}"} for i in range(8)]}
    b = BeliefState(pools, agent_id="player_1")
    before = b.total_entropy()

    moved = b.apply({"id": "suspect:3", "category": "suspect", "excluded_id": 3})
    assert moved and 3 not in b.remaining("suspect")
    assert b.evidence_for("suspect", 3) == ["suspect:3"]
    assert not b.apply({"id": "suspect:3", "category": "suspect", "excluded_id": 3})
    assert not b.apply({"id": "suspect:99", "category": "suspect", "excluded_id": 99})

    b.apply({"id": "weapon:2", "category": "weapon", "excluded_id": 2}, source="heard")
    assert "weapon:2" in b.heard_clues and "weapon:2" not in b.known_clues

    for i in range(12):
        if i != 7:
            b.apply({"id": f"suspect:{i}", "category": "suspect", "excluded_id": i})
    assert b.settled("suspect") == 7 and b.entropy("suspect") == 0.0

    assert BeliefState.from_json(b.to_json()).to_json() == b.to_json()
    assert json.dumps(b.summary(), ensure_ascii=False)   # must stay serialisable

    print("entropy:", round(before, 3), "->", round(b.total_entropy(), 3))
    print("settled:", {c: b.settled(c) for c in CATEGORIES})
    print("schema self-test passed")
