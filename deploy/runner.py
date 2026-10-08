"""
agent/runner.py

2단계: 에이전트 n개를 엔진에 묶어 한 판을 끝까지 돌린다.
전역 상태 공유 금지 원칙을 지키려고 에이전트는 서로를 직접 참조하지 않는다.
정보는 반드시 엔진의 public_log / inbox를 경유한다.

오지목 탈락 규칙 반영: 탈락자는 건너뛰고 남은 에이전트가 과제를 이어간다.
"""

import argparse

import fake_engine as eng
from ornate_gm_demo.agent.agent import RuleAgent
from schema import BeliefState, DecisionLog, make_action

N_AGENTS = 3


def run(seed=0, log_path=None, max_steps=8000, verbose=True):
    eng.reset(seed=seed)
    agents = [RuleAgent(i, seed=seed) for i in range(N_AGENTS)]
    log = DecisionLog(log_path)
    steps = 0

    while not eng.is_over() and steps < max_steps:
        steps += 1
        aid = eng.current_agent()
        view = eng.get_view(aid)

        if not view["am_i_alive"]:          # 안전장치. 엔진이 이미 건너뛴다.
            break

        ag = agents[aid]
        ag.observe(view)                      # Perception
        legal = eng.legal_actions(aid)
        if not legal:
            break

        if view["budget"]["game"] > 0:
            action, rec = ag.decide_game(view, legal)
        elif view["budget"]["comm"] > 0:
            action, rec = ag.decide_comm(view, legal)
        else:
            action, rec = ag.decide_game(view, [make_action("END_TURN")])

        res = eng.step(aid, action)           # 실행은 엔진(A)의 권한
        rec.engine_ok = res.get("ok")
        rec.engine_reason = res.get("reason")
        ag.ingest_result(action, res)
        log.add(rec)

        if verbose and action["type"] != "END_TURN":
            obs = res.get("observation", {}) or {}
            tag = obs.get("type", res.get("reason", ""))
            if obs.get("type") == "verdict":
                tag = "WIN" if obs["correct"] else "WRONG → 탈락"
            print(f"  t{view['turn']:>3} a{aid} {action['type']:<14} -> {tag}")

    e = eng._engine
    print("\n=== 종료 ===")
    print("turns:", e.turn, "result:", e.result, "winner:", e.winner,
          "steps:", steps)
    print("alive:", e.alive)
    for aid, turn, guess in e.eliminated_agents:
        print(f"  탈락: agent {aid} (t{turn}) 오답 {guess}")
    for a in agents:
        g = a.belief.best_guess()
        print(f"agent {a.id} [{a.goal_kind}] alive={e.alive[a.id]} guess={g} "
              f"entropy={a.belief.total_entropy():.2f} "
              f"own={len(a.belief.known_clues)} heard={len(a.belief.heard_clues)} "
              f"shared={len(a.belief.shared_clues)}")
    print("answer:", e._answer)               # 디버깅 전용. 에이전트는 못 본다.
    m = log.metrics()
    print("metrics:", m)

    # 2단계 완료 기준
    checks = {
        "예외 없이 종료": True,
        "belief 서로 다름": len({frozenset(a.belief.known_clues) for a in agents}) > 1,
        "누설 0건": m["leak_count"] == 0,
        "모순 0건": m["contradiction_count"] == 0,
        "직렬화 왕복": all(
            BeliefState.from_json(a.belief.to_json()).to_json() == a.belief.to_json()
            for a in agents),
        "탈락자 행동 없음": all(
            r.engine_reason != "agent_eliminated" for r in log.records),
    }
    print("\n--- 체크 ---")
    for k, v in checks.items():
        print(f"  {'OK ' if v else 'FAIL'} {k}")
    return agents, log


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log", default="decisions.jsonl")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()
    run(seed=a.seed, log_path=a.log, verbose=not a.quiet)
