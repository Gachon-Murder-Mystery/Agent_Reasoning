# from contracts import ActionRequest (temporary)


class PlayerAdapter:

    def __init__(self, agent):
        self.agent = agent
        self._action_seq = 0

    def choose_action(self, phase_notice, observation):
        self.agent.sync(phase_notice, observation)
        action, payload = self.agent.decide(phase_notice, observation)
        return self._wrap(phase_notice, action, payload)

    def receive_result(self, result):
        self.agent.ingest_result(result)

    def respond_to_question(self, asker_id, topic):
        return self.agent.respond_to_question(asker_id, topic)

    def _wrap(self, notice, action, payload):
        self._action_seq += 1
        request = {
            "game_id": notice["game_id"],
            "agent_id": notice["agent_id"],
            "cycle_id": notice["cycle_id"],
            "phase": notice["phase"],
            "phase_epoch": notice["phase_epoch"],
            "task_id": notice["task_id"],
            "action_id": f"{notice['task_id']}:{self._action_seq}", # action_id : 중복 요청 걸러냄
                                                                    # 같은 번호, 같은 요청이면 다시 반환, 같은 번호 다른 내용이면 거절 / 통신이 끊긴 경우에 재전송 방지 
            "action": action,
            "payload": payload or {},
        }
        if ActionRequest is None:
            return request
        return ActionRequest(**request)


def build_players(agent_ids, client, belief_mode=True):
    from agent import LLMAgent

    return {
        ident: PlayerAdapter(LLMAgent(ident, client, belief_mode=belief_mode))
        for ident in agent_ids
    }
  