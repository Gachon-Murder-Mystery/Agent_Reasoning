










class PlayerAdapter:
    def __init__(self, agent):
        self.agent = agent

    def choose_action(self, notice, observation):
        self.agent.observe(notice, observation)
        action, payload = self.agent.decide_game()
        return ActionRequest(
            game_id=notice["game_id"], agent_id=notice["agent_id"],
            cycle_id=observation["cycle_id"], phase=observation["phase"],
            phase_epoch=observation["phase_epoch"], task_id=notice["task_id"],
            action=action, payload=payload,
        )
