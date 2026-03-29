"""
Hard Grader — "FAANG or Bust"
Agent must get at least 1 shortlist at a big company (difficulty=2).
Score 0.0–1.0 based on: did it succeed + how efficiently?
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from job_market_env import JobMarketEnv

def grade(agent, n_episodes: int = 100, seed: int = 42) -> float:
    """
    Grade the agent on the hardest task: get shortlisted at FAANG.

    Score formula:
        - 0.0  if not shortlisted at big company at all
        - 0.5  if shortlisted at big company
        - +0.5 bonus based on efficiency (fewer apps wasted)

    Efficiency = 1 - (wasted_count / applications_used)
    Final = 0.5 + 0.5 * efficiency  (if shortlisted)
    """
    env = JobMarketEnv(seed=seed)
    scores = []

    for ep in range(n_episodes):
        state = env.reset()
        state.company_difficulty = 2  # force FAANG difficulty

        big_shortlisted = False
        apps_used = 0

        while not state.done:
            action = agent.choose_action(state.to_tuple())
            state, reward, done, info = env.step(action)

            if info.get("shortlisted") and action == 1:
                big_shortlisted = True
            if action in (0, 1):
                apps_used += 1

        if not big_shortlisted:
            scores.append(0.0)
        else:
            wasted = state.wasted_count
            efficiency = 1.0 - (wasted / max(apps_used, 1))
            score = 0.5 + 0.5 * max(efficiency, 0.0)
            scores.append(min(score, 1.0))

    return round(sum(scores) / len(scores), 4)


def grade_random(n_episodes: int = 100) -> float:
    import random

    class RandomAgent:
        def choose_action(self, state):
            return random.randint(0, 2)

    return grade(RandomAgent(), n_episodes)


if __name__ == "__main__":
    baseline = grade_random()
    print(f"[Hard Task] Random agent baseline score: {baseline:.4f}")