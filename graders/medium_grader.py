"""
Medium Grader — "Balanced Strategist"
Agent must mix upskilling and applications across difficulty levels.
Score 0.0–1.0 based on total episode reward normalized by max possible.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from job_market_env import JobMarketEnv

MAX_POSSIBLE_REWARD = 10.0  # theoretical upper bound per episode


def grade(agent, n_episodes: int = 100, seed: int = 42) -> float:
    """
    Grade the agent on balanced strategy.
    Mix of random company difficulties per episode (as in real use).

    Score formula:
        episode_score = (total_reward + 10) / (MAX_POSSIBLE_REWARD + 10)
        [shifted to handle negative rewards, then normalized 0–1]
    """
    env = JobMarketEnv(seed=seed)
    scores = []

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0.0

        while not state.done:
            action = agent.choose_action(state.to_tuple())
            state, reward, done, info = env.step(action)
            total_reward += reward

        # Normalize: worst case is 10 apps all failing = -10 reward
        shifted = total_reward + 10.0
        score   = shifted / (MAX_POSSIBLE_REWARD + 10.0)
        scores.append(min(max(score, 0.0), 1.0))

    return round(sum(scores) / len(scores), 4)


def grade_random(n_episodes: int = 100) -> float:
    import random

    class RandomAgent:
        def choose_action(self, state):
            return random.randint(0, 2)

    return grade(RandomAgent(), n_episodes)


if __name__ == "__main__":
    baseline = grade_random()
    print(f"[Medium Task] Random agent baseline score: {baseline:.4f}")