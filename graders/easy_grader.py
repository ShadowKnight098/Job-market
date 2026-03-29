"""
Easy Grader — "Startup Hustler"
Measures how well the agent applies to small companies.
Score 0.0–1.0. Graded purely on shortlist rate at small companies.
"""

from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from job_market_env import JobMarketEnv


def grade(agent, n_episodes: int = 100, seed: int = 42) -> float:
    """
    Run agent for n_episodes and return a score in [0.0, 1.0].

    Scoring:
        - Each episode: shortlisted_count / applications_used
        - Average across all episodes
        - Normalized to [0, 1]
    """
    env = JobMarketEnv(seed=seed)
    scores = []

    for ep in range(n_episodes):
        state = env.reset()
        # force easy mode: small company difficulty
        state.company_difficulty = 0
        total_reward = 0.0
        apps_used = 0

        while not state.done:
            action = agent.choose_action(state.to_tuple())
            state, reward, done, info = env.step(action)
            total_reward += reward
            if action in (0, 1):
                apps_used += 1

        # Score = shortlisted / apps used (prevent div by zero)
        if apps_used > 0:
            ep_score = state.shortlisted_count / max(apps_used, 1)
        else:
            ep_score = 0.0

        # Normalize: a perfect agent could shortlist ~4–5 times in 10 apps
        normalized = min(ep_score / 0.5, 1.0)
        scores.append(normalized)

    return round(sum(scores) / len(scores), 4)


def grade_random(n_episodes: int = 100) -> float:
    """Baseline: random agent score on this task."""
    import random

    class RandomAgent:
        def choose_action(self, state):
            return random.randint(0, 2)

    return grade(RandomAgent(), n_episodes)


if __name__ == "__main__":
    baseline = grade_random()
    print(f"[Easy Task] Random agent baseline score: {baseline:.4f}")