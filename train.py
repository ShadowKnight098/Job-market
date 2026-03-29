"""
Training script for Job Market Strategy RL Agent.
Run: python train.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
from job_market_env import JobMarketEnv
from agent import QLearningAgent


def train(
    episodes: int = 8000,
    max_applications: int = 10,
    save_path: str = "models/qtable.pkl",
    log_every: int = 500,
) -> list[float]:

    env   = JobMarketEnv(max_applications=max_applications)
    agent = QLearningAgent(
        lr=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9985,
    )

    rewards_log = []
    best_reward = -float("inf")

    print(f"Training for {episodes} episodes...")
    print(f"{'─'*55}")

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0

        while not state.done:
            s_tuple = state.to_tuple()
            action  = agent.choose_action(s_tuple)
            next_state, reward, done, info = env.step(action)
            agent.update(s_tuple, action, reward, next_state.to_tuple(), done)
            state        = next_state
            total_reward += reward

        agent.decay_epsilon()
        rewards_log.append(total_reward)

        if ep % log_every == 0:
            recent_avg = np.mean(rewards_log[-log_every:])
            print(
                f"  Ep {ep:5d}/{episodes} | "
                f"Avg reward: {recent_avg:+.3f} | "
                f"ε: {agent.epsilon:.4f}"
            )
            if recent_avg > best_reward:
                best_reward = recent_avg
                agent.save(save_path)
                print(f"  ✅ New best ({best_reward:+.3f}) — model saved.")

    # Final save
    agent.save(save_path)

    # Save reward log
    log_path = save_path.replace(".pkl", "_rewards.json")
    with open(log_path, "w") as f:
        json.dump(rewards_log, f)

    print(f"\n{'─'*55}")
    print(f"Training complete. Best avg reward: {best_reward:+.3f}")
    print(f"Model   → {save_path}")
    print(f"Rewards → {log_path}")
    return rewards_log


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train()