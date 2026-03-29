"""
Baseline Inference Script — reproducible scores across all 3 tasks.
Run: python scripts/baseline_inference.py

This script:
1. Trains a fresh agent for 8000 episodes
2. Runs the trained agent on all 3 graders (easy / medium / hard)
3. Also runs a random baseline for comparison
4. Prints a reproducibility report
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import random
from job_market_env import JobMarketEnv
from agent                 import QLearningAgent
from graders.easy_grader import grade as grade_easy
from graders.medium_grader import grade as grade_medium
from graders.hard_grader import grade as grade_hard
from train                 import train

SEED          = 42
N_EVAL_EPISODES = 200
MODEL_PATH    = "models/qtable.pkl"


# ── Random baseline ────────────────────────────────────────────────────

class RandomAgent:
    def choose_action(self, state):
        return random.randint(0, 2)


# ── Smart heuristic baseline ───────────────────────────────────────────

class HeuristicAgent:
    """
    Rule-based agent:
      - If skills < 3 and apps > 5 → upskill
      - If skills >= 4 and company_difficulty == 2 → apply big
      - Else → apply small
    """
    def choose_action(self, state):
        skills, apps, diff = state
        if skills < 3 and apps > 5:
            return 2  # upskill
        if skills >= 4 and diff == 2:
            return 1  # apply big
        return 0      # apply small


# ── Main ───────────────────────────────────────────────────────────────

def run_baseline():
    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("  JOB MARKET RL — BASELINE INFERENCE REPORT")
    print("=" * 60)

    # ── Train if no model exists ──
    if not os.path.exists(MODEL_PATH):
        print("\n[1/2] Training agent (8000 episodes)...")
        os.makedirs("models", exist_ok=True)
        train(episodes=8000, save_path=MODEL_PATH)
    else:
        print(f"\n[1/2] Loading existing model: {MODEL_PATH}")

    # ── Load trained agent ──
    trained_agent = QLearningAgent(epsilon=0.0)
    trained_agent.load(MODEL_PATH)
    trained_agent.epsilon = 0.0  # greedy evaluation

    # ── Evaluate all agents on all tasks ──
    agents = {
        "Random":    RandomAgent(),
        "Heuristic": HeuristicAgent(),
        "Q-Learning (trained)": trained_agent,
    }

    tasks = {
        "easy   (Startup Hustler)":     grade_easy,
        "medium (Balanced Strategist)": grade_medium,
        "hard   (FAANG or Bust)":       grade_hard,
    }

    results = {}
    print(f"\n[2/2] Evaluating {N_EVAL_EPISODES} episodes per task...\n")

    for agent_name, agent in agents.items():
        results[agent_name] = {}
        for task_name, grader in tasks.items():
            score = grader(agent, n_episodes=N_EVAL_EPISODES, seed=SEED)
            results[agent_name][task_name] = score

    # ── Print report ──
    print(f"\n{'─'*60}")
    print(f"  {'Agent':<25} {'Easy':>8} {'Medium':>8} {'Hard':>8}  {'Avg':>8}")
    print(f"{'─'*60}")

    for agent_name, scores in results.items():
        vals = list(scores.values())
        avg  = sum(vals) / len(vals)
        print(
            f"  {agent_name:<25} "
            f"{vals[0]:>8.4f} "
            f"{vals[1]:>8.4f} "
            f"{vals[2]:>8.4f}  "
            f"{avg:>8.4f}"
        )

    print(f"{'─'*60}")
    print(f"\n  Seed used: {SEED}")
    print(f"  Episodes per task: {N_EVAL_EPISODES}")
    print(f"  Scores are in [0.0, 1.0]\n")

    # Save results
    with open("models/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Full results saved → models/baseline_results.json")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_baseline()