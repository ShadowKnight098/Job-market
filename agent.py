"""
Q-Learning Agent for Job Market Strategy Environment.
"""

from __future__ import annotations
import numpy as np
import pickle
import os


class QLearningAgent:
    """
    Tabular Q-learning agent.

    State tuple: (skills_level [0-5], applications_left [0-10], company_difficulty [0-2])
    Action space: {0, 1, 2}
    """

    def __init__(
        self,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.lr            = lr
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q[skills(0-5)][apps_left(0-10)][difficulty(0-2)][action(0-2)]
        self.q_table = np.zeros((6, 11, 3, 3))

    def choose_action(self, state: tuple) -> int:
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(3)
        skills, apps, diff = state
        return int(np.argmax(self.q_table[skills, apps, diff]))

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        done: bool,
    ) -> None:
        """Bellman update."""
        s  = state
        ns = next_state
        current_q = self.q_table[s[0], s[1], s[2], action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[ns[0], ns[1], ns[2]])

        self.q_table[s[0], s[1], s[2], action] += self.lr * (target - current_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str = "qtable.pkl") -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "q_table": self.q_table,
                "epsilon": self.epsilon,
                "lr": self.lr,
                "gamma": self.gamma,
            }, f)
        print(f"Agent saved → {path}")

    def load(self, path: str = "qtable.pkl") -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.epsilon = data.get("epsilon", 0.0)
        print(f"Agent loaded ← {path}")

    def greedy(self) -> "QLearningAgent":
        """Return a copy of self with epsilon=0 for evaluation."""
        clone = QLearningAgent.__new__(QLearningAgent)
        clone.q_table      = self.q_table.copy()
        clone.epsilon      = 0.0
        clone.epsilon_min  = 0.0
        clone.epsilon_decay = 1.0
        clone.lr           = self.lr
        clone.gamma        = self.gamma
        return clone