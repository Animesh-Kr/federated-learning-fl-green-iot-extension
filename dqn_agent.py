# =============================================================================
# dqn_agent.py
# Deep Q-Network for carbon-aware federated client selection.
#
# UPGRADE from group baseline: adds carbon intensity as an explicit
# optimisation dimension alongside energy and device capability.
#
# State vector (dim=5):
#   [norm_carbon, battery_norm, cpu_norm, dropout_risk, round_progress]
#
# The agent outputs a scalar Q-value per client. Top-k clients by
# Q-value are selected. Trained online via experience replay.
# =============================================================================

import random
from collections import deque
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Hyper-parameters
STATE_DIM      = 5
HIDDEN_DIM     = 128
LR             = 3e-4
GAMMA          = 0.95
EPSILON_START  = 1.0
EPSILON_MIN    = 0.05
EPSILON_DECAY  = 0.96       # per FL round
BATCH_SIZE     = 64
REPLAY_CAP     = 4_000
TARGET_SYNC    = 5          # sync target net every N learn() calls


class DQNet(nn.Module):
    """3-layer MLP with LayerNorm: state vector → scalar Q-value."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """
    Online DQN agent that scores FL clients for carbon-aware selection.

    Per FL round:
        1. select_clients(states, k) → list of selected client IDs
        2. store(s, r, s', done)     → push transition to replay buffer
        3. learn()                   → one gradient step
        4. decay_epsilon()           → reduce exploration rate
    """

    def __init__(self, seed: int = 42) -> None:
        torch.manual_seed(seed)
        random.seed(seed)

        self.q_net   = DQNet()
        self.target  = DQNet()
        self.target.load_state_dict(self.q_net.state_dict())
        self.target.eval()

        self.optimizer  = optim.Adam(self.q_net.parameters(), lr=LR)
        self.memory     = deque(maxlen=REPLAY_CAP)
        self.epsilon    = EPSILON_START
        self._step      = 0

    def select_client_score(self, state: List[float]) -> float:
        """Return scalar Q-value for a single client state vector."""
        t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return float(self.q_net(t).item())

    def select_clients(
        self,
        client_states: Dict[str, List[float]],
        k: int,
    ) -> List[str]:
        """
        Epsilon-greedy selection:
          - Explore (prob ε): random sample
          - Exploit (prob 1-ε): top-k by Q-value
        """
        cids = list(client_states.keys())
        k    = min(k, len(cids))

        if random.random() < self.epsilon:
            return random.sample(cids, k)

        scores = {c: self.select_client_score(client_states[c]) for c in cids}
        return sorted(cids, key=lambda c: scores[c], reverse=True)[:k]

    def store(self, state, reward, next_state, done) -> None:
        """Push (s, r, s', done) transition to replay buffer."""
        self.memory.append((state, reward, next_state, float(done)))

    def learn(self) -> float:
        """One gradient step on a random mini-batch. Returns loss."""
        if len(self.memory) < BATCH_SIZE:
            return 0.0

        batch = random.sample(self.memory, BATCH_SIZE)
        states, rewards, next_states, dones = zip(*batch)

        s  = torch.tensor(np.array(states),      dtype=torch.float32)
        r  = torch.tensor(rewards,               dtype=torch.float32).unsqueeze(1)
        ns = torch.tensor(np.array(next_states), dtype=torch.float32)
        d  = torch.tensor(dones,                 dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            target_q = r + GAMMA * self.target(ns) * (1.0 - d)

        pred_q = self.q_net(s)
        loss   = nn.functional.mse_loss(pred_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._step += 1
        if self._step % TARGET_SYNC == 0:
            self.target.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self) -> None:
        """Call once per FL round after learn()."""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    @staticmethod
    def compute_reward(norm_energy: float, norm_carbon: float, acc_gain: float) -> float:
        """
        Reward = +1.0 × Δaccuracy − 0.4 × norm_energy − 0.6 × norm_carbon
        Carbon weighted higher (green IoT focus).
        """
        return 1.0 * acc_gain - 0.4 * norm_energy - 0.6 * norm_carbon
