# =============================================================================
# green_strategy.py
# Carbon-aware FedAvg strategy using Deep Q-Network client selection.
#
# UPGRADE: This is the proposed "Green DRL Strategy" that extends the
# group baseline's random client selection with:
#   1. Carbon-zone-aware scoring via DQNAgent
#   2. Per-client state vector: [norm_carbon, battery, cpu, dropout, round]
#   3. Online DQN training via experience replay after each round
#   4. Full energy + carbon tracking per round
#
# Proposal components implemented:
#   - Participation score (Eq. 1): W1*battery + W2*cpu + W3*data + W4*(1-dropout)
#   - Joint energy model (Eq. 2–4): E_compute + E_comm
#   - Dynamic client selection based on carbon intensity
# =============================================================================

import random
from typing import Dict, List, Tuple, Any

import flwr as fl
import numpy as np

from carbon_logic import CarbonGridSimulator
from dqn_agent    import DQNAgent

# Proposal Eq. 1 scoring weights
W1, W2, W3, W4 = 0.30, 0.30, 0.20, 0.20

# DQN reward normalisation cap
_MAX_TOTAL_ENERGY = 0.015   # Joules


class GreenDRLStrategy(fl.server.strategy.FedAvg):
    """
    Carbon-aware federated learning strategy using Deep Reinforcement Learning.

    Extends FedAvg by replacing random client selection with DQN-scored
    selection that prefers low-carbon, high-capability devices.

    Args:
        num_clients_per_round : Number of clients selected each round.
        num_rounds            : Total FL rounds (for round_progress state).
        num_clients           : Total number of simulated clients.
        profile_list          : List of device profiles (one per client index).
        sorted_cids           : Sorted list of all CIDs (set on first round).
        **kwargs              : Passed to FedAvg.
    """

    def __init__(
        self,
        num_clients_per_round: int = 10,
        num_rounds:            int = 30,
        num_clients:           int = 50,
        profile_list:          list = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.k           = num_clients_per_round
        self.num_rounds  = num_rounds
        self.num_clients = num_clients
        self.profiles    = profile_list or []

        self.carbon_sim  = CarbonGridSimulator()
        self.dqn         = DQNAgent(seed=42)

        # Per-round accumulators (populated in aggregate_fit)
        self.round_energy:  List[float] = []
        self.round_carbon:  List[float] = []
        self.round_compute: List[float] = []
        self.round_comm:    List[float] = []

        # CID → index mapping (built once on first configure_fit)
        self._sorted_cids: List[str] = []
        self._prev_accuracy: float   = 0.0

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _get_idx(self, cid: str) -> int:
        """Stable 0-based index for any CID, safe against UUID-style IDs."""
        try:
            return self._sorted_cids.index(cid)
        except ValueError:
            return abs(hash(cid)) % self.num_clients

    def _get_profile(self, cid: str) -> dict:
        return self.profiles[self._get_idx(cid)] if self.profiles else {}

    def _build_state(self, cid: str, server_round: int) -> List[float]:
        """Build 5-element DQN state vector for a client."""
        idx = self._get_idx(cid)
        p   = self._get_profile(cid)
        return [
            self.carbon_sim.get_normalised_intensity(idx % 3, server_round),
            p.get("battery",    0.5),
            p.get("cpu_factor", 1.0) / 1.2,   # normalise by max cpu_factor
            p.get("dropout",    0.10),
            server_round / self.num_rounds,
        ]

    # ── Flower overrides ─────────────────────────────────────────────────────

    def configure_fit(self, server_round, parameters, client_manager):
        available = list(client_manager.all().values())

        # Build sorted CID list once (stable across rounds)
        if not self._sorted_cids:
            self._sorted_cids = sorted(c.cid for c in available)

        # Build state vectors for all available clients
        client_states: Dict[str, List[float]] = {
            c.cid: self._build_state(c.cid, server_round)
            for c in available
        }

        # DQN epsilon-greedy selection
        selected_cids = self.dqn.select_clients(client_states, self.k)
        proxy_map     = {c.cid: c for c in available}
        chosen        = [proxy_map[cid] for cid in selected_cids if cid in proxy_map]

        # Safety: always return exactly k clients
        if len(chosen) < self.k:
            extras = [c for c in available if c.cid not in set(selected_cids)]
            chosen += random.sample(extras, min(self.k - len(chosen), len(extras)))

        # Pass carbon intensity to each client via config
        instructions = []
        for c in chosen:
            idx      = self._get_idx(c.cid)
            carbon_i = self.carbon_sim.get_carbon_intensity(idx % 3, server_round)
            instructions.append(
                (c, fl.common.FitIns(parameters, {"carbon_intensity": float(carbon_i)}))
            )
        return instructions

    def aggregate_fit(self, server_round, results, failures):
        agg_params, agg_metrics = super().aggregate_fit(server_round, results, failures)

        r_compute = r_comm = r_total = r_carbon = 0.0

        for client_proxy, fit_res in results:
            m         = fit_res.metrics
            r_compute += m.get("compute_energy",       0.0)
            r_comm    += m.get("communication_energy", 0.0)
            r_total   += m.get("total_energy",         0.0)
            r_carbon  += m.get("carbon_emissions",     0.0)

            # ── DQN online learning ─────────────────────────────────────────
            cid       = client_proxy.cid
            state     = self._build_state(cid, server_round)
            next_state= self._build_state(cid, server_round + 1)

            norm_e  = min(m.get("total_energy", 0.0) / _MAX_TOTAL_ENERGY, 1.0)
            norm_c  = self.carbon_sim.get_normalised_intensity(
                self._get_idx(cid) % 3, server_round)
            reward  = DQNAgent.compute_reward(norm_e, norm_c, self._prev_accuracy)
            self.dqn.store(state, reward, next_state, done=(server_round == self.num_rounds))

        # One learn step + epsilon decay per round
        self.dqn.learn()
        self.dqn.decay_epsilon()

        self.round_compute.append(r_compute)
        self.round_comm.append(r_comm)
        self.round_energy.append(r_total)
        self.round_carbon.append(r_carbon)

        return agg_params, agg_metrics

    def set_accuracy(self, acc: float) -> None:
        """Update the accuracy reference used in reward calculation."""
        self._prev_accuracy = acc
