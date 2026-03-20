# =============================================================================
# client.py
# Flower NumPyClient representing a simulated heterogeneous IoT device.
#
# Implements the proposal joint energy cost model:
#   E_compute = CPU_cycles × α    (Eq. 3)
#   E_comm    = data_transmitted × β   (Eq. 4)
#   E_total   = E_compute + E_comm     (Eq. 2)
#
# UPGRADE fixes vs old version:
#   - CrossEntropyLoss instantiated ONCE outside the training loop (was bug)
#   - Clients pinned to CPU (prevents OOM when Flower spawns 50 actors)
#   - Full metrics dict: compute_energy, communication_energy, total_energy,
#     carbon_emissions (old version only returned total_energy)
#   - evaluate() fully implemented (was missing)
#   - quantisation support (float16 upload)
# =============================================================================

import os
import random

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import Net

# Energy model constants (from IoT profiling literature)
ALPHA: float = 1e-6    # Joules per CPU cycle
BETA:  float = 5e-7    # Joules per transmitted byte
_J_TO_KWH: float = 1.0 / 3_600_000.0

# All FL virtual clients run on CPU to prevent VRAM OOM with 50 Ray actors
_CLIENT_DEVICE = torch.device("cpu")


class FlowerClient(fl.client.NumPyClient):
    """
    Simulated IoT device for federated training.

    Args:
        trainloader      : Local data shard DataLoader.
        device_profile   : {battery, cpu_factor, compression, dropout}.
        use_quantization : Upload float16 parameters (halves comm bytes).
        carbon_intensity : gCO2eq/kWh for this client's grid zone.
    """

    def __init__(
        self,
        trainloader,
        device_profile: dict,
        use_quantization: bool = False,
        carbon_intensity: float = 350.0,
    ) -> None:
        self.model            = Net().to(_CLIENT_DEVICE)
        self.trainloader      = trainloader
        self.profile          = device_profile
        self.use_quantization = use_quantization
        self.carbon_intensity = carbon_intensity

        # Instantiated ONCE — not inside the training loop (was a bug in old version)
        self._criterion = nn.CrossEntropyLoss()

    # ── Flower interface ──────────────────────────────────────────────────────

    def get_parameters(self, config: dict) -> list:
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, parameters: list) -> None:
        state_dict = {
            k: torch.tensor(v, device=_CLIENT_DEVICE)
            for k, v in zip(self.model.state_dict().keys(), parameters)
        }
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: list, config: dict) -> tuple:
        dropout_prob = self.profile.get("dropout", 0.10)

        # Stochastic device dropout
        if random.random() < dropout_prob:
            return self.get_parameters({}), 0, self._zero_metrics()

        self.set_parameters(parameters)

        cpu_factor  = self.profile.get("cpu_factor",  1.0)
        compression = self.profile.get("compression", 1.0)

        optimizer = optim.SGD(
            self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4
        )
        self.model.train()
        steps       = 0
        max_batches = max(1, int(len(self.trainloader) * cpu_factor))

        for i, (data, target) in enumerate(self.trainloader):
            if i >= max_batches:
                break
            data, target = data.to(_CLIENT_DEVICE), target.to(_CLIENT_DEVICE)
            optimizer.zero_grad()
            self._criterion(self.model(data), target).backward()
            optimizer.step()
            steps += 1

        # ── Energy calculation (Eq. 2, 3, 4) ─────────────────────────────────
        cpu_cycles     = steps * cpu_factor * 1_000
        compute_energy = cpu_cycles * ALPHA                         # Joules

        bytes_per_param  = 2 if self.use_quantization else 4       # fp16 vs fp32
        model_params     = sum(p.numel() for p in self.model.parameters())
        data_transmitted = model_params * bytes_per_param * compression
        comm_energy      = data_transmitted * BETA                  # Joules

        total_energy     = compute_energy + comm_energy
        carbon_emissions = total_energy * _J_TO_KWH * self.carbon_intensity  # gCO2eq

        params = self.get_parameters({})
        if self.use_quantization:
            params = [p.astype(np.float16) for p in params]

        return params, len(self.trainloader.dataset), {
            "compute_energy":       compute_energy,
            "communication_energy": comm_energy,
            "total_energy":         total_energy,
            "carbon_emissions":     carbon_emissions,
            "battery":              self.profile.get("battery",     0.5),
            "cpu":                  self.profile.get("cpu_factor",  1.0),
            "compression":          self.profile.get("compression", 1.0),
            "data_size":            len(self.trainloader.dataset),
            "dropout_risk":         dropout_prob,
            "quantized":            int(self.use_quantization),
            "training_steps":       steps,
        }

    def evaluate(self, parameters: list, config: dict) -> tuple:
        """Required by Flower. Evaluates on the client's local shard."""
        self.set_parameters(parameters)
        self.model.eval()
        loss = correct = total = 0
        with torch.no_grad():
            for data, target in self.trainloader:
                data, target = data.to(_CLIENT_DEVICE), target.to(_CLIENT_DEVICE)
                out     = self.model(data)
                loss   += self._criterion(out, target).item()
                correct += (out.argmax(1) == target).sum().item()
                total   += target.size(0)
        return float(loss), total, {"accuracy": correct / max(total, 1)}

    def _zero_metrics(self) -> dict:
        return {
            "compute_energy": 0.0, "communication_energy": 0.0,
            "total_energy": 0.0,   "carbon_emissions": 0.0,
            "battery":   self.profile.get("battery",     0.5),
            "cpu":       self.profile.get("cpu_factor",  1.0),
            "compression": self.profile.get("compression", 1.0),
            "data_size": 0, "dropout_risk": self.profile.get("dropout", 0.10),
            "quantized": int(self.use_quantization), "training_steps": 0,
        }
