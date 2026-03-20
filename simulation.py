# =============================================================================
# simulation.py
# Main experiment runner for the Carbon-Aware FL Upgrade project.
#
# COMPATIBILITY FIX: Set protobuf implementation to pure-Python before any
# other imports. This resolves the conflict between flwr 1.9.0 (needs older
# protobuf) and tensorflow 2.x (needs newer protobuf) on the same machine.
import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
#
# Runs two experiments and compares them:
#   baseline : Random client selection (group baseline approach)
#   proposed : DQN carbon-aware selection (this upgrade)
#
# Both use CIFAR-10 (upgrade from group's MNIST) to demonstrate scalability.
#
# Usage:
#   python simulation.py
#
# Output CSVs (in ./results/):
#   baseline_results.csv
#   proposed_results.csv
#   comparison_summary.csv
# =============================================================================

import os
import random
import logging
from pathlib import Path

import flwr as fl
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from client        import FlowerClient
from dataset       import load_datasets
from model         import Net
from green_strategy import GreenDRLStrategy
from carbon_logic   import CarbonGridSimulator

# ── Suppress verbose Ray / Flower logs ────────────────────────────────────────
logging.getLogger("flwr").setLevel(logging.WARNING)
os.environ.setdefault("RAY_DEDUP_LOGS", "0")

# ── Configuration ──────────────────────────────────────────────────────────────
NUM_CLIENTS       = 50
NUM_ROUNDS        = 100
CLIENTS_PER_ROUND = 10
SEED              = 42
RESULTS_DIR       = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Seeds ──────────────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Device profiles (proposal Table II) ───────────────────────────────────────
_PROFILE_TEMPLATES = {
    "sensor": {"battery": 0.35, "cpu_factor": 0.50, "compression": 0.25, "dropout": 0.30},
    "mobile": {"battery": 0.65, "cpu_factor": 0.80, "compression": 0.50, "dropout": 0.15},
    "edge":   {"battery": 1.00, "cpu_factor": 1.20, "compression": 1.00, "dropout": 0.05},
}
_prng = random.Random(SEED)
PROFILE_LIST = [
    dict(_PROFILE_TEMPLATES[_prng.choice(list(_PROFILE_TEMPLATES.keys()))])
    for _ in range(NUM_CLIENTS)
]

# ── Load datasets once ─────────────────────────────────────────────────────────
print("Loading CIFAR-10...")
TRAINLOADERS, TESTLOADER = load_datasets(NUM_CLIENTS, seed=SEED)
print(f"✅ {NUM_CLIENTS} clients | {len(TRAINLOADERS[0].dataset):,} samples each")

_cnt = {}
for t in _PROFILE_TEMPLATES:
    _cnt[t] = sum(1 for p in PROFILE_LIST if p["cpu_factor"] == _PROFILE_TEMPLATES[t]["cpu_factor"])
print(f"   Device mix: {_cnt}")


# ── CID mapping (UUID-safe, built on first configure_fit) ─────────────────────
_ALL_CIDS_SORTED = []


def get_idx(cid: str) -> int:
    try:    return _ALL_CIDS_SORTED.index(cid)
    except: return abs(hash(cid)) % NUM_CLIENTS


# ── Global test-set evaluation ─────────────────────────────────────────────────
def evaluate_global_model(parameters) -> float:
    model = Net()
    model.load_state_dict(
        {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)},
        strict=True)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in TESTLOADER:
            correct += (model(x).argmax(1) == y).sum().item()
            total   += y.size(0)
    return correct / max(total, 1)


# ── Experiment runners ─────────────────────────────────────────────────────────

def run_baseline() -> dict:
    """Random client selection — mirrors the group baseline approach."""
    global _ALL_CIDS_SORTED
    _ALL_CIDS_SORTED = []

    print(f"\n{'━'*55}")
    print(f"  BASELINE  |  {NUM_ROUNDS} rounds  |  {CLIENTS_PER_ROUND}/round")
    print(f"{'━'*55}")

    acc_log=[]; comp_log=[]; comm_log=[]; total_log=[]; carbon_log=[]
    carbon_sim = CarbonGridSimulator()

    def evaluate_fn(server_round, parameters, config):
        acc = evaluate_global_model(parameters)
        acc_log.append(acc)
        print(f"  Round {server_round:03d}/{NUM_ROUNDS}  |  accuracy = {acc:.4f}", flush=True)
        return 0.0, {"accuracy": acc}

    def client_fn(cid: str) -> FlowerClient:
        idx = get_idx(cid)
        return FlowerClient(TRAINLOADERS[idx], PROFILE_LIST[idx],
                            carbon_intensity=carbon_sim.get_carbon_intensity(idx % 3, 1))

    class BaselineStrategy(fl.server.strategy.FedAvg):
        def configure_fit(self, server_round, parameters, client_manager):
            global _ALL_CIDS_SORTED
            avail = list(client_manager.all().values())
            if not _ALL_CIDS_SORTED:
                _ALL_CIDS_SORTED = sorted(c.cid for c in avail)
            chosen = random.sample(avail, min(CLIENTS_PER_ROUND, len(avail)))
            return [(c, fl.common.FitIns(parameters, {})) for c in chosen]

        def aggregate_fit(self, server_round, results, failures):
            agg_p, agg_m = super().aggregate_fit(server_round, results, failures)
            rc = rm = rt = rcarbon = 0.0
            for _, res in results:
                m = res.metrics
                rc      += m.get("compute_energy",       0.0)
                rm      += m.get("communication_energy", 0.0)
                rt      += m.get("total_energy",         0.0)
                rcarbon += m.get("carbon_emissions",     0.0)
            comp_log.append(rc); comm_log.append(rm)
            total_log.append(rt); carbon_log.append(rcarbon)
            return agg_p, agg_m

    fl.simulation.start_simulation(
        client_fn   = client_fn,
        num_clients = NUM_CLIENTS,
        config      = fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy    = BaselineStrategy(
            fraction_fit=CLIENTS_PER_ROUND/NUM_CLIENTS,
            min_fit_clients=CLIENTS_PER_ROUND,
            min_available_clients=NUM_CLIENTS,
            evaluate_fn=evaluate_fn,
            fraction_evaluate=0.0, min_evaluate_clients=0,
        ),
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={"num_cpus": 2, "num_gpus": 0,
                       "include_dashboard": False, "ignore_reinit_error": True},
    )

    ml = min(len(acc_log), len(comp_log), len(comm_log), len(total_log), len(carbon_log))
    df = pd.DataFrame({"round": range(1, ml+1),
                        "accuracy":             acc_log[:ml],
                        "compute_energy":       comp_log[:ml],
                        "communication_energy": comm_log[:ml],
                        "total_energy":         total_log[:ml],
                        "carbon_emissions":     carbon_log[:ml]})
    df.to_csv(RESULTS_DIR / "baseline_results.csv", index=False)
    print(f"  ✓ Saved baseline_results.csv")
    return {"accuracy": acc_log[:ml], "total_energy": total_log[:ml],
            "carbon_emissions": carbon_log[:ml],
            "compute_energy": comp_log[:ml], "communication_energy": comm_log[:ml]}


def run_proposed() -> dict:
    """DQN carbon-aware selection — the upgrade contribution."""
    global _ALL_CIDS_SORTED
    _ALL_CIDS_SORTED = []

    print(f"\n{'━'*55}")
    print(f"  PROPOSED (DQN)  |  {NUM_ROUNDS} rounds  |  {CLIENTS_PER_ROUND}/round")
    print(f"{'━'*55}")

    acc_log = []

    strategy = GreenDRLStrategy(
        num_clients_per_round = CLIENTS_PER_ROUND,
        num_rounds            = NUM_ROUNDS,
        num_clients           = NUM_CLIENTS,
        profile_list          = PROFILE_LIST,
        fraction_fit          = CLIENTS_PER_ROUND / NUM_CLIENTS,
        min_fit_clients       = CLIENTS_PER_ROUND,
        min_available_clients = NUM_CLIENTS,
        fraction_evaluate     = 0.0,
        min_evaluate_clients  = 0,
        evaluate_fn           = lambda sr, params, cfg: _eval_fn(sr, params, cfg),
    )

    def _eval_fn(server_round, parameters, config):
        acc = evaluate_global_model(parameters)
        acc_log.append(acc)
        strategy.set_accuracy(acc)
        print(f"  Round {server_round:03d}/{NUM_ROUNDS}  |  accuracy = {acc:.4f}", flush=True)
        return 0.0, {"accuracy": acc}

    def client_fn(cid: str) -> FlowerClient:
        idx = get_idx(cid)
        return FlowerClient(TRAINLOADERS[idx], PROFILE_LIST[idx])

    fl.simulation.start_simulation(
        client_fn   = client_fn,
        num_clients = NUM_CLIENTS,
        config      = fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy    = strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={"num_cpus": 2, "num_gpus": 0,
                       "include_dashboard": False, "ignore_reinit_error": True},
    )

    ml = min(len(acc_log), len(strategy.round_energy),
             len(strategy.round_compute), len(strategy.round_comm),
             len(strategy.round_carbon))
    df = pd.DataFrame({"round": range(1, ml+1),
                        "accuracy":             acc_log[:ml],
                        "compute_energy":       strategy.round_compute[:ml],
                        "communication_energy": strategy.round_comm[:ml],
                        "total_energy":         strategy.round_energy[:ml],
                        "carbon_emissions":     strategy.round_carbon[:ml]})
    df.to_csv(RESULTS_DIR / "proposed_results.csv", index=False)
    print(f"  ✓ Saved proposed_results.csv")
    return {"accuracy": acc_log[:ml],
            "total_energy": strategy.round_energy[:ml],
            "carbon_emissions": strategy.round_carbon[:ml],
            "compute_energy": strategy.round_compute[:ml],
            "communication_energy": strategy.round_comm[:ml]}


# ── Run both ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  Carbon-Aware FL — Upgrade Experiment")
print("  CIFAR-10 | 50 clients | DQN vs Random selection")
print("="*55)

baseline = run_baseline()
proposed = run_proposed()


# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("  RESULTS SUMMARY")
print(f"{'='*55}")
print(f"  {'':15}  {'Final Acc':>9}  {'Total E (J)':>11}  {'CO2 (gCO2)':>10}")
print(f"  {'-'*15}  {'-'*9}  {'-'*11}  {'-'*10}")
for name, res in [("baseline", baseline), ("proposed", proposed)]:
    print(f"  {name:<15}  {res['accuracy'][-1]:>9.4f}  "
          f"{sum(res['total_energy']):>11.5f}  "
          f"{sum(res['carbon_emissions']):>10.4f}")

be = sum(baseline["total_energy"]); pe = sum(proposed["total_energy"])
bc = sum(baseline["carbon_emissions"]); pc = sum(proposed["carbon_emissions"])
if be > 0: print(f"\n  Energy reduction : {(be-pe)/be*100:.1f}%")
if bc > 0: print(f"  Carbon reduction : {(bc-pc)/bc*100:.1f}%")


# ── Plots ──────────────────────────────────────────────────────────────────────
def smooth(v, w=3):
    return pd.Series(v).rolling(w, min_periods=1).mean().tolist()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"Carbon-Aware FL Upgrade — CIFAR-10 | {NUM_CLIENTS} clients | {NUM_ROUNDS} rounds",
             fontsize=12, fontweight="bold")

for ax, (key, title, yl) in zip(axes, [
    ("accuracy",         "Global Test Accuracy",       "Accuracy"),
    ("total_energy",     "Total Energy per Round",     "Energy (J)"),
    ("carbon_emissions", "Carbon Emissions per Round", "gCO2eq"),
]):
    for name, res, colour, ls in [
        ("baseline", baseline, "#1565C0", "-"),
        ("proposed", proposed, "#B71C1C", "-"),
    ]:
        rounds = range(1, len(res[key])+1)
        ax.plot(rounds, smooth(res[key]), label=name, color=colour, linestyle=ls, linewidth=2)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Round"); ax.set_ylabel(yl)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = RESULTS_DIR / "comparison_results.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  ✓ Plot saved: {plot_path}")

# Summary CSV
summary = pd.DataFrame([
    {"run": name,
     "final_accuracy": res["accuracy"][-1],
     "peak_accuracy":  max(res["accuracy"]),
     "total_energy":   sum(res["total_energy"]),
     "total_carbon":   sum(res["carbon_emissions"]),
     "total_compute":  sum(res["compute_energy"]),
     "total_comm":     sum(res["communication_energy"])}
    for name, res in [("baseline", baseline), ("proposed", proposed)]
])
summary.to_csv(RESULTS_DIR / "comparison_summary.csv", index=False)
print(f"  ✓ Summary saved: {RESULTS_DIR}/comparison_summary.csv")
print("\nDone.")
