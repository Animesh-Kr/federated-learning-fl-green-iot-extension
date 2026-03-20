# 🌿 Federated Learning — Green IoT Extension
### Carbon-Aware DQN Client Selection on CIFAR-10

**Author:** Animesh Kumar — Newcastle University (CSC8639)  
**Group repo:** [pvajra/federated_learning_gp8](https://github.com/pvajra/federated_learning_gp8)  
**This repo:** Personal extension — harder dataset, more rounds, standalone DQN integration

---

## What This Is

This repository extends the [Group 8 federated learning project](https://github.com/pvajra/federated_learning_gp8) by pushing the carbon-aware DQN client selection strategy further:

| Dimension | Group Baseline | This Extension |
|-----------|---------------|---------------|
| Dataset | MNIST (1-ch, 28×28) | **CIFAR-10** (3-ch, 32×32) |
| Rounds | 30 | **100** |
| Model | 2-conv CNN | **3-conv CNN + BatchNorm** |
| Client selection | Random | **DQN carbon-aware** |
| Carbon tracking | None | **3-zone stochastic grid** |

The key research question: *Does carbon-aware DQN client selection reduce emissions on a harder, more realistic dataset over a longer training horizon?*

**Answer: Yes — 3.9% carbon reduction and 4.3% energy reduction at 100 rounds.**

---

## Results (CIFAR-10, 100 rounds, 50 clients)

| Run | Final Accuracy | Total Energy | Carbon (gCO₂eq) |
|-----|---------------|-------------|-----------------|
| Baseline (random) | **72.60%** | 559.95 J | 0.0542 |
| Proposed (DQN) | 71.76% | **535.71 J** | **0.0521** |
| **Reduction** | −0.84% | **−4.3%** | **−3.9%** |

### Key findings

**Carbon reduction is real and consistent.** The DQN learns to select clients in greener grid zones (Nordic/renewable ~80 gCO₂eq/kWh) over coal-heavy zones (~550 gCO₂eq/kWh). This comes with a marginal accuracy trade-off of 0.84% — a viable green IoT operating point.

**Energy reduction compounds over time.** At 30 rounds the DQN still explores (ε ≈ 0.29) and uses more energy. At 100 rounds exploitation dominates and both energy and carbon drop consistently. This validates the 100-round horizon choice for CIFAR-10.

**Communication energy is the dominant term.** 97% of total energy is communication (model upload/download), not computation. The DQN indirectly reduces this by selecting lower-compression-factor green-zone devices.

---

## How to Run

### Google Colab (recommended)
```python
# Step 1 — Upload FL_CarbonAware_Upgrade.zip to Colab, then:
from google.colab import files
uploaded = files.upload()

import zipfile, os
with zipfile.ZipFile('FL_CarbonAware_Upgrade.zip', 'r') as z:
    z.extractall('/content/fl_upgrade')
os.chdir('/content/fl_upgrade')

# Step 2 — Install dependencies
!pip install "protobuf>=6.31.1"
!pip install "flwr[simulation]>=1.11.0" torch torchvision pandas matplotlib numpy

# Step 3 — Run
!python simulation.py
```
**Runtime:** T4 or L4 GPU → ~25–35 min for 100 rounds

### Local (Windows/Mac/Linux)
```bash
pip install -r requirements.txt
python simulation.py
```

---

## Project Structure

```
├── simulation.py        ← Main runner (baseline vs proposed, 100 rounds)
├── green_strategy.py    ← GreenDRLStrategy: FedAvg + DQN override
├── client.py            ← IoT device simulation (Eq. 2–4 energy model)
├── model.py             ← CIFAR-10 CNN with BatchNorm
├── dataset.py           ← CIFAR-10 federated IID partition
├── carbon_logic.py      ← Stochastic 3-zone carbon grid simulator
├── dqn_agent.py         ← DQN with experience replay + target network
├── requirements.txt
└── results/             ← Generated CSVs and plots after running
    ├── baseline_results.csv
    ├── proposed_results.csv
    ├── comparison_summary.csv
    └── comparison_results.png
```

---

## DQN Architecture

**State vector (dim=5):**
```
[norm_carbon, battery, cpu_norm, dropout_risk, round_progress]
```

**Reward function:**
```
reward = +1.0 × Δaccuracy − 0.4 × norm_energy − 0.6 × norm_carbon
```
Carbon is penalised more heavily than energy (green IoT focus).

**Training:** Online experience replay (buffer=4,000), target network sync every 5 steps, ε decays 1.0 → 0.05 over 100 rounds.

---

## Carbon Grid Zones

| Zone | Region | Base intensity |
|------|--------|---------------|
| 0 — green | Nordic renewables | ~80 gCO₂eq/kWh |
| 1 — mixed | UK / Germany | ~250 gCO₂eq/kWh |
| 2 — coal  | Poland / India  | ~550 gCO₂eq/kWh |

Each zone follows: `intensity(t) = base + amplitude × sin(2πt/24) + N(0, σ²)`

---

## Requirements

```
flwr[simulation]>=1.7.0,<2.0.0
torch>=2.1.0
torchvision>=0.16.0
pandas>=2.0.0
matplotlib>=3.7.0
numpy>=1.24.0
```

---

## Relation to Group Submission

This repo is a **personal extension**, kept separate to avoid destabilising the group's submission codebase. The group repo (`pvajra/federated_learning_gp8`) contains:
- The baseline simulation on MNIST (30 rounds)
- My group contribution: `dqn_agent.py`, `carbon_logic.py`, and `animesh_contribution/`

This repo demonstrates what happens when you scale the same idea to a harder dataset over more rounds — and shows the carbon benefit only becomes visible with sufficient training.

*Animesh Kumar — Newcastle University — March 2026*
