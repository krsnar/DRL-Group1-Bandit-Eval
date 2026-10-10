# DRL Assignment 1 – Group 1 (BITS Pilani WILP)
Paper: **A Framework for Fair Evaluation of Variance-Aware Bandit Algorithms** – E. Wolf, arXiv:2510.27001 (2025)
Official code: https://github.com/eelisee/bandit_playground/tree/v1.0-SKILL2025

## What is in this folder (our proof of work)
| Step | File | What it does |
|---|---|---|
| 1 | `verify_official_tables.py` | Recomputes Tables 3–5 of the paper from the official repo's raw CSVs (all checked values match exactly) |
| 2 | `bandits.py` | Our own re-implementation of the 8 algorithms (ETC, ε-Greedy, UCB, UCB-Tuned, UCB-V, PAC-UCB, EUCBV, UCB-Improved) with numba |
| 3 | `run_experiments.py` | Runs all 16 configurations of Table 1 × 3 scenarios × 2 arm orders × 100 runs, T = 1,000,000 (9,600 runs, ~4.5 min on 2 cores) |
| 4 | `analysis.py` | Paper vs ours comparison, arm-order bias check, VaR(95%) of regret |
| – | `results/` | All CSV outputs (`summary.csv`, `raw_final.csv`, `regret_curves.csv`, `paper_vs_ours.csv`, `arm_order_check.csv`, `official_tables_recomputed.csv`) |

## How to run
```bash
pip install numpy pandas numba
git clone --branch v1.0-SKILL2025 https://github.com/eelisee/bandit_playground.git
python verify_official_tables.py bandit_playground/data
python run_experiments.py 1000000 100      # T, runs
python analysis.py
```
