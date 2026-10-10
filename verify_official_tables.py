"""
Step 1 of our proof of work:
Clone the official Bandit Playground repo (tag v1.0-SKILL2025) and recompute
Tables 3, 4, 5 of the paper directly from its raw per-run CSV files.

  git clone --branch v1.0-SKILL2025 https://github.com/eelisee/bandit_playground.git
  python verify_official_tables.py bandit_playground/data
"""
import sys, os, glob
import pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else "bandit_playground/data"
T = 1_000_000
SCEN = {"A": "800_900", "B": "895_900", "C": "890_895"}   # file suffix = arm means x1000
# values printed in the paper (Tables 3-5) for a few algorithms, to compare against
PAPER = {("A", "UCB-Tuned/default"): 30.67, ("A", "UCB/default"): 238.17, ("A", "ETC/exploration_rounds_100"): 10.10,
         ("B", "UCB-Tuned/default"): 212.21, ("B", "UCB/default"): 1127.17, ("B", "ETC/exploration_rounds_10000"): 150.66,
         ("C", "UCB-Tuned/default"): 226.09, ("C", "UCB/default"): 1172.57, ("C", "ETC/exploration_rounds_10000"): 170.53,
         ("C", "UCB-V/theta_1_c_1_b_1"): 367.65, ("C", "EUCBV/rho_0_5"): 476.31, ("C", "PAC-UCB/c_1_b_1_q_1_3_beta_0_05"): 417.34}

rows = []
for sc, suf in SCEN.items():
    for d in sorted(glob.glob(os.path.join(DATA, "*", "*"))):
        f = os.path.join(d, f"results_{suf}.csv")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        last = df[df.Timestep == T]
        key = os.path.relpath(d, DATA)
        rows.append(dict(scenario=sc, config=key, runs=len(last),
                         avg_regret=round(last["Total Regret"].mean(), 2),
                         reward_var=round(last["Total Reward"].var(), 2),
                         subopt_ratio=round(last["Suboptimal Arms"].mean() / T, 5),
                         paper_value=PAPER.get((sc, key), "")))
out = pd.DataFrame(rows)
out.to_csv("results/official_tables_recomputed.csv", index=False)
chk = out[out.paper_value != ""]
print(chk[["scenario", "config", "runs", "avg_regret", "paper_value"]].to_string(index=False))
print("\nAll checked values match the paper:", all(abs(chk.avg_regret - chk.paper_value.astype(float)) < 0.01))
