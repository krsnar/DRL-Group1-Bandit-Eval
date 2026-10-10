"""
Runs all algorithm configurations of Table 1 on the 3 scenarios of the paper.
T = 1,000,000 steps, 100 independent runs, both arm orders (permutation check).
Outputs: results/raw_final.csv (per run) and results/summary.csv (averages).
"""
import sys, time, os
import numpy as np, pandas as pd
from multiprocessing import Pool
from bandits import run_one, CHECKPOINTS as ALL_CK

T = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
SEED = 42
CHECKPOINTS = np.unique(np.append(ALL_CK[ALL_CK <= T], T))

SCENARIOS = {"A (0.8 vs 0.9)": [0.8, 0.9],
             "B (0.895 vs 0.9)": [0.895, 0.9],
             "C (0.89 vs 0.895)": [0.89, 0.895]}

# name, algo id, params  (Table 1 of the paper)
CONFIGS = [(f"ETC (m={m})", 0, (m, 0, 0, 0)) for m in [10, 100, 1000, 10000, 100000]] + \
          [(f"eps-Greedy (eps={e})", 1, (e, 0, 0, 0)) for e in [0.5, 0.1, 0.05, 0.01, 0.005]] + [
          ("UCB", 2, (0, 0, 0, 0)),
          ("UCB-Tuned", 3, (0, 0, 0, 0)),
          ("UCB-V (theta=1,c=1,b=1)", 4, (1, 1, 1, 0)),
          ("PAC-UCB (c=1,b=1,q=1.3,beta=0.05)", 5, (1, 1, 1.3, 0.05)),
          ("EUCBV (rho=0.5)", 6, (0.5, 0, 0, 0)),
          ("UCB-Improved (delta=1)", 7, (1, 0, 0, 0))]

CATEGORY = {0: "Standard", 1: "Standard", 2: "Standard", 3: "Variance-aware", 4: "Variance-aware",
            5: "Non-variance-aware", 6: "Variance-aware", 7: "Non-variance-aware"}


def job(args):
    sc, means, order, name, algo, prm = args
    m = np.array(means if order == 0 else means[::-1], dtype=np.float64)
    rows, curves = [], []
    for r in range(RUNS):
        out = run_one(algo, m, T, SEED * 100000 + r, CHECKPOINTS, *[float(p) for p in prm])
        rows.append((sc, name, CATEGORY[algo], order, r, out[-1, 0], out[-1, 1], out[-1, 2]))
        curves.append(out[:, 0])
    return rows, (sc, name, order, np.mean(curves, axis=0))


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    jobs = [(sc, mu, o, n, a, p) for sc, mu in SCENARIOS.items() for o in (0, 1) for n, a, p in CONFIGS]
    t0 = time.time()
    with Pool() as pool:
        res = pool.map(job, jobs)
    raw = pd.DataFrame([r for rows, _ in res for r in rows],
                       columns=["scenario", "algorithm", "category", "arm_order", "run", "regret", "reward", "subopt_pulls"])
    raw.to_csv("results/raw_final.csv", index=False)
    cur = pd.DataFrame([dict(scenario=s, algorithm=n, arm_order=o, **{f"t={c}": v for c, v in zip(CHECKPOINTS, vals)})
                        for _, (s, n, o, vals) in res])
    cur.to_csv("results/regret_curves.csv", index=False)
    summ = raw[raw.arm_order == 0].groupby(["scenario", "algorithm", "category"]).agg(
        avg_regret=("regret", "mean"), std_regret=("regret", "std"),
        reward_var=("reward", "var"), subopt_ratio=("subopt_pulls", lambda x: x.mean() / T),
        VaR95_regret=("regret", lambda x: np.quantile(x, 0.95))).reset_index()
    summ.to_csv("results/summary.csv", index=False)
    print(f"done in {time.time()-t0:.0f}s")
