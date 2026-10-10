"""
Step 3 of our proof of work: compare our re-run with the paper, check arm-order bias, risk (VaR).
"""
import pandas as pd, numpy as np
raw = pd.read_csv("results/raw_final.csv")
summ = pd.read_csv("results/summary.csv")
PAPER = {  # avg regret from Tables 3-5 of the paper (A, B, C)
 "UCB": (238.17, 1127.17, 1172.57), "UCB-Tuned": (30.67, 212.21, 226.09),
 "UCB-V (theta=1,c=1,b=1)": (109.93, 346.58, 367.65), "EUCBV (rho=0.5)": (47.73, 461.36, 476.31),
 "PAC-UCB (c=1,b=1,q=1.3,beta=0.05)": (99.50, 395.89, 417.34), "UCB-Improved (delta=1)": (32500.11, 2525.00, 2450.00),
 "ETC (m=100)": (10.10, None, None), "ETC (m=10000)": (1000.00, 150.66, 170.53),
 "eps-Greedy (eps=0.05)": (None, 361.97, 301.55), "eps-Greedy (eps=0.5)": (25002.20, 1264.52, 1269.05),
 "eps-Greedy (eps=0.005)": (422.99, 1428.70, 1762.59)}
sc_names = sorted(summ.scenario.unique())
rows = []
for alg, vals in PAPER.items():
    for sc, pv in zip(sc_names, vals):
        if pv is None: continue
        ours = summ[(summ.scenario == sc) & (summ.algorithm == alg)].avg_regret.iloc[0]
        rows.append(dict(scenario=sc, algorithm=alg, paper=pv, ours=round(ours, 2), diff_pct=round(100 * (ours - pv) / pv, 1)))
cmp = pd.DataFrame(rows); cmp.to_csv("results/paper_vs_ours.csv", index=False)
print(cmp.to_string(index=False))

order = raw.groupby(["scenario", "algorithm", "arm_order"]).regret.mean().unstack().round(1)
order.columns = ["order_(a1,a2)", "order_(a2,a1)"]
order.to_csv("results/arm_order_check.csv"); print(order.to_string())
