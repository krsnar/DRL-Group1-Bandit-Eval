"""
DRL Assignment 1 - Group 1 (BITS WILP)
Paper: "A Framework for Fair Evaluation of Variance-Aware Bandit Algorithms" (Wolf, arXiv:2510.27001)

Our own re-implementation of the 8 bandit algorithms compared in the paper.
Written with numba so that 1,000,000 steps x 100 runs finishes in minutes.
Every function simulates ONE run on a K-armed Bernoulli bandit and returns
the cumulative regret / reward / suboptimal pulls at the checkpoints.

Common rules (same as the paper's "unified simulation semantics"):
  * one arm pull per time step
  * same initialisation for all algorithms (counts = 0, sums = 0)
  * deterministic seed per run  -> fully reproducible
  * ties in argmax broken by lowest index (paper evaluates both arm orders to remove this bias)
"""
import numpy as np
from numba import njit

CHECKPOINTS = np.array([2, 3, 100, 200, 2000, 10000, 20000, 40000, 60000, 80000,
                        100000, 200000, 400000, 600000, 800000, 1000000], dtype=np.int64)


@njit(cache=True)
def _pull(p):
    # Bernoulli reward  X ~ Ber(p)
    return 1.0 if np.random.random() < p else 0.0


@njit(cache=True)
def _record(t, ck, ci, out, regret, reward, subopt):
    # store metrics when t hits the next checkpoint
    if ci < ck.shape[0] and t == ck[ci]:
        out[ci, 0] = regret
        out[ci, 1] = reward
        out[ci, 2] = subopt
        return ci + 1
    return ci


@njit(cache=True)
def run_one(algo, means, T, seed, ck, prm0, prm1, prm2, prm3):
    """
    algo: 0=ETC 1=eps-Greedy 2=UCB 3=UCB-Tuned 4=UCB-V 5=PAC-UCB 6=EUCBV 7=UCB-Improved
    prm*: algorithm hyper-parameters (see run_experiments.py)
    """
    np.random.seed(seed)
    K = means.shape[0]
    best = np.argmax(means)
    mu_star = means[best]
    n = np.zeros(K)        # T_k(t): pulls per arm
    s = np.zeros(K)        # sum of rewards per arm
    s2 = np.zeros(K)       # sum of squared rewards (for empirical variance)
    out = np.zeros((ck.shape[0], 3))
    ci = 0
    regret = 0.0
    reward = 0.0
    subopt = 0.0
    idx = np.zeros(K)

    # ---------- UCB-Improved bookkeeping (phased elimination) ----------
    active = np.ones(K)
    delta = prm0
    n_phases = int(np.floor(0.5 * np.log2(T / np.e)))
    phase = 0

    # ---------- EUCBV bookkeeping ----------
    rho = prm0
    psi = T / (K * K)
    e_delta = 1.0
    e_m = 0
    e_M = int(np.floor(0.5 * np.log2(T / np.e)))
    n0 = int(np.ceil(np.log(psi * T * e_delta ** 2) / (2 * e_delta)))
    N0 = K * n0

    t = 0
    while t < T:
        t += 1
        arm = 0
        if algo == 0:   # ---------------- Explore-Then-Commit ----------------
            m = int(prm0)
            if t <= m * K:
                arm = (t - 1) % K              # explore uniformly (round robin)
            else:
                for k in range(K):
                    idx[k] = s[k] / n[k]
                arm = np.argmax(idx)          # commit to empirical best
        elif algo == 1:  # ---------------- epsilon-Greedy -------------------
            if np.random.random() < prm0:
                arm = np.random.randint(K)    # explore
            else:
                for k in range(K):
                    idx[k] = s[k] / n[k] if n[k] > 0 else 0.0
                arm = np.argmax(idx)          # exploit
        elif algo in (2, 3, 4, 5):  # -------- index-based UCB family --------
            unplayed = -1
            for k in range(K):
                if n[k] == 0:
                    unplayed = k
                    break
            if unplayed >= 0:
                arm = unplayed                # play each arm once first
            else:
                for k in range(K):
                    q = s[k] / n[k]
                    var = s2[k] / n[k] - q * q          # empirical variance
                    if var < 0:
                        var = 0.0
                    if algo == 2:    # UCB1:  Q + sqrt(2 ln t / T_k)
                        idx[k] = q + np.sqrt(2.0 * np.log(t) / n[k])
                    elif algo == 3:  # UCB-Tuned: Q + sqrt(ln t / T_k * min(1/4, V_k))
                        V = var + np.sqrt(2.0 * np.log(t) / n[k])
                        idx[k] = q + np.sqrt(np.log(t) / n[k] * min(0.25, V))
                    elif algo == 4:  # UCB-V: Q + sqrt(2 var e / T_k) + c*3b e/T_k, e = theta ln t
                        e = prm0 * np.log(t)
                        idx[k] = q + np.sqrt(2.0 * var * e / n[k]) + prm1 * 3.0 * prm2 * e / n[k]
                    else:            # PAC-UCB: e = max(log(K T_k^q / beta), 2)  (variance-independent)
                        e = max(np.log(K * n[k] ** prm2 / prm3), 2.0)
                        idx[k] = q + np.sqrt(2.0 * var * e / n[k]) + prm0 * 3.0 * prm1 * e / n[k]
                arm = np.argmax(idx)
        elif algo == 6:  # ---------------- EUCBV -----------------------------
            if t <= K:
                arm = t - 1
            else:
                lg = np.log(psi * T * e_delta)
                best_lcb = -1e9
                bestv = -1e9
                for k in range(K):
                    q = s[k] / n[k]
                    var = s2[k] / n[k] - q * q
                    w = np.sqrt(rho * (var + 2.0) * lg / (4.0 * n[k]))
                    idx[k] = q + w
                    if active[k] > 0 and q - w > best_lcb:
                        best_lcb = q - w
                for k in range(K):
                    if active[k] > 0 and idx[k] > bestv:
                        bestv = idx[k]
                        arm = k
        else:            # ---------------- UCB-Improved -----------------------
            if phase < n_phases:
                max_pulls = np.ceil(2.0 * np.log(T * delta ** 2) / delta ** 2)
                arm = -1
                for k in range(K):            # round-robin over active arms, ONE pull per step
                    if active[k] > 0 and n[k] <= max_pulls:
                        arm = k
                        break
                if arm == -1:                 # phase finished -> eliminate + halve delta
                    w = np.sqrt(np.log(T * delta ** 2) / (2.0 * max_pulls))
                    lb = -1e9
                    for k in range(K):
                        if active[k] > 0 and s[k] / n[k] - w > lb:
                            lb = s[k] / n[k] - w
                    for k in range(K):
                        if active[k] > 0 and s[k] / n[k] + w < lb:
                            active[k] = 0
                    delta /= 2.0
                    phase += 1
                    t -= 1                    # no pull happened this iteration
                    continue
            else:
                bv = -1e9
                for k in range(K):
                    if active[k] > 0 and s[k] / n[k] > bv:
                        bv = s[k] / n[k]
                        arm = k

        # ---- pull arm, update statistics and metrics ----
        x = _pull(means[arm])
        n[arm] += 1.0
        s[arm] += x
        s2[arm] += x * x
        reward += x
        if arm != best:
            subopt += 1.0
            regret += mu_star - means[arm]     # pseudo-regret: Delta of chosen arm

        # EUCBV: arm elimination + phase update
        if algo == 6 and t > K:
            lg = np.log(psi * T * e_delta)
            best_lcb = -1e9
            for k in range(K):
                if active[k] > 0:
                    q = s[k] / n[k]
                    w = np.sqrt(rho * (s2[k] / n[k] - q * q + 2.0) * lg / (4.0 * n[k]))
                    if q - w > best_lcb:
                        best_lcb = q - w
            for k in range(K):
                if active[k] > 0:
                    q = s[k] / n[k]
                    w = np.sqrt(rho * (s2[k] / n[k] - q * q + 2.0) * lg / (4.0 * n[k]))
                    if q + w < best_lcb:
                        active[k] = 0
            if t >= N0 and e_m <= e_M:
                e_delta /= 2.0
                n0 = int(np.ceil(np.log(psi * T * e_delta ** 2) / (2 * e_delta)))
                na = 0
                for k in range(K):
                    if active[k] > 0:
                        na += 1
                N0 = t + na * n0
                e_m += 1

        ci = _record(t, ck, ci, out, regret, reward, subopt)
    return out
