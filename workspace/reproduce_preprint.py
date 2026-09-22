"""Validate our ma-QAOA + SAP/OSAP implementation against arXiv:2609.02793's own published numbers.

PRE-REGISTERED (criteria fixed in this docstring and committed before the run). Their setup, taken
literally:
- Instances: binary sparse SYK with N = 8 and K = 10 (Eq. 9). K distinct quartets are drawn uniformly
  from C(8,4) = 70, with signs of +/-1 and coefficient J/sqrt(K), J = sqrt(2).
  H_a = i^{q/2} sum J psi psi psi psi (Eq. 1, q = 4). There are 20 realizations (default_rng(7000+r)),
  and the same realizations are used at every beta.
- Encoding: their JW (Eq. 5). psi_L^i = Z^{i-1} X I^{N-i}/sqrt2 and psi_R^i = Z^{i-1} Y I^{N-i}/sqrt2
  on 8 qubits. |I> is the common kernel of psi_L + i psi_R, asserted to equal |0>^8, so it needs no
  prep gate. TFD = e^{-beta (H_L+H_R)/4}|I> (Eq. 10). H_mix = H_int (weight-1 Z's, Eq. 20).
- Algorithm: our committed maqaoa_prune.py, unchanged (Eq. 18, Alg. 1/2, N_C = 10, budgets).
  Initial fits: 5 inits x 3000 at p = 3..8, stopping at F >= 0.9999. Init set A is U(-pi, pi) and
  set B is U(-0.1, 0.1); both are reported. Pruning starts from the set that reaches at the smaller p
  (ties go to A).
- Two-qubit count: sum 2(w-1) over the active cost blocks, i.e. our logical cost minus the 1 we charge
  for |I> prep, which is zero here. This matches their "simplify the mixer only" gate count.

Agreement criteria. Their per-realization std is sigma. Two independent 20-sample means agree if
|ours - theirs| <= 2 sigma sqrt(2/20).
- T1  Table I (ma-QAOA, p = 3, mean best-of-5 F), for each set:
  - beta = 0.1 and 1: our mean >= 0.9999 (theirs: 1.000000 and 0.999999);
  - beta = 10: within 2 * 0.008990 * sqrt(0.1) = 0.0057 of 0.987927.
- T2  beta = 10 reaching p: our range overlaps their 4..7.
- T3  Table V, mean F at a fixed block count S:
  - beta = 0.1, S = 1: SAP and OSAP within 1e-4 of 0.999719;
  - beta = 1, S = 1: SAP within 4.0e-4 of 0.972688 and OSAP within 4.2e-4 of 0.972709;
  - beta = 10, S = 19: SAP within 0.0453 of 0.868664 and OSAP within 0.0128 of 0.957391.
- T4  Table III (beta = 10, the S with F closest to 0.98):
  - mean blocks: SAP 31 +/- 4.4, OSAP 25 +/- 3.2;
  - mean 2q: SAP 292 +/- 43.6, OSAP 229 +/- 38.6.
The verdict is REPRODUCED only if T1 (for at least one init set), T2, T3 and T4 all pass. Otherwise
every miss is listed. Output: workspace/director-preprint-reproduction-2026-09-22.json.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # before numpy: BLAS threads held across fork deadlock the pool

import importlib.util
import itertools
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("maqaoa_prune", HERE / "maqaoa_prune.py")
M = importlib.util.module_from_spec(spec)
sys.modules["maqaoa_prune"] = M
spec.loader.exec_module(M)
S = M.S

OUT = M.WS / "director-preprint-reproduction-2026-09-22.json"
N_REAL, K, J = 20, 10, math.sqrt(2.0)
BETAS = [0.1, 1.0, 10.0]
TABLE_V_S = {0.1: 1, 1.0: 1, 10.0: 19}
TOL = {"T1_b10": 0.0057, "T3": {(0.1, "SAP"): 1e-4, (0.1, "OSAP"): 1e-4, (1.0, "SAP"): 4.0e-4,
                                (1.0, "OSAP"): 4.2e-4, (10.0, "SAP"): 0.0453, (10.0, "OSAP"): 0.0128},
       "T4": {("SAP", "blocks"): 4.4, ("OSAP", "blocks"): 3.2, ("SAP", "2q"): 43.6, ("OSAP", "2q"): 38.6}}
THEIRS = {"T1": {0.1: 1.0, 1.0: 0.999999, 10.0: 0.987927},
          "T3": {(0.1, "SAP"): 0.999719, (0.1, "OSAP"): 0.999719, (1.0, "SAP"): 0.972688,
                 (1.0, "OSAP"): 0.972709, (10.0, "SAP"): 0.868664, (10.0, "OSAP"): 0.957391},
          "T4": {("SAP", "blocks"): 31, ("OSAP", "blocks"): 25, ("SAP", "2q"): 292, ("OSAP", "2q"): 229}}

LAB_L = {i: "Z" * (i - 1) + "X" + "I" * (8 - i) for i in range(1, 9)}
LAB_R = {i: "Z" * (i - 1) + "Y" + "I" * (8 - i) for i in range(1, 9)}


def instance(r: int, beta: float) -> dict:
    rng = np.random.default_rng(7000 + r)
    quartets = list(itertools.combinations(range(1, 9), 4))
    picks = rng.choice(len(quartets), size=K, replace=False)
    signs = rng.choice([-1, 1], size=K)
    coeff = -1.0 * J / math.sqrt(K)                      # i^{q/2} = -1 for q = 4 (Eq. 1)
    cost = []
    for idx, s in zip(picks, signs):
        for side in (LAB_L, LAB_R):
            cost.append(S._term_pauli(side, quartets[int(idx)], coeff * int(s)))
    mix = []
    for j in range(1, 9):
        lab, ph = S._label_mul(LAB_L[j], LAB_R[j])
        mix.append((lab, 1j * 0.5 * ph))
    H = sum(co * S._pauli(lab) for lab, co in cost)
    pL = [S._pauli(LAB_L[j]) / math.sqrt(2) for j in range(1, 9)]
    pR = [S._pauli(LAB_R[j]) / math.sqrt(2) for j in range(1, 9)]
    Mker = np.vstack([a + 1j * b for a, b in zip(pL, pR)])
    w, v = np.linalg.eigh(Mker.conj().T @ Mker)
    ivec = v[:, int(np.argmin(w))]
    zero = np.zeros(256, complex); zero[0] = 1.0
    assert abs(abs(np.vdot(zero, ivec)) - 1) < 1e-10, "|I> is not |0>^8 in their encoding"
    wt, vt = np.linalg.eigh(H)
    tfd = (vt * np.exp(-beta * wt / 4.0)) @ vt.conj().T @ zero
    tfd /= np.linalg.norm(tfd)
    assert all(sum(c != "I" for c in lab) == 1 for lab, _ in mix), "mixer should be weight-1 Z's (Eq. 20)"
    return {"labels": [l for l, _ in cost] + [l for l, _ in mix], "ref": zero, "target": tfd}


def fit_job(job):
    r, beta, set_name = job
    inst = instance(r, beta)
    seed0, width = M.INIT_SETS[set_name]
    rows, best_angles = [], None
    for p in M.P_VALUES:
        tab = M.build_table(inst["labels"], p)
        rng = np.random.default_rng(seed0 + p + 1000 * r)
        mask = np.ones(len(tab.labels), dtype=bool)
        best = (-1.0, None)
        for _ in range(M.N_INITS):
            a, f, _ = M.optimize(rng.uniform(-width, width, len(tab.labels)), mask, tab, inst["ref"],
                                 inst["target"], M.FIT_MAXITER, bounded=False)
            if f > best[0]:
                best = (f, a)
        rows.append({"p": p, "F": best[0]})
        best_angles = best[1]
        if best[0] >= M.F_REACH:
            break
    reached = rows[-1]["p"] if rows[-1]["F"] >= M.F_REACH else None
    return {"r": r, "beta": beta, "set": set_name, "rows": rows, "reached_p": reached,
            "angles": best_angles.tolist()}


def prune_job(job):
    r, beta, algo, p, angles = job
    inst = instance(r, beta)
    tab = M.build_table(inst["labels"], p)
    a = np.array(angles)
    mask = np.ones(len(a), dtype=bool)
    f0 = M.fidelity(tab, a, mask, inst["ref"], inst["target"])
    tr = M.run_trajectory(algo, a, mask, f0, tab, inst["ref"], inst["target"], M.PRUNE_MAXITER)
    rounds = [{"blocks": x.n_active_blocks, "twoq": x.logical_cost - 1, "F": x.F} for x in tr.rounds]
    return {"r": r, "beta": beta, "algo": algo, "p": p, "rounds": rounds}


def _at_blocks(rounds, s):
    hit = [x for x in rounds if x["blocks"] == s]
    return hit[0]["F"] if hit else None


def summarize(fits, prunes) -> dict:
    res, misses = {"T1": {}, "T2": {}, "T3": {}, "T4": {}}, []
    for beta in BETAS:
        for set_name in ("A", "B"):
            f3 = [f["rows"][0]["F"] for f in fits if f["beta"] == beta and f["set"] == set_name]
            res["T1"]["%s/%s" % (beta, set_name)] = float(np.mean(f3))
    t1_ok = []
    for set_name in ("A", "B"):
        ok = (res["T1"]["0.1/%s" % set_name] >= 0.9999 and res["T1"]["1.0/%s" % set_name] >= 0.9999
              and abs(res["T1"]["10.0/%s" % set_name] - THEIRS["T1"][10.0]) <= TOL["T1_b10"])
        t1_ok.append(ok)
    if not any(t1_ok):
        misses.append("T1: neither init set reproduces Table I")
    reach10 = [f["reached_p"] for f in fits if f["beta"] == 10.0 and f["set"] == "B"] + \
              [f["reached_p"] for f in fits if f["beta"] == 10.0 and f["set"] == "A"]
    got = sorted(set(p for p in reach10 if p))
    res["T2"] = {"reached_p_values": got, "unreached": sum(p is None for p in reach10)}
    if not got or max(got) < 4 or min(got) > 7:
        misses.append("T2: beta=10 reaching p %s does not overlap 4..7" % got)
    for (beta, algo), theirs in THEIRS["T3"].items():
        vals = [_at_blocks(x["rounds"], TABLE_V_S[beta]) for x in prunes if x["beta"] == beta and x["algo"] == algo]
        vals = [v for v in vals if v is not None]
        ours = float(np.mean(vals)) if vals else None
        res["T3"]["%s/%s" % (beta, algo)] = {"ours": ours, "theirs": theirs, "n": len(vals)}
        if ours is None or abs(ours - theirs) > TOL["T3"][(beta, algo)]:
            misses.append("T3 beta=%s %s: ours %s vs theirs %.6f" % (beta, algo, ours, theirs))
    for algo in ("SAP", "OSAP"):
        pts = []
        for x in prunes:
            if x["beta"] == 10.0 and x["algo"] == algo:
                pts.append(min(x["rounds"], key=lambda q: abs(q["F"] - 0.98)))
        for key, field in (("blocks", "blocks"), ("2q", "twoq")):
            ours = float(np.mean([q[field] for q in pts]))
            theirs = THEIRS["T4"][(algo, key)]
            res["T4"]["%s/%s" % (algo, key)] = {"ours": ours, "theirs": theirs}
            if abs(ours - theirs) > TOL["T4"][(algo, key)]:
                misses.append("T4 %s %s: ours %.1f vs theirs %d" % (algo, key, ours, theirs))
    res["verdict"] = "REPRODUCED" if not misses else "NOT REPRODUCED"
    res["misses"] = misses
    return res


def main():
    if "qiskit" in sys.modules:
        raise RuntimeError("qiskit imported before fork: refusing to run")
    ctx = mp.get_context("fork")
    fit_jobs = [(r, b, s) for b in BETAS for s in ("A", "B") for r in range(N_REAL)]
    with ctx.Pool(10) as pool:
        fits = pool.map(fit_job, fit_jobs, chunksize=1)
    print("fits done", flush=True)
    prune_jobs = []
    for b in BETAS:
        for r in range(N_REAL):
            fa = next(f for f in fits if f["r"] == r and f["beta"] == b and f["set"] == "A")
            fb = next(f for f in fits if f["r"] == r and f["beta"] == b and f["set"] == "B")
            ok = [f for f in (fa, fb) if f["reached_p"] is not None]
            chosen = min(ok, key=lambda f: (f["reached_p"], f["set"])) if ok else max((fa, fb), key=lambda f: f["rows"][-1]["F"])
            for algo in ("SAP", "OSAP"):
                prune_jobs.append((r, b, algo, chosen["rows"][-1]["p"], chosen["angles"]))
    with ctx.Pool(10) as pool:
        prunes = pool.map(prune_job, prune_jobs, chunksize=1)
    summary = summarize(fits, prunes)
    for f in fits:
        f.pop("angles")
    OUT.write_text(json.dumps({"summary": summary, "fits": fits, "prunes": prunes}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
