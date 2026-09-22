"""Post-hoc robustness probes of the pre-registered NULL (not pre-registered; never moves the verdict).

Run after the 2026-09-22 critic panel asked whether the null is an artifact. Reuses the committed
maqaoa_prune.py functions unchanged. Three probes, all generous to ma-QAOA:
  P1  larger-p start: fit at p = 4 and 5 (init set B, 5 x 3000), then OSAP-prune (N_C = 10, 300 it).
      Best F at LOGICAL cost <= 35 is reported (routing only adds gates, so this is an upper bound
      on what <= 35 routed CZ could reach).
  P2  heavy re-fit: every stored SAP/OSAP structure with logical cost <= 35 is re-fitted from
      50 random restarts x 3000 iterations (vs the pre-registered polish of 6 x 800).
  P3  routing coverage: over all routed points, the minimum routed / logical ratio, to check that no
      unrouted point (logical > 70) could route to <= 35 CZ.
Output: workspace/director-null-probes-2026-09-22.json.  Run: python workspace/null_probes.py
(about 9 minutes on 10 cores), then `python workspace/null_probes.py summarize` for the P2
result restricted to structures that route to <= 35 CZ.
"""

import importlib.util
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("maqaoa_prune", HERE / "maqaoa_prune.py")
M = importlib.util.module_from_spec(spec)
sys.modules["maqaoa_prune"] = M
spec.loader.exec_module(M)

OUT = M.WS / "director-null-probes-2026-09-22.json"
P1_P = [4, 5]
P2_RESTARTS, P2_ITERS = 50, 3000


def _setup_ctx():
    for name in M.INSTANCES:
        inst = M.S.build_instance(name, use_instance_map=True)
        M._CTX[name] = {"labels": M.layer_labels(inst), "ref": inst["I_vec"].astype(complex),
                        "target": inst["target"].astype(complex), "K": inst["K"], "jw": inst["jw_map"]}


def p1_job(job):
    name, p = job
    _, _, tbl = M.fit_set((name, "B", [p], M.N_INITS, M.FIT_MAXITER))
    tab, ref, target = M.instance_ctx(name, p)
    a = np.array(tbl["best_angles"])
    mask = np.ones(len(a), dtype=bool)
    tr = M.run_trajectory("OSAP", a, mask, M.fidelity(tab, a, mask, ref, target), tab, ref, target,
                          M.PRUNE_MAXITER)
    le35 = [r for r in tr.rounds if r.logical_cost <= M.ROUTED_CAP]
    return {"instance": name, "p": p, "fit_F": tbl["rows"][-1]["best_F"],
            "best_F_logical_le35": max((r.F for r in le35), default=None),
            "rounds": len(tr.rounds)}


def p2_job(job):
    name, p, algo, rnd, mask = job
    tab, ref, target = M.instance_ctx(name, p)
    mask = np.array(mask, dtype=bool)
    rng = np.random.default_rng(5000 + rnd)
    best = -1.0
    for _ in range(P2_RESTARTS):
        x0 = rng.uniform(-np.pi, np.pi, len(mask))
        _, f, _ = M.optimize(x0, mask, tab, ref, target, P2_ITERS, bounded=False)
        best = max(best, f)
    return {"instance": name, "algo": algo, "round": rnd, "logical_cost": int(M.logical_cost(tab, mask)),
            "refit_best_F": best}


def p3_ratio():
    pts = json.loads(Path(M.ROUTED_JSON).read_text())["points"]
    ratios = [p["routed_best"] / p["logical_cost"] for p in pts if p["logical_cost"] > 0]
    return {"n_routed": len(pts), "min_routed_over_logical": min(ratios),
            "max_logical_among_routed_le35": max(p["logical_cost"] for p in pts if p["routed_best"] <= 35)}


def main():
    if "qiskit" in sys.modules:
        raise RuntimeError("qiskit imported before fork: refusing to run")
    _setup_ctx()
    tf = M.load_traj(M.TRAJ_JSON)
    p2_jobs = []
    for inst in tf.instances:
        for tr in inst.trajectories:
            for pt in tr.points:
                if pt.logical_cost <= M.ROUTED_CAP:
                    p2_jobs.append((inst.name, inst.chosen_p, tr.algo, pt.round, list(pt.mask)))
    p1_jobs = [(n, p) for n in M.INSTANCES for p in P1_P]
    with mp.get_context("fork").Pool(10) as pool:
        p2 = pool.map(p2_job, p2_jobs, chunksize=1)
        p1 = pool.map(p1_job, p1_jobs, chunksize=1)
    f35 = {i["name"]: i["f35_best50"] for i in json.loads(Path(M.DECISION_JSON).read_text())["instances"]}
    summary = {}
    for n in M.INSTANCES:
        best_p1 = max((r["best_F_logical_le35"] or 0) for r in p1 if r["instance"] == n)
        best_p2 = max((r["refit_best_F"] for r in p2 if r["instance"] == n), default=None)
        summary[n] = {"f35_star": f35[n], "p1_best_F_logical_le35": best_p1, "p2_best_refit_F": best_p2,
                      "overturns": max(best_p1, best_p2 or 0) >= f35[n]}
    out = {"label": "POST-HOC robustness probes; not pre-registered; the verdict stays the pre-registered one",
           "p1": p1, "p2": p2, "p3": p3_ratio(), "summary": summary}
    OUT.write_text(json.dumps(out, indent=1))
    for n, s in summary.items():
        print("%s F35*=%.4f  P1(p=4,5 OSAP, logical<=35)=%.4f  P2(50x3000 refit)=%s  overturns=%s"
              % (n, s["f35_star"], s["p1_best_F_logical_le35"],
                 "%.4f" % s["p2_best_refit_F"] if s["p2_best_refit_F"] else "n/a", s["overturns"]))
    print("P3:", out["p3"])


def summarize_routed():
    """P2 restricted to the pre-registered budget: structures whose ROUTED cost is <= 35 CZ.
    (P2 itself filtered on logical cost, which admits structures that route above 35.)"""
    d = json.loads(OUT.read_text())
    routed = {(p["instance"], p["algo"], p["round"]): p["routed_best"]
              for p in json.loads(Path(M.ROUTED_JSON).read_text())["points"]}
    out = {}
    for n, s in d["summary"].items():
        ok = [r for r in d["p2"] if r["instance"] == n and routed.get((n, r["algo"], r["round"]), 10**9) <= 35]
        b = max(ok, key=lambda r: r["refit_best_F"])
        out[n] = {"f35_star": s["f35_star"], "p2_best_refit_F_routed_le35": b["refit_best_F"], "algo": b["algo"],
                  "round": b["round"], "routed": routed[(n, b["algo"], b["round"])],
                  "gap": s["f35_star"] - b["refit_best_F"], "null_margin_holds": s["f35_star"] - b["refit_best_F"] > M.NULL_MARGIN}
        print("%s gap at <=35 routed CZ after 50x3000 refit: %.4f" % (n, out[n]["gap"]))
    d["p2_routed_le35_summary"] = out
    OUT.write_text(json.dumps(d, indent=1))


if __name__ == "__main__":
    summarize_routed() if sys.argv[1:] == ["summarize"] else main()
