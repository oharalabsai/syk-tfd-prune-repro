"""One-command reproduction of the syk-tfd-prune result (called by the package's reproduce.py).

Recomputes, from the shipped code and the five public certified instances, everything the paper
reports, and exits 0 only if all of it reproduces:
  1. the exact beta = 3 TFD targets (default and instance JW maps) match the sealed npz (1e-9);
  2. the verified |0>^8 -> |I> prep has fidelity 1 in both maps;
  3. the 35-CNOT anchors F35 (best-of-50, both maps) re-simulate from their stored angles (1e-9);
  4. every stored ma-QAOA SAP/OSAP point re-simulates through the sealed scorer (1e-8, exact cx);
  5. the best F at <= 35 routed CZ, the crossings and the pre-registered NULL verdict match the
     committed decision and crossing files;
  6. the 35-CNOT block routes to A35 = 35 CZ on FakeMarrakesh (opt 3, best of the 11 seeds).
The full optimization itself reruns with `python workspace/maqaoa_prune.py run|route|verify|decide`
(about 10 minutes on 10 cores) and `python scorekeeper/seal_targets.py f35 <out>` for the anchors.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load_maqaoa():
    spec = importlib.util.spec_from_file_location("maqaoa_prune", HERE / "maqaoa_prune.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["maqaoa_prune"] = mod      # pydantic resolves the models' annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


M = _load_maqaoa()          # sets up the repo or package paths and imports seal_targets as M.S
S = M.S
NAMES = M.INSTANCES


def _check(ok: bool, msg: str) -> bool:
    print("%s  %s" % ("OK  " if ok else "FAIL", msg))
    return ok


def _block_gates(theta) -> list:
    """The 35-CNOT block with explicit angles, in the seal's order (8 H, rz then rx, cx chains)."""
    gates = [{"name": "h", "qubits": [q], "param": 0.0} for q in range(8)]
    for layer in range(6):
        for q in range(8):
            gates += [{"name": "rz", "qubits": [q], "param": float(theta[layer, q, 0])},
                      {"name": "rx", "qubits": [q], "param": float(theta[layer, q, 1])}]
        if layer < 5:
            gates += [{"name": "cx", "qubits": [q, q + 1], "param": 0.0} for q in range(7)]
    return gates


def check_targets(npz) -> bool:
    ok = True
    for n in NAMES:
        for imap, key in ((False, "%s/target" % n), (True, "%s/target_imap" % n)):
            err = float(np.max(np.abs(S.build_instance(n, imap)["target"] - npz[key])))
            ok &= _check(err <= 1e-9, "%s %s target rebuilt, max |diff| %.1e" % (n, "imap" if imap else "dmap", err))
    return ok


def check_iprep() -> bool:
    prep = S.i_prep_gates()[0]
    ok = True
    for n in NAMES:
        f = abs(np.vdot(S.build_instance(n, True)["I_vec"], S._sim_prep(prep))) ** 2
        ok &= _check(abs(f - 1) <= 1e-9, "%s |I> prep fidelity %.12f" % (n, f))
    return ok


def check_anchors(npz, sc) -> bool:
    ok = True
    for n in NAMES:
        for tag in ("dmap", "imap"):
            gates = _block_gates(npz["%s/f35_best50_%s_angles" % (n, tag)])
            target = npz["%s/target" % n] if tag == "dmap" else npz["%s/target_imap" % n]
            f, cost, _ = M.scorer_check(sc, gates, target)
            ref = float(npz["%s/f35_best50_%s" % (n, tag)])
            ok &= _check(abs(f - ref) <= 1e-9 and cost == 35,
                         "%s F35* %s best-of-50 = %.6f (stored %.6f, cx %d)" % (n, tag, f, ref, cost))
    return ok


def check_points() -> bool:
    M.cmd_verify(argparse.Namespace(traj=str(M.TRAJ_JSON)))   # raises on any mismatch
    return _check(True, "all stored SAP/OSAP points re-simulated through the sealed scorer")


def check_decision() -> bool:
    routed = json.loads(Path(M.ROUTED_JSON).read_text())
    dec = json.loads(Path(M.DECISION_JSON).read_text())
    cross = json.loads((M.WS / "director-maqaoa-crossing-2026-09-22.json").read_text())
    traj = json.loads(Path(M.TRAJ_JSON).read_text())
    ok = True
    verdicts = []
    for inst in dec["instances"]:
        n, f35 = inst["name"], inst["f35_best50"]
        inst_null = True
        for a in inst["algos"]:
            pts = [p for p in routed["points"] if p["instance"] == n and p["algo"] == a["algo"]
                   and p["routed_best"] <= M.ROUTED_CAP]
            best = max(p["F"] for p in pts)
            ok &= _check(abs(best - a["best_exact_F"]) <= 1e-12,
                         "%s %s best F at <=35 routed CZ = %.6f" % (n, a["algo"], best))
            inst_null &= max([best] + a["polish_all_F"]) < f35 - M.NULL_MARGIN
        verdicts.append("NULL" if inst_null else "NOT-NULL")
        ok &= _check(("NULL" if inst_null else "NOT-NULL") == inst["verdict"], "%s verdict %s" % (n, inst["verdict"]))
    for inst in traj["instances"]:
        n = inst["name"]
        for tr in inst["trajectories"]:
            hits = [r["logical_cost"] for r in tr["rounds"] if r["F"] >= cross[n]["f35_star"]]
            got = min(hits) if hits else None
            ok &= _check(got == cross[n][tr["algo"]]["cross_logical"],
                         "%s %s logical crossing to F35* = %s" % (n, tr["algo"], got))
    camp = "NULL" if all(v == "NULL" for v in verdicts) else "NOT-NULL"
    ok &= _check(camp == "NULL" and dec["campaign"].startswith("NULL"), "campaign verdict %s" % dec["campaign"])
    return ok


def check_a35() -> bool:
    from qiskit import transpile
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    qc, be = M.to_qiskit(M.block35_gates()), FakeMarrakesh()
    counts = [sum(1 for i in transpile(qc, backend=be, optimization_level=3, seed_transpiler=s).data
                  if i.operation.num_qubits == 2) for s in M.ROUTE_SEEDS]
    return _check(min(counts) == 35, "A35 = %d routed CZ (per seed %s)" % (min(counts), counts))


def main() -> int:
    npz = dict(np.load(M.SK / "tfd_targets.npz", allow_pickle=False))
    sc = M.load_scorer()
    ok = check_targets(npz) & check_iprep() & check_anchors(npz, sc)
    ok &= check_points() & check_decision() & check_a35()
    print("\nREPRODUCED" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
