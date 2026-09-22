"""ma-QAOA + SAP/OSAP pruning on the certified syk-tfd-prune instances (Director computation).

Executes PREREG-maqaoa-2026-09-22.md: the literal preprint Eq. 18 ansatz from |I> in each
instance's own JW map, the pre-registered initial fit (init sets A and B), SAP (Alg. 1) and
OSAP (Alg. 2, N_C = 10) pruning trajectories, scorer-format gate emission, FakeMarrakesh
routing, independent verification through the sealed scorer, and the WIN/NULL/PARTIAL rule.

Subcommands (run from the repo root):
  run        initial fit + SAP/OSAP trajectories (fork pool, never imports qiskit)
  route      FakeMarrakesh best-of-11 routing of every stored point + A35 (spawn pool)
  verify     scorer re-simulation of every stored point + qiskit statevector check
  decide F35_JSON   post-hoc polish + decision rule
  gradcheck  central-difference gradient check only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel
from scipy.optimize import minimize

_HERE = Path(__file__).resolve().parent
if (_HERE.parent / "scorekeeper").is_dir():      # reproduction-package layout: <pkg>/{workspace,scorekeeper}
    ROOT, WS, SK = _HERE.parent, _HERE, _HERE.parent / "scorekeeper"
else:                                            # autoresearch repo layout
    ROOT = _HERE.parents[2]
    WS, SK = ROOT / "campaigns/syk-tfd-prune/workspace", ROOT / "scorekeeper_private/syk-tfd-prune"
TRAJ_JSON = WS / "director-maqaoa-trajectories-2026-09-22.json"
ROUTED_JSON = WS / "director-maqaoa-routed-2026-09-22.json"
DECISION_JSON = WS / "director-maqaoa-decision-2026-09-22.json"

INSTANCES = ["INST-A", "INST-B", "INST-C", "INST-D", "INST-E"]
NQ, DIM = 8, 256
P_VALUES = [3, 4, 5, 6, 7, 8]
N_INITS = 5
FIT_MAXITER = 3000
PRUNE_MAXITER = 300
F_REACH = 0.9999
N_C = 10
STORE_COST = 70
ROUTE_SEEDS = [7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
ROUTED_CAP = 35
POLISH_INITS = 6
POLISH_MAXITER = 800
NULL_MARGIN = 0.02
LBFGS = {"ftol": 0.0, "gtol": 1e-10}
INIT_SETS = {"A": (100, math.pi), "B": (200, 0.1)}  # seed offset, half-width of U(-w, w)

sys.path.insert(0, str(SK))
os.chdir(ROOT)
import seal_targets as S  # noqa: E402


def load_scorer():
    spec = importlib.util.spec_from_file_location("syk_tfd_scorer", SK / "scorer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- records ----------------------------------------------------------------------------


class ReachRow(BaseModel):
    p: int
    best_F: float
    all_F: List[float]
    nit: List[int]


class ReachTable(BaseModel):
    init_set: str
    rows: List[ReachRow]
    reached_p: Optional[int]
    best_angles: Optional[List[float]] = None


class RoundRecord(BaseModel):
    round: int
    n_active_blocks: int
    logical_cost: int
    F: float
    removed: Optional[int] = None


class StoredPoint(BaseModel):
    round: int
    logical_cost: int
    F: float
    angles: List[float]
    mask: List[bool]


class Trajectory(BaseModel):
    algo: str
    rounds: List[RoundRecord]
    points: List[StoredPoint]
    wall_s: float


class InstanceResult(BaseModel):
    name: str
    K: int
    jw_map: List[int]
    labels_per_layer: List[str]
    gradcheck_max_err: float
    reach: Dict[str, ReachTable]
    chosen_set: Optional[str]
    chosen_p: Optional[int]
    killed: bool
    trajectories: List[Trajectory]


class TrajectoryFile(BaseModel):
    prereg: str = "PREREG-maqaoa-2026-09-22.md"
    settings: Dict[str, object]
    instances: List[InstanceResult]


class RoutedPoint(BaseModel):
    instance: str
    algo: str
    round: int
    logical_cost: int
    F: float
    routed_best: int
    per_seed: List[int]


class RoutedFile(BaseModel):
    backend: str = "FakeMarrakesh"
    optimization_level: int = 3
    seeds: List[int] = ROUTE_SEEDS
    a35_best: int
    a35_per_seed: List[int]
    points: List[RoutedPoint]


class AlgoDecision(BaseModel):
    algo: str
    best_exact_F: Optional[float]
    best_exact_round: Optional[int]
    best_exact_routed: Optional[int]
    polish_F: Optional[float]
    polish_all_F: List[float]


class InstanceDecision(BaseModel):
    name: str
    f35_best50: float
    f35_best6: float
    algos: List[AlgoDecision]
    verdict: str


class DecisionFile(BaseModel):
    a35_best: int
    instances: List[InstanceDecision]
    campaign: str


# ---- fast Pauli simulator ----------------------------------------------------------------


class PauliTable(BaseModel, arbitrary_types_allowed=True):
    """P|b> = phase[b] |b ^ flip>; stored as (P psi) = ph * psi[perm]."""
    labels: List[str]
    weights: List[int]
    perm: np.ndarray
    ph: np.ndarray


def pauli_action(label: str) -> Tuple[np.ndarray, np.ndarray]:
    b = np.arange(DIM)
    flip = 0
    phase = np.ones(DIM, dtype=complex)
    for q, c in enumerate(label):
        bit = (b >> (NQ - 1 - q)) & 1
        if c in "XY":
            flip |= 1 << (NQ - 1 - q)
        if c == "Y":
            phase *= np.where(bit == 0, 1j, -1j)
        elif c == "Z":
            phase *= np.where(bit == 0, 1.0, -1.0)
    perm = b ^ flip
    return perm, phase[perm]  # (P psi)[b'] = phase(b' ^ flip) psi[b' ^ flip]


def layer_labels(inst: dict) -> List[str]:
    return [lab for lab, _ in inst["cost_paulis"]] + [lab for lab, _ in inst["mix_paulis"]]


def build_table(labels: List[str], p: int) -> PauliTable:
    acts = [pauli_action(lab) for lab in labels]
    full = labels * p
    return PauliTable(labels=full, weights=[sum(c != "I" for c in lab) for lab in full],
                      perm=np.array([a[0] for a in acts] * p),
                      ph=np.array([a[1] for a in acts] * p))


def evolve(psi: np.ndarray, tab: PauliTable, k: int, t: float) -> np.ndarray:
    return math.cos(t) * psi - 1j * math.sin(t) * (tab.ph[k] * psi[tab.perm[k]])


def state(tab: PauliTable, angles: np.ndarray, mask: np.ndarray, ref: np.ndarray) -> np.ndarray:
    psi = ref
    for k in np.flatnonzero(mask):
        psi = evolve(psi, tab, k, angles[k])
    return psi


def fidelity(tab, angles, mask, ref, target) -> float:
    return float(abs(np.vdot(target, state(tab, angles, mask, ref))) ** 2)


def loss_and_grad(x: np.ndarray, idx: np.ndarray, tab: PauliTable, ref, target):
    """L = 1 - |<T|psi>|^2 and its adjoint gradient over the active angles x (at indices idx)."""
    phi = ref
    for k, t in zip(idx, x):
        phi = evolve(phi, tab, k, t)
    amp = np.vdot(target, phi)
    lam = target.copy()
    grad = np.empty(len(idx))
    for j in range(len(idx) - 1, -1, -1):
        k, t = idx[j], x[j]
        grad[j] = -2.0 * (np.conj(amp) * (-1j) * np.vdot(lam, tab.ph[k] * phi[tab.perm[k]])).real
        phi = evolve(phi, tab, k, -t)
        lam = evolve(lam, tab, k, -t)
    return 1.0 - float(abs(amp) ** 2), grad


def gradcheck(tab: PauliTable, ref, target, seed: int = 7, h: float = 1e-6) -> float:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(tab.labels))
    x = rng.uniform(-math.pi, math.pi, len(idx))
    _, g = loss_and_grad(x, idx, tab, ref, target)
    num = np.empty_like(g)
    for j in range(len(x)):
        e = np.zeros_like(x)
        e[j] = h
        num[j] = (loss_and_grad(x + e, idx, tab, ref, target)[0]
                  - loss_and_grad(x - e, idx, tab, ref, target)[0]) / (2 * h)
    return float(np.max(np.abs(num - g)))


def optimize(angles, mask, tab, ref, target, maxiter: int, bounded: bool):
    idx = np.flatnonzero(mask)
    x0 = angles[idx]
    bounds = [(t - math.pi, t + math.pi) for t in x0] if bounded else None
    res = minimize(loss_and_grad, x0, args=(idx, tab, ref, target), jac=True, method="L-BFGS-B",
                   bounds=bounds, options={"maxiter": maxiter, **LBFGS})
    out = np.zeros_like(angles)
    out[idx] = res.x
    return out, 1.0 - float(res.fun), int(res.nit)


# ---- cost and pruning --------------------------------------------------------------------


def logical_cost(tab: PauliTable, mask: np.ndarray) -> int:
    return 1 + sum(2 * (tab.weights[k] - 1) for k in np.flatnonzero(mask))


def nonlocal_active(tab: PauliTable, mask: np.ndarray) -> np.ndarray:
    return np.array([k for k in np.flatnonzero(mask) if tab.weights[k] >= 2], dtype=int)


def wrap(t: np.ndarray) -> np.ndarray:
    """Wrap to (-pi, pi]."""
    return math.pi - np.mod(math.pi - t, 2 * math.pi)


def smallest_blocks(tab, angles, mask, n: int) -> np.ndarray:
    act = nonlocal_active(tab, mask)
    order = np.argsort(np.abs(wrap(angles[act])), kind="stable")
    return act[order[:n]]


def remove_and_reopt(k, angles, mask, tab, ref, target, maxiter):
    m = mask.copy()
    m[k] = False
    a = angles.copy()
    a[k] = 0.0
    a, f, _ = optimize(a, m, tab, ref, target, maxiter, bounded=True)
    return a, m, f


def sap_step(angles, mask, tab, ref, target, maxiter):
    k = int(smallest_blocks(tab, angles, mask, 1)[0])
    return (k, *remove_and_reopt(k, angles, mask, tab, ref, target, maxiter))


def osap_step(angles, mask, tab, ref, target, maxiter):
    best = None
    for k in smallest_blocks(tab, angles, mask, N_C):
        a, m, f = remove_and_reopt(int(k), angles, mask, tab, ref, target, maxiter)
        if best is None or 1.0 - f < 1.0 - best[3]:
            best = (int(k), a, m, f)
    return best


def is_stored(rnd: int, cost: int, final: bool) -> bool:
    return cost <= STORE_COST or rnd % 10 == 0 or final


def run_trajectory(algo, angles, mask, f0, tab, ref, target, maxiter, max_rounds=None) -> Trajectory:
    t0 = time.time()
    step = sap_step if algo == "SAP" else osap_step
    rounds, points = [], []

    def record(rnd, a, m, f, removed, final):
        cost = logical_cost(tab, m)
        rounds.append(RoundRecord(round=rnd, n_active_blocks=len(nonlocal_active(tab, m)),
                                  logical_cost=cost, F=f, removed=removed))
        if is_stored(rnd, cost, final):
            points.append(StoredPoint(round=rnd, logical_cost=cost, F=f,
                                      angles=a.tolist(), mask=m.tolist()))

    record(0, angles, mask, f0, None, False)
    rnd = 0
    while len(nonlocal_active(tab, mask)) > 0 and (max_rounds is None or rnd < max_rounds):
        rnd += 1
        k, angles, mask, _ = step(angles, mask, tab, ref, target, maxiter)
        f = fidelity(tab, angles, mask, ref, target)
        final = len(nonlocal_active(tab, mask)) == 0 or rnd == max_rounds
        record(rnd, angles, mask, f, k, final)
    return Trajectory(algo=algo, rounds=rounds, points=points, wall_s=time.time() - t0)


# ---- gate emission -----------------------------------------------------------------------

BASIS_IN = {"X": [("h", 0.0)], "Y": [("sdg", 0.0), ("h", 0.0)], "Z": []}
BASIS_OUT = {"X": [("h", 0.0)], "Y": [("h", 0.0), ("s", 0.0)], "Z": []}
SINGLE = {"X": "rx", "Y": "ry", "Z": "rz"}


def evolution_gates(label: str, t: float) -> List[dict]:
    """exp(-i t P): basis change, ascending CNOT ladder, rz(2t) on the last support qubit, undo."""
    sup = [q for q, c in enumerate(label) if c != "I"]
    g = lambda n, qs, p=0.0: {"name": n, "qubits": qs, "param": float(p)}
    if len(sup) == 1:
        return [g(SINGLE[label[sup[0]]], [sup[0]], 2 * t)]
    pre = [g(n, [q], p) for q in sup for n, p in BASIS_IN[label[q]]]
    post = [g(n, [q], p) for q in sup for n, p in BASIS_OUT[label[q]]]
    ladder = [g("cx", [a, b]) for a, b in zip(sup[:-1], sup[1:])]
    return pre + ladder + [g("rz", [sup[-1]], 2 * t)] + ladder[::-1] + post


def emit(tab: PauliTable, angles, mask) -> List[dict]:
    gates = [dict(x) for x in S.i_prep_gates()[0]]
    for k in np.flatnonzero(mask):
        gates += evolution_gates(tab.labels[k], float(angles[k]))
    return gates


def block35_gates(seed: int = 0) -> List[dict]:
    """The 35-CNOT reference block: 8 H, 6 rz/rx layers, linear cx chain after the first 5."""
    th = np.random.default_rng(seed).uniform(-math.pi, math.pi, 96)
    gates = [{"name": "h", "qubits": [q], "param": 0.0} for q in range(NQ)]
    i = 0
    for layer in range(6):
        for q in range(NQ):
            gates += [{"name": "rz", "qubits": [q], "param": float(th[i])},
                      {"name": "rx", "qubits": [q], "param": float(th[i + 1])}]
            i += 2
        if layer < 5:
            gates += [{"name": "cx", "qubits": [q, q + 1], "param": 0.0} for q in range(NQ - 1)]
    return gates


# ---- run (fork pool; qiskit is never imported here) -------------------------------------

_CTX: Dict[str, dict] = {}


def instance_ctx(name: str, p: int) -> Tuple[PauliTable, np.ndarray, np.ndarray]:
    inst = _CTX[name]
    return build_table(inst["labels"], p), inst["ref"], inst["target"]


def fit_set(job) -> Tuple[str, str, dict]:
    name, set_name, p_values, n_inits, maxiter = job
    seed0, width = INIT_SETS[set_name]
    rows, reached, best_angles = [], None, None
    for p in p_values:
        tab, ref, target = instance_ctx(name, p)
        rng = np.random.default_rng(seed0 + p)
        mask = np.ones(len(tab.labels), dtype=bool)
        fs, nits, best = [], [], (-1.0, None)
        for _ in range(n_inits):
            x0 = rng.uniform(-width, width, len(tab.labels))
            a, f, nit = optimize(x0, mask, tab, ref, target, maxiter, bounded=False)
            fs.append(f)
            nits.append(nit)
            if f > best[0]:
                best = (f, a)
        rows.append(ReachRow(p=p, best_F=best[0], all_F=fs, nit=nits))
        best_angles = best[1].tolist()
        if best[0] >= F_REACH:
            reached = p
            break
    table = ReachTable(init_set=set_name, rows=rows, reached_p=reached, best_angles=best_angles)
    return name, set_name, table.model_dump()


def choose_set(reach: Dict[str, ReachTable]) -> Optional[str]:
    """Smaller reaching p wins; a tie goes to A; None if neither reaches 0.9999."""
    ok = [s for s in ("A", "B") if reach[s].reached_p is not None]
    return min(ok, key=lambda s: (reach[s].reached_p, s)) if ok else None


def prune_job(job) -> Tuple[str, dict]:
    name, algo, p, angles, maxiter, max_rounds = job
    tab, ref, target = instance_ctx(name, p)
    a = np.array(angles)
    mask = np.ones(len(a), dtype=bool)
    f0 = fidelity(tab, a, mask, ref, target)
    return name, run_trajectory(algo, a, mask, f0, tab, ref, target, maxiter, max_rounds).model_dump()


def cmd_run(args) -> None:
    if "qiskit" in sys.modules:
        raise RuntimeError("qiskit imported before fork: refusing to run")
    names = args.instances
    grad_err = {}
    for name in names:
        inst = S.build_instance(name, use_instance_map=True)
        _CTX[name] = {"labels": layer_labels(inst), "ref": inst["I_vec"].astype(complex),
                      "target": inst["target"].astype(complex), "K": inst["K"], "jw": inst["jw_map"]}
        tab, ref, target = instance_ctx(name, 3)
        grad_err[name] = gradcheck(tab, ref, target)
        print("gradcheck %s: max |analytic - central| = %.2e" % (name, grad_err[name]), flush=True)
        if grad_err[name] >= 1e-6:
            raise RuntimeError("gradient check failed on %s" % name)
    ctx = mp.get_context("fork")
    fit_jobs = [(n, s, args.p_values, args.inits, args.maxiter) for n in names for s in ("A", "B")]
    with ctx.Pool(min(10, len(fit_jobs))) as pool:
        fits = pool.map(fit_set, fit_jobs, chunksize=1)
    reach = {n: {} for n in names}
    for n, s, tbl in fits:
        reach[n][s] = ReachTable(**tbl)
        print("%s set %s: %s reached_p=%s" % (n, s, [round(r.best_F, 6) for r in reach[n][s].rows],
                                              reach[n][s].reached_p), flush=True)
    chosen = {n: choose_set(reach[n]) for n in names}
    if args.force_prune:
        chosen = {n: chosen[n] or "A" for n in names}
    prune_jobs = []
    for n in names:
        if chosen[n] is None:
            continue
        tbl = reach[n][chosen[n]]
        for algo in args.algos:
            prune_jobs.append((n, algo, tbl.rows[-1].p, tbl.best_angles, args.prune_maxiter, args.max_rounds))
    trajs = {n: [] for n in names}
    if prune_jobs:
        with ctx.Pool(min(10, len(prune_jobs))) as pool:
            for n, tr in pool.imap_unordered(prune_job, prune_jobs):
                trajs[n].append(Trajectory(**tr))
                print("%s %s done: %d rounds, %.0fs" % (n, tr["algo"], len(tr["rounds"]), tr["wall_s"]), flush=True)
    out = TrajectoryFile(
        settings={"p_values": args.p_values, "inits": args.inits, "maxiter": args.maxiter,
                  "prune_maxiter": args.prune_maxiter, "max_rounds": args.max_rounds,
                  "algos": args.algos, "force_prune": args.force_prune, "N_C": N_C},
        instances=[InstanceResult(
            name=n, K=_CTX[n]["K"], jw_map=list(_CTX[n]["jw"]), labels_per_layer=_CTX[n]["labels"],
            gradcheck_max_err=grad_err[n], reach=reach[n], chosen_set=chosen[n],
            chosen_p=reach[n][chosen[n]].rows[-1].p if chosen[n] else None,
            killed=chosen[n] is None, trajectories=sorted(trajs[n], key=lambda t: t.algo))
            for n in names])
    Path(args.out).write_text(out.model_dump_json(indent=1))
    print("wrote %s" % args.out)


# ---- shared helpers for route / verify / decide -----------------------------------------


def load_traj(path) -> TrajectoryFile:
    return TrajectoryFile(**json.loads(Path(path).read_text()))


def iter_points(tf: TrajectoryFile):
    for inst in tf.instances:
        tab = build_table(inst.labels_per_layer, inst.chosen_p) if inst.chosen_p else None
        for tr in inst.trajectories:
            for pt in tr.points:
                yield inst, tab, tr.algo, pt


def point_gates(tab: PauliTable, pt: StoredPoint) -> List[dict]:
    return emit(tab, np.array(pt.angles), np.array(pt.mask))


def to_qiskit(gates: List[dict]):
    """Scorer qubit q maps to qiskit qubit 7 - q, so both statevectors share one index order."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(NQ)
    for g in gates:
        qs = [NQ - 1 - q for q in g["qubits"]]
        n = g["name"]
        if n in ("rx", "ry", "rz"):
            getattr(qc, n)(g["param"], qs[0])
        else:
            getattr(qc, n)(*qs)
    return qc


# ---- route (spawn pool; each worker imports qiskit) --------------------------------------


def route_one(job) -> Tuple[int, int, int]:
    key, gates, seed = job
    from qiskit import transpile
    from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
    tq = transpile(to_qiskit(gates), backend=FakeMarrakesh(), optimization_level=3, seed_transpiler=seed)
    return key, seed, sum(1 for ins in tq.data if ins.operation.num_qubits == 2)


def cmd_route(args) -> None:
    tf = load_traj(args.traj)
    pts = list(iter_points(tf))
    circuits = [point_gates(tab, pt) for _, tab, _, pt in pts] + [block35_gates()]
    jobs = [(i, c, s) for i, c in enumerate(circuits) for s in ROUTE_SEEDS]
    counts: Dict[int, Dict[int, int]] = {i: {} for i in range(len(circuits))}
    with mp.get_context("spawn").Pool(args.procs) as pool:
        for i, s, n2 in pool.imap_unordered(route_one, jobs, chunksize=1):
            counts[i][s] = n2
    seq = lambda i: [counts[i][s] for s in ROUTE_SEEDS]
    a35 = seq(len(circuits) - 1)
    out = RoutedFile(a35_best=min(a35), a35_per_seed=a35, points=[
        RoutedPoint(instance=inst.name, algo=algo, round=pt.round, logical_cost=pt.logical_cost,
                    F=pt.F, routed_best=min(seq(i)), per_seed=seq(i))
        for i, (inst, _, algo, pt) in enumerate(pts)])
    Path(args.out).write_text(out.model_dump_json(indent=1))
    print("A35 = %d (per seed %s); routed %d points -> %s" % (out.a35_best, a35, len(pts), args.out))


# ---- verify ------------------------------------------------------------------------------


def scorer_check(sc, gates: List[dict], target: np.ndarray) -> Tuple[float, int, np.ndarray]:
    circ = sc.Circuit(n_qubits=NQ, gates=[sc.Gate(**g) for g in gates])
    reason, cost = sc._validate(circ)
    if reason is not None and reason.startswith("gate count"):
        cost = sum(sc.TWO_Q_COST.get(g.name, 0) for g in circ.gates)  # _validate stops before counting
    psi = sc._simulate(circ)
    return float(abs(np.vdot(target, psi)) ** 2), int(cost), psi


def qiskit_check(gates: List[dict], psi_scorer: np.ndarray) -> float:
    """Proves the q -> 7 - q map on a basis state, then compares the full circuit up to global phase."""
    from qiskit.quantum_info import Statevector
    probe = Statevector.from_instruction(to_qiskit([{"name": "x", "qubits": [0], "param": 0.0}]))
    if abs(probe.data[128] - 1) > 1e-12:
        raise RuntimeError("qubit map broken: scorer x(0) is not index 128 in qiskit")
    sv = Statevector.from_instruction(to_qiskit(gates)).data
    return float(1.0 - abs(np.vdot(sv, psi_scorer)))


def cmd_verify(args) -> None:
    sc = load_scorer()
    tf = load_traj(args.traj)
    targets = {inst.name: S.build_instance(inst.name, True)["target"] for inst in tf.instances}
    worst_f, n, qk_done = 0.0, 0, False
    for inst, tab, algo, pt in iter_points(tf):
        gates = point_gates(tab, pt)
        f, cost, psi = scorer_check(sc, gates, targets[inst.name])
        df = abs(f - pt.F)
        worst_f = max(worst_f, df)
        if df > 1e-8 or cost != pt.logical_cost:
            raise RuntimeError("MISMATCH %s %s round %d: F %.12f vs %.12f, cost %d vs %d" % (
                inst.name, algo, pt.round, f, pt.F, cost, pt.logical_cost))
        if not qk_done:
            err = qiskit_check(gates, psi)
            if err > 1e-8:
                raise RuntimeError("qiskit statevector mismatch %.3e" % err)
            print("qiskit statevector check (%s %s round %d): 1-|<q|s>| = %.2e" % (inst.name, algo, pt.round, err))
            qk_done = True
        n += 1
    print("verified %d points: max |dF| = %.2e, all cx counts equal logical cost" % (n, worst_f))


# ---- decide ------------------------------------------------------------------------------


def polish(tab, mask, rnd, ref, target) -> List[float]:
    rng = np.random.default_rng(1000 + rnd)
    fs = []
    for _ in range(POLISH_INITS):
        x0 = np.where(mask, rng.uniform(-math.pi, math.pi, len(mask)), 0.0)
        fs.append(optimize(x0, mask, tab, ref, target, POLISH_MAXITER, bounded=False)[1])
    return fs


def verdict(f35: float, algos: List[AlgoDecision], exact_all: List[float]) -> str:
    if any(a.best_exact_F is not None and a.best_exact_F >= f35 for a in algos):
        return "WIN"
    polish_all = [a.polish_F for a in algos if a.polish_F is not None]
    if all(f < f35 - NULL_MARGIN for f in exact_all + polish_all):
        return "NULL"
    return "PARTIAL"


def campaign_verdict(decs: Dict[str, str]) -> str:
    if any(v == "WIN" for v in decs.values()):
        return "WIN"
    if decs.get("INST-A") == "UNREACHABLE":
        return "CERTIFIED-NEGATIVE (Row 42 reachability kill)"
    if len(decs) == len(INSTANCES) and all(v == "NULL" for v in decs.values()):
        return "NULL (QUESTION-STATUS)"
    return "PARTIAL"


def cmd_decide(args) -> None:
    sc = load_scorer()
    f35 = json.loads(Path(args.f35_json).read_text())
    tf = load_traj(args.traj)
    routed = RoutedFile(**json.loads(Path(args.routed).read_text()))
    rmap = {(r.instance, r.algo, r.round): r.routed_best for r in routed.points}
    decs = []
    for inst in tf.instances:
        fa, fb = float(f35[inst.name]["f35_best50"]), float(f35[inst.name]["f35_best6"])
        if inst.killed:
            decs.append(InstanceDecision(name=inst.name, f35_best50=fa, f35_best6=fb, algos=[], verdict="UNREACHABLE"))
            continue
        tab = build_table(inst.labels_per_layer, inst.chosen_p)
        ref = S.build_instance(inst.name, True)
        target, iv = ref["target"].astype(complex), ref["I_vec"].astype(complex)
        algos, exact_all = [], []
        for tr in inst.trajectories:
            ok = [pt for pt in tr.points if rmap[(inst.name, tr.algo, pt.round)] <= ROUTED_CAP]
            exact_all += [pt.F for pt in ok]
            if not ok:
                algos.append(AlgoDecision(algo=tr.algo, best_exact_F=None, best_exact_round=None,
                                          best_exact_routed=None, polish_F=None, polish_all_F=[]))
                continue
            best = max(ok, key=lambda pt: pt.F)
            mask = np.array(best.mask)
            pf = polish(tab, mask, best.round, iv, target)
            sf = scorer_check(sc, point_gates(tab, best), target)[0]
            if abs(sf - best.F) > 1e-8:
                raise RuntimeError("scorer mismatch on decision point %s %s" % (inst.name, tr.algo))
            algos.append(AlgoDecision(algo=tr.algo, best_exact_F=best.F, best_exact_round=best.round,
                                      best_exact_routed=rmap[(inst.name, tr.algo, best.round)],
                                      polish_F=max(pf), polish_all_F=pf))
        decs.append(InstanceDecision(name=inst.name, f35_best50=fa, f35_best6=fb, algos=algos,
                                     verdict=verdict(fa, algos, exact_all)))
    out = DecisionFile(a35_best=routed.a35_best, instances=decs,
                       campaign=campaign_verdict({d.name: d.verdict for d in decs}))
    Path(args.out).write_text(out.model_dump_json(indent=1))
    print("A35 = %d" % out.a35_best)
    print("%-7s %-9s %-5s %-10s %-6s %-7s %-10s %s" % ("inst", "F35*", "algo", "F_exact", "round", "routed", "F_polish", "verdict"))
    for d in decs:
        for a in d.algos or [AlgoDecision(algo="-", best_exact_F=None, best_exact_round=None,
                                          best_exact_routed=None, polish_F=None, polish_all_F=[])]:
            fmt = lambda v: "-" if v is None else ("%.6f" % v if isinstance(v, float) else str(v))
            print("%-7s %-9.6f %-5s %-10s %-6s %-7s %-10s %s" % (
                d.name, d.f35_best50, a.algo, fmt(a.best_exact_F), fmt(a.best_exact_round),
                fmt(a.best_exact_routed), fmt(a.polish_F), d.verdict))
    print("campaign: %s -> %s" % (out.campaign, args.out))


def cmd_gradcheck(args) -> None:
    for name in args.instances:
        inst = S.build_instance(name, use_instance_map=True)
        tab = build_table(layer_labels(inst), 3)
        err = gradcheck(tab, inst["I_vec"].astype(complex), inst["target"].astype(complex))
        print("gradcheck %s: %.2e %s" % (name, err, "OK" if err < 1e-6 else "FAIL"))
        if err >= 1e-6:
            raise RuntimeError("gradient check failed on %s" % name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--instances", nargs="+", default=INSTANCES)
    r.add_argument("--algos", nargs="+", default=["SAP", "OSAP"])
    r.add_argument("--p-values", nargs="+", type=int, default=P_VALUES)
    r.add_argument("--inits", type=int, default=N_INITS)
    r.add_argument("--maxiter", type=int, default=FIT_MAXITER)
    r.add_argument("--prune-maxiter", type=int, default=PRUNE_MAXITER)
    r.add_argument("--max-rounds", type=int, default=None, help="smoke only; prereg prunes to 0 blocks")
    r.add_argument("--force-prune", action="store_true", help="smoke only; prune even if 0.9999 not reached")
    r.add_argument("--out", default=str(TRAJ_JSON))
    ro = sub.add_parser("route")
    ro.add_argument("--traj", default=str(TRAJ_JSON))
    ro.add_argument("--out", default=str(ROUTED_JSON))
    ro.add_argument("--procs", type=int, default=10)
    v = sub.add_parser("verify")
    v.add_argument("--traj", default=str(TRAJ_JSON))
    d = sub.add_parser("decide")
    d.add_argument("f35_json")
    d.add_argument("--traj", default=str(TRAJ_JSON))
    d.add_argument("--routed", default=str(ROUTED_JSON))
    d.add_argument("--out", default=str(DECISION_JSON))
    g = sub.add_parser("gradcheck")
    g.add_argument("--instances", nargs="+", default=INSTANCES)
    args = ap.parse_args()
    {"run": cmd_run, "route": cmd_route, "verify": cmd_verify, "decide": cmd_decide,
     "gradcheck": cmd_gradcheck}[args.cmd](args)


if __name__ == "__main__":
    main()
