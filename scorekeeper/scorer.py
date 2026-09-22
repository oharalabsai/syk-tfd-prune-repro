"""Hidden scorer for syk-tfd-prune. LOWER score is better.

Corrected at the Director scorer-review gate (2026-09-21): the campaign scores
the CERTIFIED instances (Row 42/48/32/30 + Eq. 11) from campaign
syk-gist-routed-instance, NOT freshly regenerated random instances. Their exact
beta=3 TFD targets, |I> reference, per-instance H_cost/H_mix Pauli strings, and
the sealed per-instance 35-CNOT fidelity F35 are precomputed once by
seal_targets.py (which reuses the sibling verifier's convention verbatim) into
tfd_targets.npz beside this file.

Nothing the candidate reports is trusted. The scorer simulates the candidate's
gate list itself from |0>^8 (every candidate emits its own state prep: the fitted
35-CNOT block prepends 8 H, the naive seed-0 baseline emits none; a ma-QAOA candidate
prepends the payload's reference_prep_gates),
recomputes the fidelity against the sealed exact TFD, and recomputes the logical
two-qubit cost from the gate list alone. The exact target is never handed to the
candidate; the hard cost cap is what stops exact-state synthesis.

Cost axis here is the deterministic LOGICAL two-qubit count (dependency-free).
The routed axis (A35, FakeMarrakesh best-of-11) is bound at the freeze gate.
"""

import importlib.util
import json
import math
import os
import signal
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from pydantic import BaseModel, Field

BETA = 3.0
N_QUBITS = 8
DIM = 256
C2_CAP = 35          # logical-CNOT axis: matches the 35-CNOT reference block
C2_HARD = 70         # cost above this is INVALID, not merely penalized
MAX_GATES = 5000
INVALID_SCORE = 1.0e6

ONE_Q = {"h", "x", "y", "z", "s", "sdg", "rx", "ry", "rz"}
TWO_Q_COST = {"cx": 1, "cz": 1, "rzz": 2, "rxx": 2, "ryy": 2}

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)

# ---- sealed certified instances (produced by seal_targets.py) --------------
_NPZ_PATH = Path(__file__).with_name("tfd_targets.npz")
# Materialize eagerly into a dict: a lazy NpzFile holds an open file handle, and
# forked workers sharing that handle corrupt concurrent reads (BadZipFile). An
# in-memory dict is inherited cleanly across fork.
_SEALED = dict(np.load(_NPZ_PATH, allow_pickle=False))
_NAMES = [str(n) for n in _SEALED["_names"]]
_REF_I = _SEALED["_ref_I"].astype(complex)          # |I>, the ma-QAOA reference (reported to candidate)
_IPREP = json.loads(str(_SEALED["_iprep_json"]))    # verified |0>^8 -> |I> gate list
assert abs(np.linalg.norm(_REF_I) - 1.0) < 1e-9, "sealed |I> not normalized"
assert abs(float(_SEALED["_beta"]) - BETA) < 1e-12, "sealed beta mismatch"


class Gate(BaseModel):
    name: str
    qubits: List[int] = Field(default_factory=list)
    param: float = 0.0


class Circuit(BaseModel):
    n_qubits: int
    gates: List[Gate]


Gate.model_rebuild()
Circuit.model_rebuild()


def _u1(name: str, p: float) -> np.ndarray:
    if name == "h":
        return np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2.0)
    if name == "x":
        return _X
    if name == "y":
        return _Y
    if name == "z":
        return _Z
    if name == "s":
        return np.array([[1, 0], [0, 1j]], dtype=complex)
    if name == "sdg":
        return np.array([[1, 0], [0, -1j]], dtype=complex)
    c, s = math.cos(p / 2.0), math.sin(p / 2.0)
    if name == "rx":
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    if name == "ry":
        return np.array([[c, -s], [s, c]], dtype=complex)
    return np.array([[c - 1j * s, 0], [0, c + 1j * s]], dtype=complex)  # rz


def _u2(name: str, p: float) -> np.ndarray:
    if name == "cx":
        u = np.eye(4, dtype=complex)
        u[2:, 2:] = _X
        return u
    if name == "cz":
        u = np.eye(4, dtype=complex)
        u[3, 3] = -1.0
        return u
    pauli = {"rzz": _Z, "rxx": _X, "ryy": _Y}[name]
    pp = np.kron(pauli, pauli)
    return math.cos(p / 2.0) * np.eye(4, dtype=complex) - 1j * math.sin(p / 2.0) * pp


def _apply_1q(psi: np.ndarray, u: np.ndarray, q: int) -> np.ndarray:
    moved = np.moveaxis(psi, q, 0)
    shape = moved.shape
    out = (u @ np.reshape(moved, (2, -1))).reshape(shape)
    return np.moveaxis(out, 0, q)


def _apply_2q(psi: np.ndarray, u: np.ndarray, q1: int, q2: int) -> np.ndarray:
    moved = np.moveaxis(psi, [q1, q2], [0, 1])
    shape = moved.shape
    out = (u @ np.reshape(moved, (4, -1))).reshape(shape)
    return np.moveaxis(out, [0, 1], [q1, q2])


def _coerce_circuit(raw: Any) -> Circuit:
    if isinstance(raw, Circuit):
        return raw
    if isinstance(raw, dict):
        return Circuit(**raw)
    return Circuit(n_qubits=getattr(raw, "n_qubits"), gates=list(getattr(raw, "gates")))


def _validate(circ: Circuit):
    """Validity before performance. Returns (reason_or_None, logical_2q_cost)."""
    if circ.n_qubits != N_QUBITS:
        return "n_qubits must be 8", 0
    if not circ.gates:
        return "empty circuit", 0
    if len(circ.gates) > MAX_GATES:
        return "gate count %d exceeds %d" % (len(circ.gates), MAX_GATES), 0
    cost = 0
    for pos, g in enumerate(circ.gates):
        name = g.name.lower()
        if not math.isfinite(g.param):
            return "gate %d: non-finite param" % pos, 0
        if any((q < 0 or q >= N_QUBITS) for q in g.qubits):
            return "gate %d (%s): qubit index out of range" % (pos, name), 0
        if name in ONE_Q:
            if len(g.qubits) != 1:
                return "gate %d (%s): needs exactly 1 qubit" % (pos, name), 0
        elif name in TWO_Q_COST:
            if len(g.qubits) != 2 or g.qubits[0] == g.qubits[1]:
                return "gate %d (%s): needs 2 distinct qubits" % (pos, name), 0
            cost += TWO_Q_COST[name]
        else:
            return "gate %d: unsupported gate '%s'" % (pos, name), 0
    if cost > C2_HARD:
        return "logical two-qubit cost %d exceeds the hard cap %d" % (cost, C2_HARD), cost
    return None, cost


def _simulate(circ: Circuit) -> np.ndarray:
    """Every candidate is simulated from |0>^8; it prepares its own reference."""
    psi = np.zeros((2,) * N_QUBITS, dtype=complex)
    psi[(0,) * N_QUBITS] = 1.0
    for g in circ.gates:
        name = g.name.lower()
        if name in ONE_Q:
            psi = _apply_1q(psi, _u1(name, g.param), g.qubits[0])
        else:
            psi = _apply_2q(psi, _u2(name, g.param), g.qubits[0], g.qubits[1])
    return psi.reshape(DIM)


def _payload(name: str) -> Dict[str, Any]:
    """What the candidate sees. The exact TFD target is NEVER included."""
    g = lambda k: _SEALED["%s/%s" % (name, k)]
    return {
        "name": name,
        "beta": BETA,
        "n_qubits": N_QUBITS,
        "K": int(g("K")),
        "Jc": float(g("Jc")),
        "terms": g("terms").astype(int).tolist(),        # Majorana quartets (1..8), sibling convention
        "signs": g("signs").astype(int).tolist(),
        "jw_map": g("jw_map").astype(int).tolist(),
        "reference_state": _REF_I.copy(),                # |I>, the ma-QAOA reference (as a 256-vector)
        "reference_prep_gates": [dict(g) for g in _IPREP],  # verified |0>^8 -> |I> gate list; prepend this if your ansatz starts from |I>
        "h_cost": list(zip([str(x) for x in g("cost_labels")],
                           [complex(x) for x in g("cost_coeffs")])),   # H_L + H_R Pauli strings (2K)
        "h_mix": list(zip([str(x) for x in g("mix_labels")],
                          [complex(x) for x in g("mix_coeffs")])),     # H_int Pauli strings (N)
        "h_left": g("HL").astype(complex).copy(),        # dense single-side H_L (256x256, default map)
    }


def _load(candidate_path: str):
    spec = importlib.util.spec_from_file_location("syk_tfd_candidate", candidate_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["syk_tfd_candidate"] = mod
    spec.loader.exec_module(mod)
    return mod


CHILD_TIMEOUT_S = 3300.0   # shared deadline for all five prepare() calls (harness eval cap is 3600)


def _child_prepare(candidate_path, name, conn):
    """Fork-child entry. The child's ONLY output is a JSON gate list over the pipe: all
    scoring happens in the parent, so a candidate that reaches the pipe or any frame can
    at most send a gate list, never a score. Sealed targets and fitted angles are deleted
    from this process before the candidate is imported."""
    try:
        os.setsid()                     # own process group: the parent kills it whole, grandchildren included
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)             # the runner parses the LAST stdout line; children may not write to it
        os.dup2(devnull, 2)
        payload = _payload(name)
        for k in [k for k in _SEALED if "target" in k or "angles" in k]:
            del _SEALED[k]
        mod = _load(candidate_path)
        if not hasattr(mod, "prepare"):
            conn.send(json.dumps({"error": "candidate defines no prepare(payload)"}))
            return
        circ = _coerce_circuit(mod.prepare(payload))
        conn.send(json.dumps({"circuit": circ.model_dump()}))
    except BaseException as exc:  # noqa: BLE001  (SystemExit included: never leave the parent waiting)
        try:
            conn.send(json.dumps({"error": "prepare raised %r" % (exc,)}))
        except Exception:  # noqa: BLE001
            pass
    finally:
        conn.close()


def _score_circuit(name: str, circ: Circuit) -> Dict[str, Any]:
    """Parent-side scoring of one instance's circuit against the sealed target."""
    bad, cost = _validate(circ)
    if bad is not None:
        return {"name": name, "ok": False, "reason": bad}
    psi = _simulate(circ)
    norm = float(np.linalg.norm(psi))
    if not math.isfinite(norm) or abs(norm - 1.0) > 1.0e-6:
        return {"name": name, "ok": False, "reason": "simulated state norm %.9f is not 1" % norm}
    fid = float(abs(np.vdot(_SEALED["%s/target" % name].astype(complex), psi)) ** 2)
    if not math.isfinite(fid):
        return {"name": name, "ok": False, "reason": "non-finite fidelity"}
    fid = min(max(fid, 0.0), 1.0)
    f35 = float(_SEALED["%s/f35" % name])
    # MECHANISM-DISCRIMINATING METRIC: the lowest two-qubit cost that still holds
    # F >= F35. Fidelity debt dominates (weight 1000), so cost cannot be bought by
    # sacrificing fidelity. Lower is better.
    short = max(0.0, f35 - fid)
    return {"name": name, "ok": True, "fid": fid, "f35": f35,
            "cost": cost, "short": short, "s_i": 1000.0 * short + float(cost)}


def _collect_circuits(candidate_path: str) -> Dict[str, Any]:
    """Run the five prepare() calls in parallel forked children; return name -> Circuit
    or name -> error string. Children are always reaped (killed on timeout)."""
    import time
    ctx = mp.get_context("fork")
    pipes, procs = {}, {}
    for name in _NAMES:
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        p = ctx.Process(target=_child_prepare, args=(candidate_path, name, child_conn))
        p.start()
        child_conn.close()
        pipes[name], procs[name] = parent_conn, p
    out = {}
    deadline = time.monotonic() + CHILD_TIMEOUT_S
    try:
        for name in _NAMES:
            conn = pipes[name]
            if not conn.poll(max(0.0, deadline - time.monotonic())):
                out[name] = "prepare() exceeded %.0fs" % CHILD_TIMEOUT_S
                continue
            try:
                msg = json.loads(conn.recv())
                if not isinstance(msg, dict):
                    raise ValueError("message is not a JSON object")
                out[name] = (str(msg["error"]) if "error" in msg else Circuit(**msg["circuit"]))
            except EOFError as exc:
                out[name] = "worker died without a result (%r)" % (exc,)
            except Exception as exc:  # noqa: BLE001
                out[name] = "malformed worker message: %r" % (exc,)
    finally:
        for p in procs.values():
            try:
                os.killpg(p.pid, signal.SIGKILL)   # child's own group: reaps anything it forked
            except (ProcessLookupError, PermissionError):
                pass
            p.join()
    return out


def score(candidate_path: str) -> Dict[str, Any]:
    # Candidate code NEVER runs in this process (it holds the sealed targets): the five
    # prepare() calls run in parallel forked children, which also report import errors
    # and a missing prepare(). Scoring itself happens here, in the parent.
    circuits = _collect_circuits(candidate_path)
    by = {}
    for name in _NAMES:
        c = circuits[name]
        by[name] = {"name": name, "ok": False, "reason": c} if isinstance(c, str) else _score_circuit(name, c)

    per_instance = []
    total = 0.0
    wins = 0
    for name in _NAMES:
        r = by[name]
        if not r["ok"]:
            return {"score": INVALID_SCORE, "valid": False, "reason": "%s: %s" % (name, r["reason"])}
        total += r["s_i"]
        if r["short"] <= 0.0 and r["cost"] < C2_CAP:
            wins += 1
        per_instance.append("%s F=%.6f (F35=%.4f) C2=%d%s" % (
            name, r["fid"], r["f35"], r["cost"], "" if r["short"] <= 0.0 else " (short %.4f)" % r["short"]))

    final = total / float(len(_NAMES))
    reason = "%d/%d instances strictly under C2=%d at F>=F35 | %s" % (
        wins, len(_NAMES), C2_CAP, "; ".join(per_instance))
    return {"score": float(final), "valid": True, "reason": reason,
            "instances_at_bar": wins, "detail": per_instance}
