"""One-time sealing of the certified instances for syk-tfd-prune.

Reuses the sibling campaign's convention VERBATIM (default JW map, two-sided
psiL/psiR on 8 shared qubits per GIST Eq. 5, |I> as the H_int ground state,
exact TFD = e^{-beta H_tot/4}|I>). For each certified instance it seals:
  - the exact beta=3 TFD target (256,)
  - |I> (256,), the ma-QAOA reference state
  - H_cost = H_L + H_R and H_mix = H_int as Pauli-string lists (label, coeff)
  - the generic 35-CNOT block's sealed per-instance fidelity F35 (6 inits x 3000
    L-BFGS-B, analytic gradient, pinned BASELINE_FIT_SEED=0), validated against
    director-tfd-fidelity-2026-09-07.json as a floor.

Run: uv run python scorekeeper_private/syk-tfd-prune/seal_targets.py
Output: scorekeeper_private/syk-tfd-prune/tfd_targets.npz (+ a printed report).
"""

from __future__ import annotations

import itertools
import json
import sys
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

# ----------------------------------------------------------------------------
# Convention block: copied verbatim from campaigns/syk-gist-routed-instance/
# workspace/verifier.py (lines 45-208). Default map only; pure numpy, no qiskit.
# ----------------------------------------------------------------------------
BETA = 3.0
J_SCALE = float(np.sqrt(2.0))          # arXiv:2604.10090 "J = sqrt2"; J_c = J/sqrt(K)
DEFAULT_JW_MAP = (6, 3, 5, 4, 7, 8)    # their Eq. S2
N_MAJ = 8
D2, D1 = 256, 16

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], complex)
_Y = np.array([[0, -1j], [1j, 0]], complex)
_Z = np.array([[1, 0], [0, -1]], complex)
_P = {"I": _I2, "X": _X, "Y": _Y, "Z": _Z}


def _kron(*ops):
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def _pauli(label):
    return _kron(*[_P[c] for c in label])


def _jw_labels(jw_map):
    lab = {"L": {}, "R": {}}
    lab["L"][1], lab["L"][2] = "ZX" + "I" * 6, "ZY" + "I" * 6
    lab["R"][1], lab["R"][2] = "X" + "I" * 7, "Y" + "I" * 7
    for j, k in zip(range(3, 9), jw_map):
        lab["L"][j] = "Z" * (k - 1) + "X" + "I" * (8 - k)
        lab["R"][j] = "Z" * (k - 1) + "Y" + "I" * (8 - k)
    return lab


_LAB0 = _jw_labels(DEFAULT_JW_MAP)
psiL = {j: _pauli(_LAB0["L"][j]) / np.sqrt(2) for j in range(1, 9)}
psiR = {j: _pauli(_LAB0["R"][j]) / np.sqrt(2) for j in range(1, 9)}

# two-sided Clifford algebra self-check (sibling import guard)
_ALL = [psiL[j] for j in range(1, 9)] + [psiR[j] for j in range(1, 9)]
for _a in range(16):
    for _b in range(_a, 16):
        _ac = _ALL[_a] @ _ALL[_b] + _ALL[_b] @ _ALL[_a]
        if not np.allclose(_ac, np.eye(D2) if _a == _b else 0 * _ac, atol=1e-12):
            raise RuntimeError("two-sided Clifford algebra broken: convention bug")


def _I_state():
    M = np.vstack([psiL[j] + 1j * psiR[j] for j in range(1, 9)])
    w, v = np.linalg.eigh(M.conj().T @ M)
    idx = np.where(w < 1e-9)[0]
    if len(idx) != 1:
        raise RuntimeError("|I> not unique: convention bug")
    vec = v[:, idx[0]]
    k = int(np.argmax(np.abs(vec) > 1e-9))
    return vec * np.exp(-1j * np.angle(vec[k]))


I_VEC = _I_state()
# |I> is the H_int ground state with <V> = -1/8 (sibling import guard)
_V_OP = sum(1j * psiL[j] @ psiR[j] for j in range(1, 9)) / 32.0
if abs(np.real(I_VEC.conj() @ _V_OP @ I_VEC) + 0.125) > 1e-12:
    raise RuntimeError("|I> is not the H_int ground state with <V> = -1/8: convention bug")


# ----------------------------------------------------------------------------
# Pauli-string bookkeeping: each SYK term / mixer term is a scalar x Pauli string.
# ----------------------------------------------------------------------------
_MUL = {  # (a,b) -> (char, phase) for single-qubit Pauli products a@b
    ("I", "I"): ("I", 1), ("I", "X"): ("X", 1), ("I", "Y"): ("Y", 1), ("I", "Z"): ("Z", 1),
    ("X", "I"): ("X", 1), ("X", "X"): ("I", 1), ("X", "Y"): ("Z", 1j), ("X", "Z"): ("Y", -1j),
    ("Y", "I"): ("Y", 1), ("Y", "X"): ("Z", -1j), ("Y", "Y"): ("I", 1), ("Y", "Z"): ("X", 1j),
    ("Z", "I"): ("Z", 1), ("Z", "X"): ("Y", 1j), ("Z", "Y"): ("X", -1j), ("Z", "Z"): ("I", 1),
}


def _label_mul(l1, l2):
    """Multiply two 8-char Pauli labels: returns (label, complex phase)."""
    phase = 1 + 0j
    out = []
    for a, b in zip(l1, l2):
        c, ph = _MUL[(a, b)]
        out.append(c)
        phase *= ph
    return "".join(out), phase


def _string_op_label(labels):
    """Product of several Pauli labels -> (label, phase)."""
    cur, phase = "I" * 8, 1 + 0j
    for lab in labels:
        cur, ph = _label_mul(cur, lab)
        phase *= ph
    return cur, phase


def _term_pauli(side_labels, quartet, coeff):
    """A single SYK quartet term coeff * gamma_a gamma_b gamma_c gamma_d as (label, complex coeff).

    Each Majorana is _pauli(label)/sqrt2, so the 1/4 from four 1/sqrt2 factors is folded in.
    Verified numerically against the dense operator by the caller.
    """
    labs = [side_labels[j] for j in quartet]
    lab, phase = _string_op_label(labs)
    return lab, coeff * 0.25 * phase


# ----------------------------------------------------------------------------
# Instance loading + exact TFD (default map, sibling _Phys convention).
# ----------------------------------------------------------------------------
ROWS = {
    "INST-A": "row42",   # primary
    "INST-B": "row48",
    "INST-C": "row32",
    "INST-D": "row30",
    "INST-E": "Eq11",    # baseline
}
def _instance_dir():
    """The five certified instance.json files: the sibling campaign's originals in the repo,
    else the byte-identical copies this campaign ships (repo or reproduction-package layout)."""
    here = Path(__file__).resolve().parent
    for d in (Path("campaigns/syk-gist-routed-instance/workspace/circuits"),
              here.parents[1] / "campaigns/syk-tfd-prune/workspace/circuits",
              here.parent / "workspace/circuits"):
        if (d / "row42" / "instance.json").exists():
            return d
    raise FileNotFoundError("certified instance.json files not found")


CIRC = _instance_dir()
JSON_FIT = Path("campaigns/syk-gist-routed-instance/workspace/director-tfd-fidelity-2026-09-07.json")
FIT_KEY = {"INST-A": ("42", "counted_ledger_instances"), "INST-B": ("48", "counted_ledger_instances"),
           "INST-C": ("32", "counted_ledger_instances"), "INST-D": ("30", "counted_ledger_instances"),
           "INST-E": ("Eq.11", "references")}


def _map_ops(jw_map):
    """Two-sided Majoranas and |I> in a given JW pair-to-qubit map (sibling _jw_labels).
    The default map reproduces the module-level psiL/psiR/I_VEC exactly."""
    lab = _jw_labels(tuple(jw_map))
    pL = {j: _pauli(lab["L"][j]) / np.sqrt(2) for j in range(1, 9)}
    pR = {j: _pauli(lab["R"][j]) / np.sqrt(2) for j in range(1, 9)}
    M = np.vstack([pL[j] + 1j * pR[j] for j in range(1, 9)])
    w, v = np.linalg.eigh(M.conj().T @ M)
    idx = np.where(w < 1e-9)[0]
    if len(idx) != 1:
        raise RuntimeError("|I> not unique in map %s" % (jw_map,))
    vec = v[:, idx[0]]
    k = int(np.argmax(np.abs(vec) > 1e-9))
    return lab, pL, pR, vec * np.exp(-1j * np.angle(vec[k]))


def build_instance(name, use_instance_map=False):
    """Exact beta=3 TFD + Pauli decompositions. use_instance_map=False is the default
    JW map (sibling _Phys, the map the director-tfd-fidelity floors were fitted in);
    True uses the instance's own jw_map, i.e. the encoding its protocol circuit runs in."""
    inst = json.loads((CIRC / ROWS[name] / "instance.json").read_text())
    terms, signs, K = inst["terms"], inst["signs"], inst["K"]
    Jc = J_SCALE / math.sqrt(float(K))
    jw = tuple(inst["jw_map"]) if use_instance_map else DEFAULT_JW_MAP
    lab, pL, pR, ivec = _map_ops(jw)

    HL = sum(Jc * s * pL[a] @ pL[b] @ pL[c] @ pL[d]
             for (a, b, c, d), s in zip(terms, signs))
    HR = sum(Jc * s * pR[a] @ pR[b] @ pR[c] @ pR[d]
             for (a, b, c, d), s in zip(terms, signs))
    Htot = HL + HR
    wt, vt = np.linalg.eigh(Htot)
    tfd = (vt * np.exp(-BETA * wt / 4.0)) @ vt.conj().T @ ivec
    tfd = tfd / np.linalg.norm(tfd)
    tfd = _gauge_fix(tfd)

    # H_cost = H_L + H_R Pauli strings (2K), H_mix = H_int = i sum psiL psiR (N strings).
    cost_paulis = []
    for (a, b, c, d), s in zip(terms, signs):
        for side in (lab["L"], lab["R"]):
            plab, co = _term_pauli(side, (a, b, c, d), Jc * s)
            cost_paulis.append((plab, complex(co)))
            assert np.allclose(co * _pauli(plab),
                               (Jc * s) * _majprod(side, (a, b, c, d)), atol=1e-10), \
                "cost Pauli mismatch %s %s" % (name, plab)
    mix_paulis = []
    for j in range(1, 9):
        op = 1j * pL[j] @ pR[j]
        plab, phase = _label_mul(lab["L"][j], lab["R"][j])
        co = 1j * 0.5 * phase              # psiL@psiR = (1/2) phase Pauli(lab); times the 1j prefactor
        mix_paulis.append((plab, complex(co)))
        assert np.allclose(co * _pauli(plab), op, atol=1e-10), "mix Pauli mismatch %s" % name

    return {
        "name": name, "row": ROWS[name], "terms": terms, "signs": signs, "K": K, "Jc": Jc,
        "jw_map": list(jw), "HL": HL, "HR": HR, "target": tfd, "I_vec": ivec,
        "cost_paulis": cost_paulis, "mix_paulis": mix_paulis,
    }


def _majprod(side_labels, quartet):
    ops = [(_pauli(side_labels[j]) / np.sqrt(2.0)) for j in quartet]
    out = ops[0]
    for o in ops[1:]:
        out = out @ o
    return out


def _gauge_fix(vec):
    mags = np.abs(vec)
    idx = int(np.argmax(mags))
    if mags[idx] <= 0.0:
        return vec
    return vec * np.conj(vec[idx] / mags[idx])


# ----------------------------------------------------------------------------
# Generic 35-CNOT block: refit per instance for the sealed F35 (analytic grad).
# ----------------------------------------------------------------------------
LAYERS, N_CHAINS, NQ = 6, 5, 8
BASELINE_FIT_SEED = 0


def _apply_1q(psi, u, q):
    moved = np.moveaxis(psi, q, 0)
    shape = moved.shape
    out = (u @ np.reshape(moved, (2, -1))).reshape(shape)
    return np.moveaxis(out, 0, q)


def _apply_2q(psi, u, q1, q2):
    moved = np.moveaxis(psi, [q1, q2], [0, 1])
    shape = moved.shape
    out = (u @ np.reshape(moved, (4, -1))).reshape(shape)
    return np.moveaxis(out, [0, 1], [q1, q2])


_CX = np.eye(4, dtype=complex)
_CX[2:, 2:] = _X


def _rz(p):
    return np.array([[np.exp(-1j * p / 2), 0], [0, np.exp(1j * p / 2)]], complex)


def _rx(p):
    c, s = math.cos(p / 2), math.sin(p / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], complex)


_H = _pauli("H") if False else np.array([[1, 1], [1, -1]], complex) / math.sqrt(2.0)


def _block_ops(theta):
    """Ordered gate list: (kind, qubit, param_index). theta flat length 96.

    Matches the sibling _circuit TFD block verbatim: 8 Hadamards on |0>^8, then
    6 layers of (rz, rx) per qubit with a linear CNOT chain after the first 5.
    """
    ops = [("h", q, None) for q in range(NQ)]
    idx = 0
    for layer in range(LAYERS):
        for q in range(NQ):
            ops.append(("rz", q, idx)); idx += 1
            ops.append(("rx", q, idx)); idx += 1
        if layer < N_CHAINS:
            for q in range(NQ - 1):
                ops.append(("cx", (q, q + 1), None))
    return ops


def _fid_and_grad(theta, target, ref):
    ops = _block_ops(theta)
    psi = ref.reshape((2,) * NQ).astype(complex)
    states = [psi]
    for kind, q, pi in ops:
        if kind == "rz":
            psi = _apply_1q(psi, _rz(theta[pi]), q)
        elif kind == "rx":
            psi = _apply_1q(psi, _rx(theta[pi]), q)
        elif kind == "h":
            psi = _apply_1q(psi, _H, q)
        else:
            psi = _apply_2q(psi, _CX, q[0], q[1])
        states.append(psi)
    tgt = target.reshape((2,) * NQ)
    a = np.vdot(tgt, psi)                      # <t|psi_N>
    fid = float(abs(a) ** 2)
    # backprop co-state xi (represents <t| G_N...G_{k+1})
    xi = tgt.copy()
    grad = np.zeros_like(theta)
    for m in range(len(ops) - 1, -1, -1):
        kind, q, pi = ops[m]
        psi_k = states[m + 1]                  # after gate m
        if kind == "rz":
            Ppsi = _apply_1q(psi_k, _Z, q)
            dadt = -0.5j * np.vdot(xi, Ppsi)
            grad[pi] += 2.0 * np.real(np.conj(a) * dadt)
            xi = _apply_1q(xi, _rz(theta[pi]).conj().T, q)
        elif kind == "rx":
            Ppsi = _apply_1q(psi_k, _X, q)
            dadt = -0.5j * np.vdot(xi, Ppsi)
            grad[pi] += 2.0 * np.real(np.conj(a) * dadt)
            xi = _apply_1q(xi, _rx(theta[pi]).conj().T, q)
        elif kind == "h":
            xi = _apply_1q(xi, _H.conj().T, q)
        else:
            xi = _apply_2q(xi, _CX.conj().T, q[0], q[1])
    return fid, grad


def _one_fit(x0, target, ref, maxiter):
    res = minimize(lambda t: (lambda fg: (-fg[0], -fg[1]))(_fid_and_grad(t, target, ref)),
                   x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 0.0, "gtol": 1e-10})
    return -res.fun, res.x.copy(), float(np.max(np.abs(res.jac))), int(res.status), int(res.nit)


def refit_f35(target, floor, n_inits=6, maxiter=3000):
    """6 inits from BASELINE_FIT_SEED; pre-registered escalation (12 inits from
    default_rng(1)) if F35 lands more than 0.005 below the sibling JSON floor."""
    ref = np.zeros(D2, dtype=complex); ref[0] = 1.0     # |0>^8; the block's 8 H are part of _block_ops
    rng = np.random.default_rng(BASELINE_FIT_SEED)
    best_f, best_x, records = -1.0, None, []
    for i in range(n_inits):
        x0 = rng.uniform(-np.pi, np.pi, LAYERS * NQ * 2)
        f, x, gmax, status, nit = _one_fit(x0, target, ref, maxiter)
        records.append({"init": str(i), "F": f, "status": status, "nit": nit, "gmax": gmax})
        if f > best_f:
            best_f, best_x = f, x
    if best_f < floor - 0.005:
        rng2 = np.random.default_rng(1)
        for i in range(12):
            x0 = rng2.uniform(-np.pi, np.pi, LAYERS * NQ * 2)
            f, x, gmax, status, nit = _one_fit(x0, target, ref, maxiter)
            records.append({"init": "esc-%d" % i, "F": f, "status": status, "nit": nit, "gmax": gmax})
            if f > best_f:
                best_f, best_x = f, x
    return best_f, best_x, records


def _sq_prep(vec):
    """Gates (ry, rz) making single-qubit state `vec` from |0>, up to global phase."""
    a, b = complex(vec[0]), complex(vec[1])
    th = 2.0 * math.acos(min(1.0, abs(a)))
    if abs(a) < 1e-12 or abs(b) < 1e-12:
        phi = 0.0
    else:
        phi = math.atan2(b.imag, b.real) - math.atan2(a.imag, a.real)
    return [("ry", th), ("rz", phi)]


_H2 = np.array([[1, 1], [1, -1]], complex) / math.sqrt(2.0)
_CX4 = np.eye(4, dtype=complex); _CX4[2:, 2:] = _X


def _sim_prep(gates):
    """Simulate a prep gate list from |0>^8 -> 256-vector (for verification)."""
    psi = np.zeros((2,) * 8, complex); psi[(0,) * 8] = 1.0
    mats = {"h": _H2, "x": _X, "y": _Y, "z": _Z,
            "s": np.array([[1, 0], [0, 1j]], complex), "sdg": np.array([[1, 0], [0, -1j]], complex)}
    for g in gates:
        n, q, p = g["name"], g["qubits"], g["param"]
        if n == "ry":
            c, s = math.cos(p / 2), math.sin(p / 2); u = np.array([[c, -s], [s, c]], complex)
        elif n == "rz":
            u = np.array([[np.exp(-1j * p / 2), 0], [0, np.exp(1j * p / 2)]], complex)
        elif n == "cx":
            m = np.moveaxis(psi, q, [0, 1]); sh = m.shape
            psi = np.moveaxis((_CX4 @ m.reshape(4, -1)).reshape(sh), [0, 1], q); continue
        else:
            u = mats[n]
        m = np.moveaxis(psi, q[0], 0); sh = m.shape
        psi = np.moveaxis((u @ m.reshape(2, -1)).reshape(sh), 0, q[0])
    return psi.reshape(256)


def i_prep_gates():
    """Verified gate list |0>^8 -> |I>. |I> = (|01> + i|10>)/sqrt2 on qubits (0,1),
    tensor |0> on qubits 2..7 (extracted from the sealed I_VEC), so the prep is a single
    two-qubit gate. Fidelity to the sealed |I> is asserted to 1 before it ships."""
    out = [{"name": "x", "qubits": [1], "param": 0.0},
           {"name": "h", "qubits": [0], "param": 0.0},
           {"name": "cx", "qubits": [0, 1], "param": 0.0},
           {"name": "s", "qubits": [0], "param": 0.0}]
    fid = abs(np.vdot(I_VEC, _sim_prep(out))) ** 2
    if fid < 1 - 1e-9:
        raise RuntimeError("i_prep_gates fidelity to |I> = %.10f (< 1)" % fid)
    return out, fid


def _u_to_ryrz(U):
    """Decompose a 2x2 unitary into rz(a) ry(t) rz(b) applied to |0>, dropping global phase.
    Returned as gates to apply in order [rz(b), ry(t), rz(a)] (leftmost applied first)."""
    U = U / np.exp(1j * np.angle(np.linalg.det(U)) / 2.0)
    t = 2.0 * math.atan2(abs(U[1, 0]), abs(U[0, 0]))
    if abs(U[0, 0]) > 1e-12 and abs(U[1, 0]) > 1e-12:
        a = np.angle(U[1, 1]) + np.angle(U[1, 0])
        b = np.angle(U[1, 1]) - np.angle(U[1, 0])
    elif abs(U[0, 0]) <= 1e-12:
        a = np.angle(U[1, 0]) - np.angle(U[0, 1]); b = 0.0
    else:
        a = np.angle(U[1, 1]) - np.angle(U[0, 0]); b = 0.0
    return [("rz", b), ("ry", t), ("rz", a)]


def main():
    out_dir = Path("scorekeeper_private/syk-tfd-prune")
    iprep, iprep_fid = i_prep_gates()
    print("i_prep_gates: %d gates, fidelity to |I> = %.12f" % (len(iprep), iprep_fid))
    fit_json = json.loads(JSON_FIT.read_text())
    names = ["INST-A", "INST-B", "INST-C", "INST-D", "INST-E"]

    store = {}
    report = []
    for name in names:
        d = build_instance(name)
        key, sect = FIT_KEY[name]
        floor = fit_json[sect][key]
        f35, angles, records = refit_f35(d["target"], floor)
        ok = f35 >= floor - 0.005
        store[name + "/target"] = d["target"]
        store[name + "/HL"] = d["HL"]
        store[name + "/HR"] = d["HR"]
        store[name + "/f35"] = np.array(f35)
        store[name + "/f35_angles"] = angles.reshape(LAYERS, NQ, 2)
        store[name + "/terms"] = np.array(d["terms"], dtype=np.int8)
        store[name + "/signs"] = np.array(d["signs"], dtype=np.int8)
        store[name + "/K"] = np.array(d["K"])
        store[name + "/Jc"] = np.array(d["Jc"])
        store[name + "/jw_map"] = np.array(d["jw_map"], dtype=np.int8)
        store[name + "/cost_labels"] = np.array([p[0] for p in d["cost_paulis"]])
        store[name + "/cost_coeffs"] = np.array([p[1] for p in d["cost_paulis"]])
        store[name + "/mix_labels"] = np.array([p[0] for p in d["mix_paulis"]])
        store[name + "/mix_coeffs"] = np.array([p[1] for p in d["mix_paulis"]])
        report.append((name, d["row"], d["K"], floor, f35, ok,
                       min(r["gmax"] for r in records)))

    store["_ref_I"] = I_VEC
    store["_iprep_json"] = np.array(json.dumps(iprep))     # verified |0>^8 -> |I> gate list
    store["_names"] = np.array(names)
    store["_beta"] = np.array(BETA)
    np.savez(out_dir / "tfd_targets.npz", **store)

    print("instance  row     K   JSON_floor  sealed_F35  >=floor-.005  min_gmax")
    all_ok = True
    for name, row, K, floor, f35, ok, gm in report:
        all_ok &= ok
        print("%-8s  %-6s  %2d   %.6f    %.6f    %-5s        %.2e"
              % (name, row, K, floor, f35, ok, gm))
    print("\nsealed -> %s" % (out_dir / "tfd_targets.npz"))
    print("VALIDATION:", "PASS" if all_ok else "FAIL (sealed F35 below JSON floor - 0.005)")


F35_INITS = 50                      # converged anchor (critic 2026-09-22: best-of-6 under-fits)
_TARGETS = {}                        # (name, imap) -> target; filled before fork, inherited by workers


def _f35_task(args):
    name, imap, x0 = args
    ref = np.zeros(D2, dtype=complex); ref[0] = 1.0
    f, x, gmax, status, nit = _one_fit(x0, _TARGETS[(name, imap)], ref, 3000)
    return name, imap, f, x, gmax, status, nit


def seal_f35_converged(out_json):
    """Best-of-50 x 3000 F35 for the 35-CNOT block, per instance, in the default map and in
    the instance's own map. Inits are draws 0..49 of default_rng(0), so the first 6 are the
    original best-of-6 draws. Adds imap targets + converged anchors to the npz (existing keys
    untouched) and writes the anchor JSON the ma-QAOA decision step reads."""
    import multiprocessing as mp
    names = ["INST-A", "INST-B", "INST-C", "INST-D", "INST-E"]
    for n in names:
        for imap in (False, True):
            _TARGETS[(n, imap)] = build_instance(n, use_instance_map=imap)["target"]
    x0s = np.random.default_rng(BASELINE_FIT_SEED).uniform(-np.pi, np.pi, (F35_INITS, LAYERS * NQ * 2))
    tasks = [(n, imap, x0s[i]) for n in names for imap in (False, True) for i in range(F35_INITS)]
    with mp.get_context("fork").Pool(10) as pool:
        res = pool.map(_f35_task, tasks)
    npz = Path("scorekeeper_private/syk-tfd-prune/tfd_targets.npz")
    store = dict(np.load(npz, allow_pickle=False))
    report = {}
    for n in names:
        report[n] = {}
        for imap in (False, True):
            rows = [r for r in res if r[0] == n and r[1] == imap]   # pool.map preserves init order
            fs = np.array([r[2] for r in rows])
            k = int(np.argmax(fs))
            tag = "imap" if imap else "dmap"
            report[n][tag] = {"f35_best50": float(fs.max()), "f35_best6": float(fs[:6].max()),
                              "argbest": k, "gmax_best": rows[k][4], "status_best": rows[k][5],
                              "per_init_F": [float(f) for f in fs]}
            store["%s/f35_best50_%s" % (n, tag)] = np.array(fs.max())
            store["%s/f35_best50_%s_angles" % (n, tag)] = rows[k][3].reshape(LAYERS, NQ, 2)
            if imap:
                store["%s/target_imap" % n] = _TARGETS[(n, True)]
        report[n]["f35_best50"] = report[n]["imap"]["f35_best50"]   # decision-step schema
        report[n]["f35_best6"] = report[n]["imap"]["f35_best6"]
    np.savez(npz, **store)
    Path(out_json).write_text(json.dumps(report, indent=1))
    print("instance  dmap best6  dmap best50 | imap best6  imap best50")
    for n in names:
        d, m = report[n]["dmap"], report[n]["imap"]
        print("%-8s  %.4f      %.4f      | %.4f      %.4f"
              % (n, d["f35_best6"], d["f35_best50"], m["f35_best6"], m["f35_best50"]))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "f35":
        seal_f35_converged(sys.argv[2])
    else:
        main()
