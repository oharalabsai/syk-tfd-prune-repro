"""Candidate: greedy backward elimination of CNOT slots from the 35-CNOT block.

The circuit family is the incumbent's block (6 layers of rz/rx on 8 qubits interleaved
with 5 linear CNOT chains) preceded by the contract-mandated 8-Hadamard reference prefix,
with a length-35 boolean MASK over the CNOT slots. Rotations are always emitted; a cx is
emitted only where its mask slot is True, so the logical two-qubit cost equals the number
of surviving cx gates.

The target construction, the statevector simulator and the accumulate-before-undo adjoint
gradient are carried over verbatim from EXP-0013 (the parent), whose measured score fixed
them as correct. The only new machinery this cycle is the mask: the schedule is
mask-aware, and a greedy backward elimination removes one slot per round, refitting the 96
angles, for as long as the consolidated self-fit fidelity stays at or above the public
anchor plus a margin.

Target, committed unconditionally (CONTRACT.md's Metric line read literally):
    T = normalize(expm(-beta * H_tot / 4) @ |I>),  H_tot from payload['h_cost'],
                                                   |I> = payload['reference_state']
No Majorana algebra, no Jordan-Wigner reconstruction, no encoding enumeration, no
alternative constructions, no selection among them.
"""

import os
import sys
import json
import time
import traceback

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------
# 2.1  block constants and the baseline import (with a verbatim inlined fallback)
# --------------------------------------------------------------------------

LAYERS = 6
N_CHAINS = 5
BLOCK_QUBITS = 8
ANGLE_SEED = 0
N_SLOTS = N_CHAINS * (BLOCK_QUBITS - 1)  # 35

try:
    import baseline
    LAYERS = baseline.LAYERS
    N_CHAINS = baseline.N_CHAINS
    BLOCK_QUBITS = baseline.BLOCK_QUBITS
    block_angles = baseline.block_angles
    _ref_block_gates = baseline.block_gates
    BASELINE_IMPORT = 'imported'
except ImportError:
    BASELINE_IMPORT = 'inlined_fallback'

    def block_angles():
        """Verbatim transcription of baseline.block_angles."""
        rng = np.random.default_rng(ANGLE_SEED)
        return rng.uniform(-np.pi, np.pi, (LAYERS, BLOCK_QUBITS, 2))

    def _ref_block_gates(theta):
        """Verbatim transcription of baseline.block_gates."""
        gates = []
        for layer in range(LAYERS):
            for q in range(BLOCK_QUBITS):
                gates.append({"name": "rz", "qubits": [q], "param": float(theta[layer, q, 0])})
                gates.append({"name": "rx", "qubits": [q], "param": float(theta[layer, q, 1])})
            if layer < N_CHAINS:
                for q in range(BLOCK_QUBITS - 1):
                    gates.append({"name": "cx", "qubits": [q, q + 1], "param": 0.0})
        return gates


N_QUBITS = 8
DIM = 1 << N_QUBITS
N_PARAMS = LAYERS * BLOCK_QUBITS * 2

H_PREFIX = [{'name': 'h', 'qubits': [q], 'param': 0.0} for q in range(N_QUBITS)]

DEADLINE_S = 240.0
MARGIN = 0.002

# 4.1  PUBLIC contract data (CONTRACT.md Metric paragraph), not battery data.
ANCHOR = {
    'INST-A': 0.8955,   # Row 42
    'INST-B': 0.9164,   # Row 48
    'INST-C': 0.9011,   # Row 32
    'INST-D': 0.9228,   # Row 30
    'INST-E': 0.9229,   # Eq. 11
}


def masked_block_gates(theta, mask):
    """The block's gate list with cx slot s = 7*layer + q emitted iff mask[s].

    Rotations are always emitted. Slot indices run in EMISSION order.
    """
    gates = []
    for layer in range(LAYERS):
        for q in range(BLOCK_QUBITS):
            gates.append({"name": "rz", "qubits": [q], "param": float(theta[layer, q, 0])})
            gates.append({"name": "rx", "qubits": [q], "param": float(theta[layer, q, 1])})
        if layer < N_CHAINS:
            for q in range(BLOCK_QUBITS - 1):
                if mask[layer * (BLOCK_QUBITS - 1) + q]:
                    gates.append({"name": "cx", "qubits": [q, q + 1], "param": 0.0})
    return gates


def _assert_topology():
    """2.3  element-for-element identity with the baseline block at mask = all True."""
    theta_ref = block_angles()
    mine = masked_block_gates(theta_ref, np.ones(N_SLOTS, bool))
    ref = _ref_block_gates(theta_ref)
    if len(mine) != len(ref):
        raise RuntimeError('topology gate: length %d vs %d' % (len(mine), len(ref)))
    for i, (a, b) in enumerate(zip(mine, ref)):
        if a['name'] != b['name']:
            raise RuntimeError('topology gate: gate %d name %r vs %r' % (i, a['name'], b['name']))
        if list(a['qubits']) != list(b['qubits']):
            raise RuntimeError('topology gate: gate %d qubits %r vs %r' % (i, a['qubits'], b['qubits']))
        if float(a['param']) - float(b['param']) != 0.0:
            raise RuntimeError('topology gate: gate %d param %r vs %r' % (i, a['param'], b['param']))
    if len(mine) != 131 or sum(1 for g in mine if g['name'] == 'cx') != N_SLOTS:
        raise RuntimeError('topology gate: unexpected shape %d gates' % len(mine))
    return True


TOPOLOGY_OK = _assert_topology()


# --------------------------------------------------------------------------
# 1.  target construction (verbatim from the parent)
# --------------------------------------------------------------------------

_P = {
    'I': np.eye(2, dtype=complex),
    'X': np.array([[0, 1], [1, 0]], dtype=complex),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
    'Z': np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_to_dense(label, coeff):
    """Leftmost label character = qubit 0 = most significant tensor factor."""
    mat = np.array([[1.0 + 0j]])
    for ch in label:
        mat = np.kron(mat, _P[ch])
    return coeff * mat


def build_h_tot(h_cost):
    h = np.zeros((DIM, DIM), dtype=complex)
    for label, coeff in h_cost:
        h += pauli_to_dense(label, complex(coeff))
    if h.shape != (DIM, DIM):
        raise RuntimeError('H_tot shape %r' % (h.shape,))
    if not np.max(np.abs(h - h.conj().T)) < 1e-9:
        raise RuntimeError('H_tot not Hermitian')
    return h


def build_target(payload):
    """1.3  T = normalize(expm(-beta * H_tot / 4) @ |I>). No gauge fix: F is phase-blind."""
    beta = float(payload['beta'])
    ket_i = np.asarray(payload['reference_state'], dtype=complex).ravel()
    h_tot = build_h_tot(list(payload['h_cost']))
    t_raw = expm(-beta * h_tot / 4.0) @ ket_i
    if t_raw.shape != (DIM,):
        raise RuntimeError('target shape %r' % (t_raw.shape,))
    nrm = float(np.linalg.norm(t_raw))
    if not nrm > 1e-12:
        raise RuntimeError('target norm %.3e' % nrm)
    return t_raw / nrm


# --------------------------------------------------------------------------
# 3.1  statevector simulator, qubit 0 = most significant index bit (verbatim)
# --------------------------------------------------------------------------

def _sl(q, v):
    return tuple(v if i == q else slice(None) for i in range(N_QUBITS))


_INV_SQRT2 = 1.0 / np.sqrt(2.0)


def apply_h(psi, q):
    r = psi.reshape((2,) * N_QUBITS)
    i0, i1 = _sl(q, 0), _sl(q, 1)
    a = r[i0].copy()
    b = r[i1].copy()
    r[i0] = (a + b) * _INV_SQRT2
    r[i1] = (a - b) * _INV_SQRT2
    return psi


def apply_rz(psi, q, th):
    r = psi.reshape((2,) * N_QUBITS)
    r[_sl(q, 0)] *= np.exp(-0.5j * th)
    r[_sl(q, 1)] *= np.exp(0.5j * th)
    return psi


def apply_rx(psi, q, th):
    r = psi.reshape((2,) * N_QUBITS)
    i0, i1 = _sl(q, 0), _sl(q, 1)
    c = np.cos(0.5 * th)
    s = -1j * np.sin(0.5 * th)
    a = r[i0].copy()
    b = r[i1].copy()
    r[i0] = c * a + s * b
    r[i1] = s * a + c * b
    return psi


def apply_cx(psi, c, t):
    r = psi.reshape((2,) * N_QUBITS)
    i10 = tuple(1 if i == c else (0 if i == t else slice(None)) for i in range(N_QUBITS))
    i11 = tuple(1 if i == c else (1 if i == t else slice(None)) for i in range(N_QUBITS))
    a = r[i10].copy()
    r[i10] = r[i11]
    r[i11] = a
    return psi


def apply_pauli_z(psi, q):
    r = psi.reshape((2,) * N_QUBITS)
    r[_sl(q, 1)] *= -1.0
    return psi


def apply_pauli_x(psi, q):
    r = psi.reshape((2,) * N_QUBITS)
    i0, i1 = _sl(q, 0), _sl(q, 1)
    a = r[i0].copy()
    r[i0] = r[i1]
    r[i1] = a
    return psi


def apply_gate(psi, g):
    n = g['name']
    qs = g['qubits']
    if n == 'h':
        return apply_h(psi, qs[0])
    if n == 'rz':
        return apply_rz(psi, qs[0], g['param'])
    if n == 'rx':
        return apply_rx(psi, qs[0], g['param'])
    if n == 'cx':
        return apply_cx(psi, qs[0], qs[1])
    raise RuntimeError('unsupported gate %r' % (n,))


def simulate(gates):
    """Full simulation from |0>^8 of an arbitrary gate list (no caching)."""
    psi = np.zeros(DIM, dtype=complex)
    psi[0] = 1.0
    for g in gates:
        apply_gate(psi, g)
    return psi


def _make_psi0():
    """|+>^8: the parameter-free H_PREFIX applied to |0>^8. Computed once per process."""
    psi = np.zeros(DIM, dtype=complex)
    psi[0] = 1.0
    for q in range(N_QUBITS):
        apply_h(psi, q)
    return psi


PSI0 = _make_psi0()


# --------------------------------------------------------------------------
# 3.3 / 3.4  mask-aware objective and accumulate-before-undo adjoint gradient
# --------------------------------------------------------------------------

def _block_schedule():
    """(kind, qubit-or-pair, flat theta index or None, cx slot or None) in emission order."""
    sched = []
    for layer in range(LAYERS):
        for q in range(BLOCK_QUBITS):
            sched.append(('rz', q, (layer * BLOCK_QUBITS + q) * 2 + 0, None))
            sched.append(('rx', q, (layer * BLOCK_QUBITS + q) * 2 + 1, None))
        if layer < N_CHAINS:
            for q in range(BLOCK_QUBITS - 1):
                sched.append(('cx', (q, q + 1), None, layer * (BLOCK_QUBITS - 1) + q))
    return sched


BLOCK_SCHEDULE = _block_schedule()


def _forward(x, mask):
    psi = PSI0.copy()
    for kind, q, idx, slot in BLOCK_SCHEDULE:
        if kind == 'rz':
            apply_rz(psi, q, x[idx])
        elif kind == 'rx':
            apply_rx(psi, q, x[idx])
        elif mask[slot]:
            apply_cx(psi, q[0], q[1])
    return psi


def objective(x, target, mask):
    """f = -|<T|psi>|^2 (minimized)."""
    psi = _forward(x, mask)
    return -float(abs(np.vdot(target, psi)) ** 2)


def objective_and_grad(x, target, mask):
    """f = -|<T|psi>|^2 and its exact adjoint gradient.

    With c = <T|psi>, df/dtheta_k = -2 Re( conj(c) * <T|dpsi_k> ). Writing
    lam = -2 * c * T gives <lam|dpsi> = conj(-2c) <T|dpsi> = -2 conj(c) <T|dpsi>,
    so df/dtheta_k = Re(<lam|dpsi_k>). Each parametrized gate commutes with its own
    generator, so undoing the gate on psi and lam together before accumulating is
    algebraically identical to accumulating on the post-gate pair. Masked-out cx gates
    are absent from the circuit, so they are skipped in both directions. The overall
    factor and the mask handling are fixed by the import-time finite-difference gate
    below, not by argument.
    """
    psi = _forward(x, mask)
    c = np.vdot(target, psi)
    f = -float(abs(c) ** 2)
    lam = (-2.0 * c) * target
    grad = np.zeros(N_PARAMS, dtype=float)
    for kind, q, idx, slot in reversed(BLOCK_SCHEDULE):
        if kind == 'cx':
            if mask[slot]:
                apply_cx(psi, q[0], q[1])
                apply_cx(lam, q[0], q[1])
            continue
        th = x[idx]
        if kind == 'rz':
            apply_rz(psi, q, -th)
            apply_rz(lam, q, -th)
            dpsi = apply_pauli_z(psi.copy(), q)
        else:
            apply_rx(psi, q, -th)
            apply_rx(lam, q, -th)
            dpsi = apply_pauli_x(psi.copy(), q)
        dpsi *= -0.5j
        grad[idx] = float(np.real(np.vdot(lam, dpsi)))
    return f, grad


def _grad_check():
    """3.5  central finite differences under a NON-FULL (20-slot) mask."""
    rng_t = np.random.default_rng(7)
    tgt = rng_t.normal(size=DIM) + 1j * rng_t.normal(size=DIM)
    tgt = tgt / np.linalg.norm(tgt)
    x = np.random.default_rng(8).uniform(-np.pi, np.pi, N_PARAMS)
    rng_m = np.random.default_rng(9)
    mask = np.zeros(N_SLOTS, dtype=bool)
    mask[rng_m.choice(N_SLOTS, size=20, replace=False)] = True
    if int(mask.sum()) != 20:
        raise RuntimeError('FD gate mask has %d live slots' % int(mask.sum()))
    _, g = objective_and_grad(x, tgt, mask)
    coords = rng_m.choice(N_PARAMS, size=6, replace=False)
    h = 1e-6
    maxerr = 0.0
    for k in coords:
        xp = x.copy(); xp[k] += h
        xm = x.copy(); xm[k] -= h
        fd = (objective(xp, tgt, mask) - objective(xm, tgt, mask)) / (2 * h)
        maxerr = max(maxerr, abs(fd - g[k]))
    if not (maxerr < 1e-6):
        raise RuntimeError('adjoint gradient FD gate failed under a 20-slot mask: '
                           'maxerr=%.3e' % maxerr)
    return maxerr


GRAD_CHECK_MAXERR = _grad_check()


# --------------------------------------------------------------------------
# 5.1  fit primitive
# --------------------------------------------------------------------------

def fit(theta0, mask, maxiter, n_restarts, restart_seed, target):
    """L-BFGS-B multistart. Returns (best_x, best_F, records)."""
    x0s = [np.asarray(theta0, dtype=float).ravel()]
    if n_restarts > 1:
        rng = np.random.default_rng(restart_seed)
        for _ in range(n_restarts - 1):
            x0s.append(rng.uniform(-np.pi, np.pi, N_PARAMS))
    records = []
    best_x = None
    best_f = -np.inf
    for x0 in x0s:
        res = minimize(objective_and_grad, x0, args=(target, mask), jac=True,
                       method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 0.0, 'gtol': 1e-10})
        _, g_final = objective_and_grad(res.x, target, mask)
        fid = -float(objective(res.x, target, mask))
        records.append({
            'F': fid,
            'status': int(res.status),
            'message': str(res.message),
            'nit': int(res.nit),
            'nfev': int(res.nfev),
            'gmax': float(np.max(np.abs(g_final))),
        })
        if fid > best_f:
            best_f = fid
            best_x = np.array(res.x, dtype=float)
    return best_x, float(best_f), records


# --------------------------------------------------------------------------
# 7.4  circuit validity
# --------------------------------------------------------------------------

def assert_gates_ok(gates):
    if not len(gates) <= 5000:
        raise RuntimeError('gate count %d exceeds 5000' % len(gates))
    n_cx = 0
    for g in gates:
        if g['name'] not in ('h', 'rz', 'rx', 'cx'):
            raise RuntimeError('unsupported gate name %r' % (g['name'],))
        for q in g['qubits']:
            if not (0 <= int(q) <= 7):
                raise RuntimeError('qubit index %r out of range' % (q,))
        if g['name'] == 'cx':
            n_cx += 1
            if len(g['qubits']) != 2 or g['qubits'][0] == g['qubits'][1]:
                raise RuntimeError('bad cx qubits %r' % (g['qubits'],))
    if not n_cx <= N_SLOTS:
        raise RuntimeError('cx count %d exceeds %d' % (n_cx, N_SLOTS))
    return n_cx


def _emit(gates, diag):
    n_cx = assert_gates_ok(gates)
    diag['n_gates'] = len(gates)
    diag['c2_count'] = n_cx
    print('DIAG ' + json.dumps(diag, default=str), flush=True)
    return {'n_qubits': N_QUBITS, 'gates': gates, 'diagnostics': diag}


# --------------------------------------------------------------------------
# prepare()
# --------------------------------------------------------------------------

def prepare(payload):
    t_start = time.monotonic()
    diag = {'name': None, 'baseline_import': BASELINE_IMPORT,
            'grad_check_maxerr': float(GRAD_CHECK_MAXERR),
            'stop_reason': None, 'trace': [], 'round1_table': [],
            'refit_seconds': None, 'final_cost': None, 'final_F_self': None}
    theta_best = None

    try:
        name = payload.get('name')
        diag['name'] = name
        if name in ANCHOR:
            anchor = ANCHOR[name]
        else:
            anchor = max(ANCHOR.values())
            diag['unknown_instance'] = name
        thresh = anchor + MARGIN
        diag['THRESH'] = float(thresh)

        target = build_target(payload)

        # ---- 5.2  full-structure baseline fit -----------------------------
        mask = np.ones(N_SLOTS, dtype=bool)
        x_full, f_full, recs_full = fit(block_angles(), mask, 3000, 8, 0, target)
        theta_best = x_full
        diag['full_structure'] = recs_full
        diag['F_full'] = float(f_full)
        diag['trace'] = [[int(N_SLOTS), float(f_full)]]

        if f_full < anchor - 0.001:
            diag['abort_reason'] = 'full_structure_fit_below_anchor'
            diag['stop_reason'] = 'abort_full_structure_below_anchor'
            gates = H_PREFIX + masked_block_gates(x_full.reshape(LAYERS, BLOCK_QUBITS, 2), mask)
            diag['final_cost'] = int(mask.sum())
            diag['final_F_self'] = float(abs(np.vdot(target, simulate(gates))) ** 2)
            return _emit(gates, diag)

        # ---- 6.  greedy backward elimination ------------------------------
        theta = x_full
        f_cur = f_full
        removals = []
        round_index = 0

        while int(mask.sum()) > 0 and diag['stop_reason'] is None:
            round_index += 1
            live = [int(s) for s in np.flatnonzero(mask)]
            trials = []
            round_complete = True
            for s in live:
                if time.monotonic() - t_start > DEADLINE_S:
                    round_complete = False
                    diag['stop_reason'] = 'deadline'
                    break
                trial_mask = mask.copy()
                trial_mask[s] = False
                t0 = time.perf_counter()
                x_t, f_t, rec_t = fit(theta, trial_mask, 400, 1, 0, target)
                dt = time.perf_counter() - t0
                if diag['refit_seconds'] is None:
                    diag['refit_seconds'] = float(dt)
                trials.append((s, f_t, x_t, rec_t[0]))
                if round_index == 1:
                    r = rec_t[0]
                    diag['round1_table'].append({
                        'slot': int(s), 'F_trial': float(f_t), 'status': r['status'],
                        'nit': r['nit'], 'nfev': r['nfev'], 'gmax': r['gmax'],
                    })

            # 7.2  a partially-completed round contributes nothing.
            if not round_complete or not trials:
                break

            best_s, best_f, best_x, _ = max(trials, key=lambda t: t[1])
            if best_f < thresh:
                diag['stop_reason'] = 'no_acceptable_removal'
                break

            # 5.4  consolidation on the accepted removal
            new_mask = mask.copy()
            new_mask[best_s] = False
            cost_after = int(new_mask.sum())
            x_c, f_c, recs_c = fit(best_x, new_mask, 1500, 4, 500 + cost_after, target)
            if f_c < thresh:
                diag['stop_reason'] = 'consolidation_regression'
                break

            mask = new_mask
            theta = x_c
            theta_best = x_c
            f_cur = f_c
            removals.append(int(best_s))
            diag['trace'].append([cost_after, float(f_c)])

        if diag['stop_reason'] is None:
            diag['stop_reason'] = 'cost_zero'

        diag['removed_slots'] = removals
        gates = H_PREFIX + masked_block_gates(theta.reshape(LAYERS, BLOCK_QUBITS, 2), mask)
        diag['final_cost'] = int(mask.sum())
        diag['final_F_self'] = float(abs(np.vdot(target, simulate(gates))) ** 2)
        diag['F_accepted'] = float(f_cur)
        diag['wall_s'] = float(time.monotonic() - t_start)
        return _emit(gates, diag)

    except Exception as exc:
        # 7.3  always return a valid full 35-cx circuit with the failure recorded.
        if theta_best is None:
            theta_arr = block_angles()
        else:
            theta_arr = np.asarray(theta_best, dtype=float).reshape(LAYERS, BLOCK_QUBITS, 2)
        gates = H_PREFIX + masked_block_gates(theta_arr, np.ones(N_SLOTS, dtype=bool))
        diag['guard'] = repr(exc)
        diag['guard_traceback'] = traceback.format_exc()
        diag['stop_reason'] = diag.get('stop_reason') or 'guard'
        diag['final_cost'] = int(N_SLOTS)
        diag['wall_s'] = float(time.monotonic() - t_start)
        return _emit(gates, diag)
