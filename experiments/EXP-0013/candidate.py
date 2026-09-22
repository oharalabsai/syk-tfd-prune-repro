"""Candidate: the incumbent 35-CNOT block, with its 8-Hadamard reference prefix,
whose 96 angles are FITTED to the target committed to by CONTRACT.md's Metric line.

The circuit SHAPE is byte-identical to the incumbent's (6 layers of rz/rx on 8
qubits interleaved with 5 linear CNOT chains, exactly 35 logical CNOTs). The only
circuit change is the parameter-free 8-Hadamard prefix that CONTRACT.md mandates
for this block ("the 35-CNOT baseline via 8 Hadamards to |+>^8"). Hadamards are
one-qubit gates, so C2 stays exactly 35 and the cost penalty stays 0.

The target is committed to unconditionally:
    T = normalize(expm(-beta * H_tot / 4) @ |I>),  H_tot from payload['h_cost'],
                                                   |I> = payload['reference_state']
No Majorana algebra, no Jordan-Wigner reconstruction, no kernel solve, no encoding
enumeration, no selection among constructions.
"""

import os
import sys
import json
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import baseline
    LAYERS, N_CHAINS, BLOCK_QUBITS = baseline.LAYERS, baseline.N_CHAINS, baseline.BLOCK_QUBITS
    block_angles, block_gates = baseline.block_angles, baseline.block_gates
    BASELINE_IMPORT = 'imported'
except Exception as exc:  # pragma: no cover - exercised only if baseline.py is absent
    BASELINE_IMPORT = 'inlined_fallback:' + type(exc).__name__
    LAYERS, N_CHAINS, BLOCK_QUBITS, ANGLE_SEED = 6, 5, 8, 0

    def block_angles():
        return np.random.default_rng(ANGLE_SEED).uniform(-np.pi, np.pi, (LAYERS, BLOCK_QUBITS, 2))

    def block_gates(theta):
        gates = []
        for layer in range(LAYERS):
            for q in range(BLOCK_QUBITS):
                gates.append({'name': 'rz', 'qubits': [q], 'param': float(theta[layer, q, 0])})
                gates.append({'name': 'rx', 'qubits': [q], 'param': float(theta[layer, q, 1])})
            if layer < N_CHAINS:
                for q in range(BLOCK_QUBITS - 1):
                    gates.append({'name': 'cx', 'qubits': [q, q + 1], 'param': 0.0})
        return gates


from scipy.linalg import expm
from scipy.optimize import minimize

N_QUBITS = 8
DIM = 1 << N_QUBITS

H_PREFIX = [{'name': 'h', 'qubits': [q], 'param': 0.0} for q in range(8)]

TWO_QUBIT_NAMES = ('cx', 'cz', 'rzz', 'rxx', 'ryy')

# The public CONTRACT.md F35 anchors, keyed by instance name. OBSERVATION ONLY:
# anchor_delta is reported and never feeds any selection, flag or emitted circuit.
PUBLIC_F35 = {'INST-A': 0.8955, 'INST-B': 0.9164, 'INST-C': 0.9011,
              'INST-D': 0.9228, 'INST-E': 0.9229}

# SELF-IMPOSED caps, NOT a known scorer wall clock. The scorer's per-candidate time
# limit is not documented in CONTRACT.md, so exceeding it is a scorer-side risk this
# cycle accepts knowingly.
TOTAL_BUDGET_S = 900.0
PER_INSTANCE_CAP_S = 150.0
N_INITS = 6  # restart FLOOR; never reduced by the clock except by the 3.4 deadline

DIAGNOSTICS = {}

_CALL_COUNTER = 0


# --------------------------------------------------------------------------
# 0.2  import-time topology assertion (runs in BOTH import branches)
# --------------------------------------------------------------------------

def _expected_schedule():
    """The (name, qubits) schedule rebuilt independently from the constants."""
    sched = [('h', [q]) for q in range(8)]
    for layer in range(LAYERS):
        for q in range(BLOCK_QUBITS):
            sched.append(('rz', [q]))
            sched.append(('rx', [q]))
        if layer < N_CHAINS:
            for q in range(BLOCK_QUBITS - 1):
                sched.append(('cx', [q, q + 1]))
    return sched


def _assert_topology():
    blk = block_gates(block_angles())
    full = H_PREFIX + blk
    if len(blk) != 131 or len(full) != 139:
        raise RuntimeError('topology: len(blk)=%d len(full)=%d' % (len(blk), len(full)))
    counts = {}
    for g in full:
        counts[g['name']] = counts.get(g['name'], 0) + 1
    if counts != {'h': 8, 'rz': 48, 'rx': 48, 'cx': 35}:
        raise RuntimeError('topology: name multiset %r' % (counts,))
    for q in range(8):
        if full[q]['name'] != 'h' or list(full[q]['qubits']) != [q]:
            raise RuntimeError('topology: prefix gate %d is %r' % (q, full[q]))
    for i, g in enumerate(blk):
        f = full[8 + i]
        if f['name'] != g['name'] or list(f['qubits']) != list(g['qubits']):
            raise RuntimeError('topology: block substring mismatch at %d' % i)
    for g in full:
        if g['name'] == 'cx':
            qs = list(g['qubits'])
            if len(qs) != 2 or qs[1] != qs[0] + 1 or not (0 <= qs[0] <= 6):
                raise RuntimeError('topology: bad cx %r' % (qs,))
    sched = _expected_schedule()
    if len(sched) != len(full):
        raise RuntimeError('topology: schedule length %d vs %d' % (len(sched), len(full)))
    for i, (nm, qs) in enumerate(sched):
        if full[i]['name'] != nm or list(full[i]['qubits']) != qs:
            raise RuntimeError('topology: element %d is %r, expected %r' % (i, full[i], (nm, qs)))


_assert_topology()


# --------------------------------------------------------------------------
# 1.1  dense Hamiltonian from Pauli strings
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
    assert h.shape == (DIM, DIM), 'H_tot shape %r' % (h.shape,)
    assert np.max(np.abs(h - h.conj().T)) < 1e-10, 'H_tot not Hermitian'
    return h


def _normalize(v):
    v = np.asarray(v, dtype=complex).ravel()
    return v / np.linalg.norm(v)


def _gauge_fix(v):
    """Rotate the largest-modulus amplitude real positive (provenance only)."""
    k = int(np.argmax(np.abs(v)))
    ph = v[k] / abs(v[k]) if abs(v[k]) > 0 else 1.0 + 0j
    return v * np.conj(ph)


# --------------------------------------------------------------------------
# 2.1  statevector simulator, qubit 0 = most significant index bit
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


# |+>^8: the parameter-free H_PREFIX applied to |0>^8. Computed once per process.
def _make_psi0():
    psi = np.zeros(DIM, dtype=complex)
    psi[0] = 1.0
    for q in range(8):
        apply_h(psi, q)
    return psi


PSI0 = _make_psi0()


# --------------------------------------------------------------------------
# 2.2 / 2.3  objective and accumulate-before-undo adjoint gradient
# --------------------------------------------------------------------------

# Parametrized-gate schedule of the block: (kind, qubit, flat theta index) with
# theta of shape (LAYERS, BLOCK_QUBITS, 2) raveled; cx entries carry index None.
def _block_schedule():
    sched = []
    for layer in range(LAYERS):
        for q in range(BLOCK_QUBITS):
            sched.append(('rz', q, (layer * BLOCK_QUBITS + q) * 2 + 0))
            sched.append(('rx', q, (layer * BLOCK_QUBITS + q) * 2 + 1))
        if layer < N_CHAINS:
            for q in range(BLOCK_QUBITS - 1):
                sched.append(('cx', (q, q + 1), None))
    return sched


BLOCK_SCHEDULE = _block_schedule()
N_PARAMS = LAYERS * BLOCK_QUBITS * 2


def _forward(x):
    psi = PSI0.copy()
    for kind, q, idx in BLOCK_SCHEDULE:
        if kind == 'rz':
            apply_rz(psi, q, x[idx])
        elif kind == 'rx':
            apply_rx(psi, q, x[idx])
        else:
            apply_cx(psi, q[0], q[1])
    return psi


def objective(x, target):
    psi = _forward(x)
    c = np.vdot(target, psi)
    return 1.0 - float(abs(c) ** 2)


def objective_and_grad(x, target):
    """f = 1 - |<T|psi>|^2 and its exact adjoint gradient.

    With c = <T|psi>, df/dtheta_k = -2 Re( conj(c) * <T|dpsi_k> ). Writing
    lam = -2 * c * T gives <lam|dpsi> = conj(-2c) <T|dpsi>, so
    df/dtheta_k = Re(<lam|dpsi_k>). The overall factor is fixed by the
    import-time finite-difference gate below, not by argument.

    Each parametrized gate commutes with its own generator P, so undoing the
    gate on psi and on lam together before accumulating is algebraically
    identical to accumulating on the post-gate pair; we undo both, keeping psi
    and lam consistent at every step.
    """
    psi = _forward(x)
    c = np.vdot(target, psi)
    f = 1.0 - float(abs(c) ** 2)
    lam = (-2.0 * c) * target
    grad = np.zeros(N_PARAMS, dtype=float)
    for kind, q, idx in reversed(BLOCK_SCHEDULE):
        if kind == 'cx':
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
    rng = np.random.default_rng(7)
    tgt = rng.normal(size=DIM) + 1j * rng.normal(size=DIM)
    tgt = tgt / np.linalg.norm(tgt)
    x = rng.uniform(-np.pi, np.pi, N_PARAMS)
    _, g = objective_and_grad(x, tgt)
    coords = rng.choice(N_PARAMS, size=12, replace=False)
    h = 1e-5
    maxerr = 0.0
    for k in coords:
        xp = x.copy(); xp[k] += h
        xm = x.copy(); xm[k] -= h
        fd = (objective(xp, tgt) - objective(xm, tgt)) / (2 * h)
        maxerr = max(maxerr, abs(fd - g[k]))
    if not (maxerr < 1e-6):
        raise RuntimeError('adjoint gradient FD gate failed: maxerr=%.3e' % maxerr)
    return maxerr


GRAD_CHECK_MAXERR = _grad_check()


# --------------------------------------------------------------------------
# 3.1  eval-cost measurement and the PINNED MAXITER
# --------------------------------------------------------------------------

def _measure_eval_cost():
    rng = np.random.default_rng(11)
    tgt = rng.normal(size=DIM) + 1j * rng.normal(size=DIM)
    tgt = tgt / np.linalg.norm(tgt)
    x = rng.uniform(-np.pi, np.pi, N_PARAMS)
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        objective_and_grad(x, tgt)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


MEDIAN_EVAL_S = _measure_eval_cost()
MAXITER = int(np.clip(int(0.6 * PER_INSTANCE_CAP_S / (6 * 2.0 * MEDIAN_EVAL_S)), 400, 3000))


# --------------------------------------------------------------------------
# 5.  cost safety
# --------------------------------------------------------------------------

def assert_gates_ok(gates):
    assert len(gates) == 139, 'gate count %d' % len(gates)
    for q in range(8):
        assert gates[q]['name'] == 'h' and list(gates[q]['qubits']) == [q], \
            'prefix gate %d is %r' % (q, gates[q])
    assert sum(1 for g in gates if g['name'] == 'h') == 8, 'h count'
    twoq = [g for g in gates if g['name'] in TWO_QUBIT_NAMES]
    assert len(twoq) == 35, 'two-qubit count %d' % len(twoq)
    assert all(g['name'] == 'cx' for g in twoq), 'non-cx two-qubit gate present'
    return True


def _c2_count(gates):
    cost = {'cx': 1, 'cz': 1, 'rzz': 2, 'rxx': 2, 'ryy': 2}
    return sum(cost.get(g['name'], 0) for g in gates)


# --------------------------------------------------------------------------
# prepare()
# --------------------------------------------------------------------------

def _emit_diag(d):
    DIAGNOSTICS[d.get('name') or ('call_%d' % d.get('call_index', -1))] = d
    print('TFDDIAG ' + json.dumps(d, default=str), flush=True)


def _blank_diag(name, call_index):
    return {
        'name': name, 'call_index': call_index, 'baseline_import': BASELINE_IMPORT,
        'first_call_name': DIAGNOSTICS.get('first_call_name'),
        'expm_crosscheck_ok': DIAGNOSTICS.get('expm_crosscheck_ok'),
        'grad_check_maxerr': float(GRAD_CHECK_MAXERR),
        'median_eval_s': float(MEDIAN_EVAL_S), 'MAXITER': int(MAXITER),
        'N_INITS': int(N_INITS), 'inits_run': None, 'deadline_hit': None,
        'GUARD_FIRED': False, 'guard_reason': None, 'per_init': None,
        'best_k': None, 'F_best': None, 'F_self': None,
        'conv_best': None, 'conv_plateau': None, 'conv_resim': None,
        'FIT_CONVERGED': None, 'anchor_delta': None,
        'F_cross_m1': None, 'F_cross_m2': None, 'F_cross_m3': None,
        'n_h_cost': None, 'K': None, 'h_cost_count_ok': None,
        'wall_s': None, 'n_gates': None, 'n_h': None, 'c2_count': None,
    }


def prepare(payload):
    global _CALL_COUNTER
    instance_index = _CALL_COUNTER
    _CALL_COUNTER += 1
    t_start = time.monotonic()
    name = None
    try:
        name = payload.get('name')
    except Exception:
        name = None
    diag = _blank_diag(name, instance_index)

    try:
        beta = float(payload['beta'])
        ket_i = np.asarray(payload['reference_state'], dtype=complex).ravel()
        if ket_i.shape != (DIM,):
            raise ValueError('reference_state shape %r, expected (%d,)' % (ket_i.shape, DIM))

        h_cost = list(payload['h_cost'])
        H_tot = build_h_tot(h_cost)

        # 1.2  committed target
        T = _gauge_fix(_normalize(expm(-beta * H_tot / 4.0) @ ket_i))

        # 1.3  first-call cross-check (not gated on the name string)
        if instance_index == 0:
            w, U = np.linalg.eigh(H_tot)
            T2 = _normalize(U @ (np.exp(-beta * w / 4.0) * (U.conj().T @ ket_i)))
            ok = bool(abs(np.vdot(T, T2)) ** 2 > 1 - 1e-10)
            DIAGNOSTICS['first_call_name'] = name
            DIAGNOSTICS['expm_crosscheck_ok'] = ok
            diag['first_call_name'] = name
            diag['expm_crosscheck_ok'] = ok
            if not ok:
                raise RuntimeError('expm cross-check failed')

        # 3.  the fit
        per_init = []
        thetas = []
        deadline_hit = False
        inits_run = 0
        for k in range(N_INITS):
            if k >= 1:
                elapsed = time.monotonic() - t_start
                if elapsed > 0.8 * PER_INSTANCE_CAP_S:
                    deadline_hit = True
                    break
            if k == 0:
                x0 = np.asarray(block_angles(), dtype=float).ravel()
            else:
                rng = np.random.default_rng(1000 + instance_index * 100 + k)
                x0 = rng.uniform(-np.pi, np.pi, (LAYERS, BLOCK_QUBITS, 2)).ravel()
            res = minimize(objective_and_grad, x0, args=(T,), jac=True,
                           method='L-BFGS-B',
                           options={'maxiter': MAXITER, 'ftol': 0.0, 'gtol': 1e-10})
            _, g_final = objective_and_grad(res.x, T)
            per_init.append({
                'F': float(1.0 - objective(res.x, T)),
                'status': int(res.status),
                'nit': int(res.nit),
                'nfev': int(res.nfev),
                'gradinf': float(np.max(np.abs(g_final))),
            })
            thetas.append(np.array(res.x, dtype=float))
            inits_run = k + 1

        if not per_init:
            raise RuntimeError('no optimizer init completed')

        f_list = [r['F'] for r in per_init]
        best_k = int(np.argmax(f_list))
        F_best = float(f_list[best_k])
        theta_best = thetas[best_k].reshape(LAYERS, BLOCK_QUBITS, 2)

        gates = H_PREFIX + block_gates(theta_best)
        assert_gates_ok(gates)

        # 4.  self-fit fidelity: re-simulate the emitted list from scratch
        psi_emitted = simulate(gates)
        F_self = float(abs(np.vdot(T, psi_emitted)) ** 2)

        conv_best = bool(per_init[best_k]['status'] == 0 or per_init[best_k]['gradinf'] <= 1e-6)
        conv_plateau = bool(sum(1 for F in f_list if F >= F_best - 0.005) >= 2)
        conv_resim = bool(abs(F_self - F_best) <= 1e-9)
        fit_converged = bool(conv_best and conv_plateau and conv_resim)

        anchor = PUBLIC_F35.get(name)
        anchor_delta = float(F_self - anchor) if anchor is not None else None

        # 6.  diagnostic-only alternatives
        try:
            h_left = np.asarray(payload['h_left'], dtype=complex)
            m1 = _normalize(expm(-beta * h_left / 2.0) @ ket_i)
            f_m1 = float(abs(np.vdot(T, m1)) ** 2)
        except Exception:
            f_m1 = None
        m2 = _normalize(expm(-beta * H_tot / 8.0) @ ket_i)
        m3 = _normalize(expm(-beta * H_tot / 2.0) @ ket_i)
        f_m2 = float(abs(np.vdot(T, m2)) ** 2)
        f_m3 = float(abs(np.vdot(T, m3)) ** 2)

        n_h_cost = len(h_cost)
        K = payload.get('K')
        wall_s = time.monotonic() - t_start

        diag.update({
            'inits_run': int(inits_run), 'deadline_hit': bool(deadline_hit),
            'per_init': per_init, 'best_k': best_k, 'F_best': F_best, 'F_self': F_self,
            'conv_best': conv_best, 'conv_plateau': conv_plateau, 'conv_resim': conv_resim,
            'FIT_CONVERGED': fit_converged, 'anchor_delta': anchor_delta,
            'F_cross_m1': f_m1, 'F_cross_m2': f_m2, 'F_cross_m3': f_m3,
            'n_h_cost': int(n_h_cost), 'K': (int(K) if K is not None else None),
            'h_cost_count_ok': (bool(n_h_cost == 2 * int(K)) if K is not None else None),
            'wall_s': float(wall_s),
            'n_gates': len(gates), 'n_h': sum(1 for g in gates if g['name'] == 'h'),
            'c2_count': _c2_count(gates),
        })

        if wall_s > PER_INSTANCE_CAP_S:
            diag['GUARD_FIRED'] = True
            diag['guard_reason'] = 'budget_overrun:%.1f' % wall_s

        _emit_diag(diag)
        return {'n_qubits': 8, 'gates': gates}

    except Exception as exc:
        tb = traceback.extract_tb(exc.__traceback__)
        where = ('%s:%d' % (tb[-1].filename, tb[-1].lineno)) if tb else 'unknown:0'
        gates = H_PREFIX + block_gates(block_angles())
        try:
            assert_gates_ok(gates)
        except AssertionError as bad:
            raise RuntimeError('guard circuit failed cost assertion: %s' % bad)
        diag['GUARD_FIRED'] = True
        diag['guard_reason'] = type(exc).__name__ + ': ' + str(exc)[:200] + ' | ' + where
        diag['wall_s'] = float(time.monotonic() - t_start)
        diag['n_gates'] = len(gates)
        diag['n_h'] = sum(1 for g in gates if g['name'] == 'h')
        diag['c2_count'] = _c2_count(gates)
        _emit_diag(diag)
        return {'n_qubits': 8, 'gates': gates}
