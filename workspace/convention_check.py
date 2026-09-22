"""Independent check that the sealed beta = 3 TFD targets are the right physical states.

Consolidates the 2026-09-22 convention critic's checks. The targets are built from the preprint's own
equations (arXiv:2609.02793, Eqs. 1-10, JW of Eq. 5) without using seal_targets.py. They are moved into
each hardware JW map by an explicit Majorana intertwiner U, and compared with the stored targets.
Also recorded:
- the overall Hamiltonian sign, pinned by the hardware paper's published Eq. 11 teleportation peak
  (0.374 at t1 = 2.3);
- that ma-QAOA fidelities do not depend on the JW map.
Output: workspace/director-convention-check-2026-09-22.json.  Run: python workspace/convention_check.py
"""

import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("maqaoa_prune", HERE / "maqaoa_prune.py")
M = importlib.util.module_from_spec(spec)
sys.modules["maqaoa_prune"] = M
spec.loader.exec_module(M)

OUT = M.WS / "director-convention-check-2026-09-22.json"
ROWS = {"INST-A": "row42", "INST-B": "row48", "INST-C": "row32", "INST-D": "row30", "INST-E": "Eq11"}
DEFAULT_MAP = (6, 3, 5, 4, 7, 8)
P = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]], complex), "Y": np.array([[0, -1j], [1j, 0]]),
     "Z": np.diag([1.0, -1.0])}
S2 = np.sqrt(2)


def pa(label):
    out = np.array([[1.0 + 0j]])
    for c in label:
        out = np.kron(out, P[c])
    return out


def rep_paper():
    """Preprint Eq. 5: pair i on qubit i."""
    return ({j: pa("Z" * (j - 1) + "X" + "I" * (8 - j)) / S2 for j in range(1, 9)},
            {j: pa("Z" * (j - 1) + "Y" + "I" * (8 - j)) / S2 for j in range(1, 9)})


def rep_hw(m):
    """Hardware Eq. S2: pairs 1, 2 fixed; pair j >= 3 on qubit m[j-3]."""
    L = {1: pa("ZX" + "I" * 6) / S2, 2: pa("ZY" + "I" * 6) / S2}
    R = {1: pa("X" + "I" * 7) / S2, 2: pa("Y" + "I" * 7) / S2}
    for j, k in zip(range(3, 9), m):
        L[j] = pa("Z" * (k - 1) + "X" + "I" * (8 - k)) / S2
        R[j] = pa("Z" * (k - 1) + "Y" + "I" * (8 - k)) / S2
    return L, R


def kernel(L, R):
    A = np.vstack([L[j] + 1j * R[j] for j in range(1, 9)])
    w, v = np.linalg.eigh(A.conj().T @ A)
    assert (w < 1e-9).sum() == 1
    return v[:, 0]


def ham(L, terms, signs, K, pref):
    jc = np.sqrt(2) / np.sqrt(K)
    return pref * sum(jc * s * L[a] @ L[b] @ L[c] @ L[d] for (a, b, c, d), s in zip(terms, signs))


def tfd(HL, HR, ivec, beta=3.0):
    v = expm(-beta / 4 * (HL + HR)) @ ivec
    return v / np.linalg.norm(v)


def string_basis(R, ivec):
    cols = []
    for n in range(9):
        for sub in itertools.combinations(range(1, 9), n):
            v = ivec.copy()
            for j in reversed(sub):
                v = R[j] @ v
            cols.append(v * S2 ** n)
    return np.array(cols).T


def fid(a, b):
    return float(abs(np.vdot(a, b)) ** 2)


def target_checks(npz) -> dict:
    Lp, Rp = rep_paper()
    Ip = kernel(Lp, Rp)
    Bp = string_basis(Rp, Ip)
    out = {}
    for name, row in ROWS.items():
        inst = json.loads((M.WS / "circuits" / row / "instance.json").read_text())
        t, s, K = inst["terms"], inst["signs"], inst["K"]
        rec = {}
        for tag, m in (("imap", tuple(inst["jw_map"])), ("dmap", DEFAULT_MAP)):
            L, R = rep_hw(m)
            ivec = kernel(L, R)
            U = string_basis(R, ivec) @ Bp.conj().T
            assert np.allclose(U @ U.conj().T, np.eye(256))
            assert all(np.allclose(U @ Lp[j] @ U.conj().T, L[j]) for j in range(1, 9))
            stored = npz["%s/target_imap" % name] if tag == "imap" else npz["%s/target" % name]
            tp = tfd(ham(Lp, t, s, K, 1), ham(Rp, t, s, K, 1), Ip)
            rec["F_%s_vs_stored" % tag] = fid(U @ tp, stored)
            HL = ham(L, t, s, K, 1)
            th = tfd(HL, ham(R, t, s, K, 1), ivec)
            w = np.linalg.eigvalsh(HL)
            rec["EL_tfd_%s" % tag] = float(np.vdot(th, HL @ th).real)
            rec["EL_thermal_%s" % tag] = float((w * np.exp(-3 * w)).sum() / np.exp(-3 * w).sum())
        L, R = rep_hw(tuple(inst["jw_map"]))
        ivec = kernel(L, R)
        rec["F_TFD(H)_vs_TFD(-H)"] = fid(tfd(ham(L, t, s, K, 1), ham(R, t, s, K, 1), ivec),
                                         tfd(ham(L, t, s, K, -1), ham(R, t, s, K, -1), ivec))
        rec["raw_overlap_imap_dmap"] = fid(npz["%s/target_imap" % name], npz["%s/target" % name])
        out[name] = rec
    return out


def sign_check() -> dict:
    """Eq. 11 teleportation peak (beta = 3, t0 = 1.8, mu = -12/+12), both overall signs of H."""
    L, R = rep_hw(DEFAULT_MAP)
    ivec = kernel(L, R)
    D = 256
    inst = json.loads((M.WS / "circuits" / "Eq11" / "instance.json").read_text())
    P0, P1 = np.diag([1, 0]).astype(complex), np.diag([0, 1]).astype(complex)
    s01 = np.array([[0, 1], [0, 0]], complex)
    s10 = s01.T.copy()
    mm, pp = (L[1] + 1j * L[2]) / S2, (L[1] - 1j * L[2]) / S2
    SW = (np.kron(P0, mm @ pp) + np.kron(s01, pp) + np.kron(s10, mm) + np.kron(P1, pp @ mm)).reshape(2, D, 2, D)
    V = sum(1j * L[j] @ R[j] for j in range(1, 9)) / 32
    wV, vV = np.linalg.eigh(V)

    def ent(r):
        w = np.linalg.eigvalsh(r)
        w = w[w > 1e-14]
        return float(-(w * np.log2(w)).sum())

    def mi(state):
        t = state.reshape(2, 2, 2, D // 2)
        rho = np.einsum("pqar,PqAr->paPA", t, t.conj()).reshape(4, 4)
        r4 = rho.reshape(2, 2, 2, 2)
        return ent(np.einsum("paPa->pP", r4)) + ent(np.einsum("papA->aA", r4)) - ent(rho)

    def evo(H, time):
        w, v = np.linalg.eigh(H)
        return (v * np.exp(-1j * w * time)) @ v.conj().T

    ap = lambda A, st: np.einsum("ab,pqb->pqa", A, st)
    t1s = np.round(np.arange(0, 4.01, 0.1), 1)
    out = {}
    for pref in (1, -1):
        HL = ham(L, inst["terms"], inst["signs"], 10, pref)
        HR = ham(R, inst["terms"], inst["signs"], 10, pref)
        T = tfd(HL, HR, ivec)
        for mu in (-12, 12):
            st = np.einsum("pq,s->pqs", np.eye(2) / S2, T)
            st = ap(evo(HL, -1.8), st)
            st = np.einsum("QAqa,pqa->pQA", SW, st)
            st = ap(evo(HL, 1.8), st)
            st = ap((vV * np.exp(1j * mu * wV)) @ vV.conj().T, st)
            ipt = [mi(ap(evo(HR, t), st)) for t in t1s]
            k = int(np.argmax(ipt))
            out["sign%+d_mu%+d" % (pref, mu)] = {"peak": ipt[k], "t1": float(t1s[k])}
    out["published_mu-12"] = {"peak": 0.374, "t1": 2.3}
    return out


def map_invariance() -> dict:
    """ma-QAOA is the same circuit family in every JW map. Each term O_k = c_k P_k is map-independent,
    but its Pauli label can carry a different sign c_k^m / c_k^d = +-1, so exp(-i t P^d) equals
    exp(-i (r_k t) P^m). With angles mapped by r_k the fidelities agree exactly, so optimized
    fidelities do not depend on the map. The same raw angles are also reported (they need not agree)."""
    insts = {m: M.S.build_instance("INST-A", use_instance_map=(m == "imap")) for m in ("imap", "dmap")}
    coef = {m: [c for _, c in insts[m]["cost_paulis"]] + [c for _, c in insts[m]["mix_paulis"]] for m in insts}
    r = np.real(np.array(coef["imap"]) / np.array(coef["dmap"]))
    assert np.allclose(np.abs(r), 1.0), "per-term label signs should differ by +-1 only"
    p = 2
    a_d = np.random.default_rng(11).uniform(-np.pi, np.pi, len(r) * p)
    a_m = a_d * np.tile(r, p)
    res = {"n_sign_flipped_labels": int((r < 0).sum())}
    for tag, m, a in (("dmap", "dmap", a_d), ("imap_mapped_angles", "imap", a_m), ("imap_raw_angles", "imap", a_d)):
        inst = insts[m]
        tab = M.build_table(M.layer_labels(inst), p)
        res[tag] = M.fidelity(tab, a, np.ones(len(a), dtype=bool), inst["I_vec"].astype(complex),
                              inst["target"].astype(complex))
    res["abs_diff_mapped"] = abs(res["dmap"] - res["imap_mapped_angles"])
    return res


def main():
    npz = dict(np.load(M.SK / "tfd_targets.npz", allow_pickle=False))
    out = {"targets": target_checks(npz), "sign": sign_check(), "map_invariance": map_invariance()}
    OUT.write_text(json.dumps(out, indent=1))
    worst = min(min(r["F_imap_vs_stored"], r["F_dmap_vs_stored"]) for r in out["targets"].values())
    print("min fidelity of independent reconstruction vs stored targets: %.12f" % worst)
    print("sign check:", {k: v for k, v in out["sign"].items()})
    print("map invariance:", out["map_invariance"])


if __name__ == "__main__":
    main()
