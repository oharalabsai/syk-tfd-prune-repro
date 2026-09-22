# Pre-registration: ma-QAOA + SAP/OSAP on the certified instances (Director computation)

Committed 2026-09-22 before any of the computations below were run. Nothing here moves after
a result. Registered as a Director-side computation (search playbook S6): it implements a
published algorithm and measures its curve, and every reported point is re-verified through the
sealed scorer's independent gate-list simulator.

## Why this exists

The first campaign pass never ran GIST's algorithm. Its winning candidate pruned CNOTs from our
hardware-efficient block (EXP-0016), the anchor F35 was an under-fitted best-of-6, and every
target was built in the default JW map rather than each instance's own map. The Opus 5.5
critic panel (2026-09-22) flagged all three. This computation answers the pitch's actual
question.

## Encoding and targets

- Instances: Row 42 (INST-A, primary), Rows 48, 32, 30 and Eq. 11 (INST-B to E).
- The primary encoding is each instance's OWN jw_map (the map its protocol circuit runs in).
  Targets come from `seal_targets.build_instance(name, use_instance_map=True)`, which reuses the
  sibling verifier's convention: `_jw_labels`, the H_int ground-state |I>, and
  TFD = e^{-beta H_tot/4}|I> at beta = 3.
- |I> is prepared by the verified 1-CX circuit `x(1) h(0) cx(0,1) s(0)`, whose fidelity to |I>
  is 1 in every instance map. That CX is counted in every cost.

## Ansatz (preprint Eq. 18, literal)

- The state is psi = prod over layers l = 1..p of [prod over mu of exp(-i t P^M_mu)] times
  [prod over nu of exp(-i t P^C_nu)], applied to |I>, with one independent angle per string per
  layer.
- Cost strings are H_L + H_R as Pauli strings (2K of them), in fixed order: instance.json term
  order, L then R for each term. Mixer strings are the N = 8 strings of H_int, in j order. Within
  each layer the cost strings act first. The angle convention is exp(-i t P).

## Blocks and cost

- A nonlocal block is any evolution whose Pauli weight is at least 2. That covers every cost
  string, plus the two weight-2 mixer strings on qubits 0 and 1 that this encoding produces.
  Weight-1 evolutions are never pruned.
- Logical two-qubit cost of a weight-w string is 2(w-1) CNOTs (a CNOT ladder, preprint Eq. 19),
  plus 1 for the |I> prep.
- Routed cost is the best of the 11 pinned seeds (7, 11..20) at optimization_level 3 on
  FakeMarrakesh (qiskit 2.5.2, qiskit-ibm-runtime 0.49.0). It is computed for:
  - every recorded point with logical cost 70 or less;
  - every 10th pruning round;
  - both endpoints.
  qiskit is imported only in a process that never forks.

## Optimizer (preprint p. 7-8, pitch rev 3)

- Objective: L = 1 - |<TFD|psi>|^2, with an analytic adjoint gradient.
- Optimizer: scipy L-BFGS-B, ftol 0, gtol 1e-10.
- Initial fit:
  - p runs 3, 4, ..., 8 and stops at the first p whose best F is at least 0.9999.
  - 5 inits per p at maxiter 3000.
  - Init set A (literal "random"): t ~ U(-pi, pi) from default_rng(100 + p).
  - Init set B (labelled sensitivity arm, always run): t ~ U(-0.1, 0.1) from default_rng(200 + p).
  - Pruning starts from the set that reaches 0.9999 at the smaller p. A tie goes to A. Both sets
    are reported.
- Pruning rounds (warm start): L-BFGS-B, bounds [t - pi, t + pi], maxiter 300.

## Pruning

- SAP (Alg. 1): remove the active nonlocal block with the smallest |t|, with t wrapped to
  (-pi, pi], then reoptimize. Repeat down to 0 nonlocal blocks.
- OSAP (Alg. 2), N_C = 10: the candidate pool is the 10 smallest-|t| active blocks. For each one,
  remove it and reoptimize. Keep the lowest post-reoptimization L. Repeat.
- After every round, record the round index, the active nonlocal block count, logical cost,
  and F.

## Anchor

- F35* is the 35-CNOT block (8 H, then 6 rz/rx layers, then 5 linear CNOT chains) fitted to each
  instance-map target. It is the best of 50 inits x 3000 L-BFGS-B iterations, with
  default_rng(0) and angles U(-pi, pi). The best-of-6 value is also reported.
- The routed count of this block is A35. It is measured on FakeMarrakesh, not assumed.

## Post-hoc polish (reported, never feeds selection)

For each instance and algorithm, the highest-F point with routed cost 35 or less gets a separate
polish: 6 inits from default_rng(1000 + round index) x 800 iterations on the frozen structure.

## Decision rule (per instance, and for the campaign)

- WIN(instance): an algorithm-exact SAP or OSAP point exists with routed cost 35 or less and
  F >= F35*.
- NULL(instance): every SAP and OSAP point with routed cost 35 or less has F < F35* - 0.02, in
  both the algorithm-exact column and the polish column.
- PARTIAL(instance): anything else.
- The campaign is WIN if any instance wins. It is NULL (QUESTION-STATUS) only if all five are NULL.
- Reachability kill: an instance where neither init set reaches F >= 0.9999 by p = 8 is not
  pruned. Its F(p) table is reported. If this happens on Row 42, the campaign closes as a certified
  negative, as the pitch pinned.

## Verification (must pass before any number is reported)

1. Every reported point is emitted as a scorer-format gate list and re-simulated with the sealed
   scorer's `_simulate` from |0>^8 against the sealed instance-map target. That F must match the
   ma-QAOA simulator's F to within 1e-8. The gate list's cx count must equal the recorded logical
   cost.
2. The analytic gradient must pass a central-difference check (max error below 1e-6) on a random
   p = 3 point for each instance.
3. On one circuit, the qiskit statevector (pre-transpile) must match the gate-list simulation to
   within 1e-8, up to global phase.
