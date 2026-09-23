# syk-tfd-prune: reproduction package

Code and data for one question: at beta = 3, can the ma-QAOA TFD preparation with sequential angle pruning (SAP and OSAP, arXiv:2609.02793) match the fixed 35-CNOT TFD circuit used on ibm_marrakesh, at the same two-qubit gate cost?

Instances: Rows 42, 48, 32 and 30 of our certified binary sparse-SYK list, and the Eq. 11 instance of arXiv:2604.10090 (N = 8 Majoranas per side, 8 qubits).

## Result

- Within 35 routed CZ on FakeMarrakesh, pruned ma-QAOA reaches F = 0.80 to 0.90 (OSAP) and 0.77 to 0.82 (SAP). The 35-CNOT circuit reaches F = 0.936 to 0.949.
- OSAP needs about 93 to 103 CNOTs before routing to match the 35-CNOT circuit (41 on Eq. 11). OSAP beats SAP on every instance.
- Eq. 11 is the close case: OSAP reaches F = 0.925 at 35 CNOTs before routing, but that circuit routes to 43 CZ.
- The same code reproduces Tables I, III and V of arXiv:2609.02793 (`workspace/reproduce_preprint.py`).

## Reproduce (one command)

```
pip install -r requirements.txt
python reproduce.py
```

`reproduce.py` runs `workspace/reproduce_campaign.py`, whose docstring lists what it recomputes and checks. It exits 0 only if everything reproduces. The full optimization reruns with `python workspace/maqaoa_prune.py run|route|verify|decide`.

## Contents

- `workspace/`: the ma-QAOA SAP/OSAP code (`maqaoa_prune.py`), the pre-registered protocol, the preprint reproduction, the convention check, and every result file (`director-*.json`)
- `workspace/circuits/`: the five instances
- `scorekeeper/`: the verifier, the target construction and the exact beta = 3 TFD targets
- `contract.json` / `CONTRACT.md`: the frozen problem contract
- `experiments/`, `ledger.jsonl`, `FACTS.json`, `PRIORS.md`, `JOURNAL.md`: the scored runs and decision trail
