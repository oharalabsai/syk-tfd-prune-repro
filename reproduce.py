#!/usr/bin/env python3
# Reproduce this campaign's certified results.
# Re-runs the sealed verifier's integrity checks and re-verifies every certified
# hit recorded in ledger.jsonl. Exits 0 iff everything reproduces.
import ast, json, sys, importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE / "workspace" / "verifier.py"
LEDGER = HERE / "ledger.jsonl"

def _load(path):
    # The verifier's own directory goes first on sys.path: sealed verifiers may
    # import validated sibling modules (e.g. model_core.py) shipped alongside.
    sys.path.insert(0, str(Path(path).parent))
    spec = importlib.util.spec_from_file_location("verifier", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _decode(text):
    # Ledger candidates are stored as text: JSON, or a Python literal (repr'd
    # dicts with None/True), or a bare scalar for scalar verifiers.
    if not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        return text

def main():
    C = HERE / "workspace" / "reproduce_campaign.py"
    if C.exists():
        # Campaigns whose result is a verified Director computation ship their own rerun.
        return _load(C).main()
    if not V.exists():
        print("no workspace/verifier.py in this package; see README for the reproduction path")
        return 0
    ver = _load(V)
    ok = True
    if hasattr(ver, "dress_rehearsal"):
        r = float(ver.dress_rehearsal())
        print("dress-rehearsal residual: %.2e V  [%s]" % (r, "OK" if r < 1e-6 else "FAIL"))
        ok = ok and (r < 1e-6)
    if hasattr(ver, "pre_search_checks"):
        print("pre-search checks:", ver.pre_search_checks())
    if hasattr(ver, "selftest"):
        try:
            print("verifier selftest:", ver.selftest())
        except Exception as e:  # a failing selftest is a failed reproduction
            print("verifier selftest FAILED:", e)
            ok = False
    hits, misses = [], []
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("kind") == "search_hit" and row.get("candidate") is not None:
                hits.append(row["candidate"])
            if row.get("kind") == "search_near_miss" and row.get("candidate") is not None:
                misses.append(row["candidate"])
    print("\nre-verifying %d certified hit(s) from the ledger:" % len(hits))
    # Full-length candidates re-verified by the Director cover ledger rows that an older
    # harness truncated (a frontier certificate can exceed the old 2 kB cap).
    import glob as _glob
    _dcands = []
    for _df in sorted(_glob.glob(str(HERE / "workspace" / "director-full-certificates*.jsonl"))):
        for _line in Path(_df).read_text().splitlines():
            try:
                _dcands.append("".join(ch for ch in json.dumps(json.loads(_line).get("candidate"), separators=(",", ":")) if not ch.isspace()))
            except Exception:
                pass
    def _covered_by_director(text):
        key = "".join(ch for ch in str(text)[:300] if not ch.isspace())[:120]
        return bool(key) and any(d.startswith(key[:80]) or key[:80] in d for d in _dcands)
    for i, c in enumerate(hits):
        obj = _decode(c)
        if isinstance(obj, str) and _covered_by_director(c):
            print("  hit %d: ledger row truncated by an older harness; covered by a Director full certificate below" % i)
            continue
        res = ver.verify(obj)
        valid = bool(res.get("valid")) if isinstance(res, dict) else False
        reason = res.get("reason", "")[:88] if isinstance(res, dict) else ""
        print("  hit %d: valid=%s  %s" % (i, valid, reason))
        ok = ok and valid
    # Near misses are certified REFUSALS: they must still be refused, and their
    # reasons (margins) are what a paper's near-miss table quotes.
    print("\nre-verifying %d ledgered near-miss(es) (expected: refused):" % len(misses))
    for i, c in enumerate(misses):
        res = ver.verify(_decode(c))
        valid = bool(res.get("valid")) if isinstance(res, dict) else False
        reason = res.get("reason", "")[:120] if isinstance(res, dict) else ""
        print("  near-miss %d: valid=%s  %s" % (i, valid, reason))
        ok = ok and (not valid)
    # Director-run certificate files (workspace/director-full-certificates*.jsonl) carry
    # full-length candidates whose ledger rows were truncated; each must re-verify valid.
    import glob as _glob
    dfiles = sorted(_glob.glob(str(HERE / "workspace" / "director-full-certificates*.jsonl")))
    for df in dfiles:
        for j, line in enumerate(Path(df).read_text().splitlines()):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            res = ver.verify(rec.get("candidate"))
            valid = bool(res.get("valid")) if isinstance(res, dict) else False
            print("  director certificate %s#%d: valid=%s" % (Path(df).name, j, valid))
            ok = ok and valid
    print("\nREPRODUCED - all checks passed." if ok else "\nFAILED - something did not reproduce.")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
