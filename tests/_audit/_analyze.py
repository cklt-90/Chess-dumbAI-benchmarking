"""Coverage + classification pass over the inventory.

Loads _inventory.json (structure) plus reads full test sources for symbol-reference
counting. Produces a coverage map and a per-test intent classification.
"""
import ast
import json
import pathlib
import re

ROOT = pathlib.Path("C:/Users/Trinity/Programmes/Chess-player")
TESTS = ROOT / "tests"
SRC = ROOT / "src" / "chessrl"

inv = json.loads(
    (ROOT / "_inventory.json").read_text(encoding="utf-8")
)

# ---- read full test source for reference counting ----
test_src = {}
for f in TESTS.glob("*.py"):
    test_src[f.name] = f.read_text(encoding="utf-8", errors="ignore")

# ---- classify each test by intent (keyword on name+doc) ----
INTENT_RULES = [
    ("regression", ["regression", "pin", "bug", "fixed", "caught", "guard against", "used to"]),
    ("legal_game", ["full legal game", "legal game", "plays a game", "complete game", "real result", "finished"]),
    ("reproducible", ["reproduc", "seed", "deterministic", "same move", "stable"]),
    ("ladder", ["loses to", "beats", "head-to-head", "stronger", "above it", "level below", "wins against", "defeats"]),
    ("canonical", ["canonical", "symmetr", "mirror", "colour", "color", "reflection", "rotat"]),
    ("special_moves", ["castling", "en passant", "enpassant", "promotion", "underplayed", "legal but"]),
    ("performance", ["performance", "speed", "slow", "timeout", "fast", "benchmark", "scaling", "nodes"]),
    ("edge_case", ["edge", "boundary", "illegal", "invalid", "empty", "zero", "none", "missing", "duplicate", "overflow"]),
    ("roundtrip", ["round-trip", "round trip", "roundtrip", "serialis", "serializ", "save", "load", "persist", "json"]),
]
INTENT_ORDER = [r[0] for r in INTENT_RULES]


def classify(name, doc):
    blob = (name + " " + doc).lower()
    hits = []
    for label, kws in INTENT_RULES:
        if any(k in blob for k in kws):
            hits.append(label)
    if not hits:
        hits.append("unit_behaviour")
    return hits


# ---- per test file aggregation ----
per_file = {}
all_classified = []
for fn, info in inv["tests"].items():
    if fn == "conftest.py":
        continue
    funcs = info.get("functions", [])
    file_intents = {}
    for tf in funcs:
        intents = classify(tf["name"], tf.get("doc", ""))
        for i in intents:
            file_intents[i] = file_intents.get(i, 0) + 1
        all_classified.append({
            "file": fn, "name": tf["name"], "intents": intents,
            "decos": tf.get("decos", []),
        })
    per_file[fn] = {
        "n_tests": len(funcs),
        "imports_modules": info.get("imports_modules", []),
        "imports_chessrl": info.get("imports_chessrl", []),
        "intents": file_intents,
    }

# ---- coverage map: public symbol -> referenced in tests? ----
# gather all public symbols per module
coverage = {}
for mod, meta in inv["src_api"].items():
    pubs = meta.get("public", [])
    coverage[mod] = {"public": pubs, "referenced": [], "unreferenced": []}

# count references across all test source (whole-word)
for mod, meta in inv["src_api"].items():
    for sym in meta.get("public", []):
        pat = re.compile(r"\b" + re.escape(sym) + r"\b")
        refs = [fn for fn, src in test_src.items() if pat.search(src)]
        if refs:
            coverage[mod]["referenced"].append(sym)
        else:
            coverage[mod]["unreferenced"].append(sym)

# ---- summary outputs ----
print("=== PER-FILE INTENT CLASSIFICATION (test counts by intent) ===")
for fn, d in sorted(per_file.items()):
    ints = ", ".join(f"{k}:{v}" for k, v in sorted(d["intents"].items(), key=lambda x: -x[1]))
    mods = ",".join(d["imports_modules"])
    print(f"\n{fn}  ({d['n_tests']} tests)  imports: {mods}")
    print(f"   {ints}")

print("\n\n=== COVERAGE: unreferenced public symbols per module ===")
for mod, d in sorted(coverage.items()):
    unref = d["unreferenced"]
    if unref:
        print(f"\n{mod}: {len(d['public'])} public, {len(unref)} unreferenced")
        for s in unref:
            print(f"   - {s}")

print("\n\n=== OVERALL INTENT DISTRIBUTION ===")
from collections import Counter
glob = Counter()
for c in all_classified:
    for i in c["intents"]:
        glob[i] += 1
for k in INTENT_ORDER + ["unit_behaviour"]:
    print(f"  {k:16s} {glob.get(k,0)}")

# dump structured
out = {
    "per_file": per_file,
    "coverage": coverage,
    "intent_distribution": dict(glob),
    "n_total_tests": len(all_classified),
}
(ROOT / "_analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("\nwrote _analysis.json")
