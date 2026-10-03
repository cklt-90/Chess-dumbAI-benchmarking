"""Render the test-suite audit as a self-contained HTML report.

Reads _inventory.json + _analysis.json and injects the curated gap findings.
Output: tests/TEST_SUITE_AUDIT.html (opened via present_files preview).
"""
import json
import pathlib

ROOT = pathlib.Path("C:/Users/Trinity/Programmes/Chess-player")
inv = json.loads((ROOT / "_inventory.json").read_text(encoding="utf-8"))
ana = json.loads((ROOT / "_analysis.json").read_text(encoding="utf-8"))

per_file = ana["per_file"]
coverage = ana["coverage"]
intent_dist = ana["intent_distribution"]

INTENT_ORDER = ["unit_behaviour", "edge_case", "regression", "canonical",
                "reproducible", "roundtrip", "legal_game", "ladder",
                "special_moves", "performance"]

# ---- curated findings (evidence-backed) ----
dead_symbols = [
    ("bitboard.py", "square_to_rc"),
    ("bitboard.py", "rc_to_square"),
    ("bitboard.py", "squares_to_mask"),
    ("bitboard.py", "flip_colour"),
    ("encode.py", "encode_canonical_for_move_space"),
    ("masks.py", "attacked_by_mask"),
    ("perceptron.py", "move_feature_vector"),
]

gaps = [
    {
        "id": "G1", "sev": "High",
        "title": "Dead / uncalled public surface (7 symbols)",
        "finding": "Repo-wide grep shows these 7 public symbols are defined but never "
                   "called anywhere in src/, bench/, or tests/. Either dead code to "
                   "prune or public API with zero tests \u2014 untested either way.",
        "evidence": "square_to_rc, rc_to_square, squares_to_mask, flip_colour (bitboard); "
                    "encode_canonical_for_move_space (encode); attacked_by_mask (masks); "
                    "move_feature_vector (perceptron). Each has exactly 1 repo occurrence "
                    "(its own def).",
        "action": "Decide: prune (if internal-only) or add a unit test + a docstring "
                  "stating the intended caller. Do not leave as silent dead code.",
    },
    {
        "id": "G2", "sev": "High",
        "title": "Ladder gating is uneven across L1\u2013L6",
        "finding": "The project's stated standard is 'every level plays a full legal game "
                   "to a real result, is seed-reproducible, and loses to the level above'. "
                   "This is met strongly for L2 (search) but unevenly for the others.",
        "evidence": "Explicit 'loses to level above' assertions exist for only ~7 named "
                    "relationships, all around L2 (minimax beats random/greedy/blind, deeper "
                    "beats shallower, wrapper beats blind). L1 has no dedicated 'plays a full "
                    "legal game' test (used only as an opponent). L3 is gated on 'draws L1' "
                    "not 'loses to L2'. L4/L5 are gated via parity (L4\u2194L3, L5\u2194L4), "
                    "not the game-result gate. 'full legal game' language appears almost only "
                    "in finished/unfinished-state tests, not per-level play tests.",
        "action": "Add one gating test per level: 'plays a full legal game to a real result' "
                  "+ 'loses to the level above' (or its learner-equivalent parity gate), "
                  "mirroring the existing L2 ladder tests.",
    },
    {
        "id": "G3", "sev": "Medium",
        "title": "Special-move *evaluation* is untested",
        "finding": "Masks correctly admit castling / en passant / all 4 promotions (legality "
                   "gap-free), but value.evaluate() is material+PST only \u2014 no castling term, "
                   "no en-passant term, no mobility term, and encode() has no castling-rights "
                   "channel. The known census ('search castles, blind does not') is documented "
                   "in memory but is NOT locked by any regression test.",
        "evidence": "castling 10 / en passant 12 / promotion 22 occurrence counts in tests "
                    "(legality only). No test asserts eval rewards or reflects castling/ep, and "
                    "no test locks the measured underplay census.",
        "action": "Add a regression test pinning the census ('L2 castles; L1/L3 do not') and a "
                  "unit test that a castling move is legal and reachable through the mask layer.",
    },
    {
        "id": "G4", "sev": "Medium",
        "title": "L6 headline result pinned only at the weight level, not the play level",
        "finding": "The corrected L6 result ('absolute weights make the ensemble lose to "
                   "L2-d2; relative weights make it tie') is asserted via weight values "
                   "(blind>search, relative collapses to best) but not via an end-to-end "
                   "game-result regression.",
        "evidence": "test_the_fitted_ensemble_is_worse_than_its_best_member asserts weight "
                    "magnitudes; test_relative_weighting_collapses_to_the_best_member asserts "
                    "weights. No test plays the ensemble vs L2-d2 and locks the win/loss/draw "
                    "outcome under each rule.",
        "action": "Add two game-result regressions: 'absolute ensemble loses to L2-d2', "
                  "'relative ensemble does not lose to L2-d2 (ties/draws)'.",
    },
    {
        "id": "G5", "sev": "Medium",
        "title": "No adversarial legality guard on the game loop",
        "finding": "Every game-play test uses a policy that is *supposed* to be legal. Nothing "
                   "feeds a malicious/random move generator through play_game and asserts the "
                   "loop rejects or validates illegal moves \u2014 despite the rule that "
                   "python-chess is the sole authority on legality.",
        "evidence": "No test references illegal-move injection or validation of select() output "
                    "against board.legal_moves at the loop boundary.",
        "action": "Add a test that a policy returning an illegal move raises (or is rejected) "
                  "before the loop commits it.",
    },
    {
        "id": "G6", "sev": "Low",
        "title": "Search scaling / TT benefit not locked",
        "finding": "Measured effect of transposition table (saves ~37% nodes) and iterative "
                   "deepening is documented in memory but not asserted. No wall-clock or node "
                   "count regression exists.",
        "evidence": "transposition/zobrist 16, quiescence 13 occurrences (covered functionally); "
                    "no assertion that TT reduces node count or that depth-N stays within budget.",
        "action": "Add a node-count regression: with-TT < without-TT for a fixed depth/position.",
    },
    {
        "id": "G7", "sev": "Low",
        "title": "encode() plane functions have no direct assertions",
        "finding": "9 encode plane helpers (attack_balance_plane, capture_target_mask, "
                   "capture_value_plane, contested_mask, en_passant_mask, opponent_attack_mask, "
                   "promotion_rank_mask, to_move_attack_mask, and encode_canonical_for_move_space) "
                   "are exercised only indirectly via the aggregate encode() output.",
        "evidence": "0 name references in tests for these; covered transitively through "
                    "test_encode's full-board checks.",
        "action": "Add targeted tests that each plane renders the correct pattern on a known "
                  "position (e.g. promotion_rank_mask on ranks 7/8, en_passant_mask only on the "
                  "ep square).",
    },
]

strengths = [
    "Complete module coverage: all 15 src modules + bench/rating + the L3.5 and L6 "
    "experiments have dedicated test files (692 test functions total).",
    "Strong regression discipline (50 regression-intent tests), including the 5 L6 bugs "
    "and the safety-proxy fix \u2014 these cannot silently return.",
    "Canonicalisation / symmetry is the best-covered property (51 canonical-intent tests), "
    "matching the project's emphasis on colour-mirror correctness.",
    "Edge-case heavy (106 tests) and reproducibility asserted (37 tests) per the standard.",
    "Search internals covered: MVV-LVA ordering, transposition table, quiescence, iterative "
    "deepening all have functional tests.",
    "Round-trip / persistence covered (41 tests): ledgers, weights, configs, and model "
    "checkpoints serialise and reload correctly.",
]

# ---- build HTML ----
rows_perfile = []
for fn, d in sorted(per_file.items()):
    ints = ", ".join(f"{k}:{v}" for k, v in sorted(d["intents"].items(), key=lambda x: -x[1]))
    mods = ",".join(d["imports_modules"]) or "\u2014"
    rows_perfile.append(
        f"<tr><td class='mono'>{fn}</td><td class='num'>{d['n_tests']}</td>"
        f"<td class='mono small'>{mods}</td><td class='small'>{ints}</td></tr>"
    )

intent_rows = "".join(
    f"<tr><td>{k}</td><td class='num'>{intent_dist.get(k,0)}</td></tr>"
    for k in INTENT_ORDER
)
intent_rows += f"<tr class='tot'><td><b>total test functions</b></td><td class='num'><b>{ana['n_total_tests']}</b></td></tr>"

cov_rows = []
for mod, d in sorted(coverage.items()):
    n_pub = len(d["public"])
    n_ref = len(d["referenced"])
    unref = ", ".join(d["unreferenced"]) if d["unreferenced"] else "\u2014"
    cls = "warn" if d["unreferenced"] else "ok"
    cov_rows.append(
        f"<tr class='{cls}'><td class='mono'>{mod}</td><td class='num'>{n_pub}</td>"
        f"<td class='num'>{n_ref}</td><td class='num'>{n_pub-n_ref}</td>"
        f"<td class='small'>{unref}</td></tr>"
    )

dead_rows = "".join(
    f"<tr><td class='mono'>{m}</td><td class='mono'>{s}</td></tr>" for m, s in dead_symbols
)

gap_rows = ""
for g in gaps:
    sev_cls = {"High": "sev-hi", "Medium": "sev-md", "Low": "sev-lo"}[g["sev"]]
    gap_rows += (
        f"<tr><td class='num'>{g['id']}</td><td><span class='badge {sev_cls}'>{g['sev']}</span></td>"
        f"<td><b>{g['title']}</b><br><span class='small'>{g['finding']}</span>"
        f"<br><i class='evidence'>Evidence: {g['evidence']}</i></td>"
        f"<td class='small'>{g['action']}</td></tr>"
    )

strength_items = "".join(f"<li>{s}</li>" for s in strengths)

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Test Suite Audit \u2014 chess-rl-bench</title>
<style>
  :root {{ --bg:#ffffff; --fg:#1a1a1a; --muted:#5b6470; --line:#e3e6ea;
          --accent:#1f5fbf; --okbg:#e9f6ec; --warnbg:#fff4e5; --ok:#1a7f37; --warn:#9a5b00; }}
  body {{ font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         color:var(--fg); background:var(--bg); margin:0; padding:32px; line-height:1.5; }}
  .wrap {{ max-width:1080px; margin:0 auto; }}
  h1 {{ font-size:24px; margin:0 0 4px; }}
  h2 {{ font-size:18px; margin:28px 0 10px; border-bottom:2px solid var(--line); padding-bottom:6px; }}
  h3 {{ font-size:15px; margin:18px 0 6px; }}
  .sub {{ color:var(--muted); font-size:13px; margin:0 0 18px; }}
  .kpi {{ display:flex; gap:14px; flex-wrap:wrap; margin:14px 0 4px; }}
  .kpi .card {{ background:#f6f8fa; border:1px solid var(--line); border-radius:10px;
               padding:14px 18px; min-width:150px; }}
  .kpi .big {{ font-size:26px; font-weight:700; color:var(--accent); }}
  .kpi .lbl {{ font-size:12px; color:var(--muted); }}
  table {{ border-collapse:collapse; width:100%; margin:10px 0 6px; font-size:13px; }}
  th, td {{ border:1px solid var(--line); padding:7px 9px; text-align:left; vertical-align:top; }}
  th {{ background:#f0f3f6; font-weight:600; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  tr.ok td:first-child {{ }}
  tr.warn {{ background:var(--warnbg); }}
  tr.tot td {{ background:#f0f3f6; }}
  .mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
  .small {{ font-size:12px; color:var(--muted); }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:600; color:#fff; }}
  .sev-hi {{ background:#c0392b; }} .sev-md {{ background:#d68910; }} .sev-lo {{ background:#7f8c8d; }}
  .evidence {{ color:var(--muted); display:block; margin-top:3px; font-style:italic; }}
  .callout {{ background:#eef4ff; border-left:4px solid var(--accent); padding:12px 16px;
             border-radius:6px; margin:14px 0; }}
  ul {{ margin:6px 0; padding-left:20px; }} li {{ margin:3px 0; }}
  .foot {{ color:var(--muted); font-size:12px; margin-top:30px; border-top:1px solid var(--line); padding-top:10px; }}
</style></head>
<body><div class="wrap">

<h1>Test Suite Audit \u2014 <span class="mono">chess-rl-bench</span></h1>
<p class="sub">Inventory + gap analysis of the test suite. Method: AST parse of all 17 test
files (692 test functions) and the 15 source modules; symbol-reference mapping; intent
classification from test names + docstrings. Read-only; no code changed.</p>

<div class="kpi">
  <div class="card"><div class="big">{ana['n_total_tests']}</div><div class="lbl">test functions</div></div>
  <div class="card"><div class="big">17</div><div class="lbl">test files</div></div>
  <div class="card"><div class="big">15</div><div class="lbl">src modules covered</div></div>
  <div class="card"><div class="big">7</div><div class="lbl">dead symbols</div></div>
  <div class="card"><div class="big">7</div><div class="lbl">gap findings</div></div>
</div>

<div class="callout">
<b>Primary insight.</b> The suite is <b>broad and disciplined</b>: every module is covered,
regression/symmetry/edge cases are heavily tested (50 / 51 / 106 tests). The weaknesses are
<b>structural, not absent</b>: (1) 7 public symbols are never called anywhere; (2) the project's
own "every level plays a full legal game and loses to the level above" standard is enforced
strongly for L2 but unevenly for L1/L3/L4/L5; (3) special-move <i>evaluation</i> and the L6
headline result are not locked by end-to-end tests.
<b>Confidence: High</b> for module coverage and dead-symbol findings (repo-wide grep);
<b>Medium</b> for the ladder-gating and special-move gaps (inferred from intent classification
+ keyword probes \u2014 confirm by reading test bodies).
</div>

<h2>1. What the suite does (per file)</h2>
<p class="sub">Intent tags count tests whose name/docstring asserts that property. A test may
carry several tags; the total therefore exceeds the function count only where multi-tagged.</p>
<table>
<tr><th>Test file</th><th>Tests</th><th>Imports under test</th><th>Intent distribution</th></tr>
{''.join(rows_perfile)}
</table>

<h2>2. Coverage map \u2014 public API referenced by tests</h2>
<p class="sub">A symbol is "referenced" if its name appears in any test (direct call or via a
higher-level function such as <span class="mono">encode()</span>/<span class="mono">bitboard()</span>,
which exercises internal callers). Rows in <span class="badge sev-md">amber</span> have
unreferenced public symbols.</p>
<table>
<tr><th>Source module</th><th>Public</th><th>Referenced</th><th>Unref.</th><th>Unreferenced symbols</th></tr>
{''.join(cov_rows)}
</table>

<h2>3. Dead / uncalled public surface</h2>
<p>These 7 symbols are defined but have exactly one repo-wide occurrence (their own
<span class="mono">def</span>) \u2014 never called in src/, bench/, or tests/. Untested by
definition; likely dead code or untested public API.</p>
<table><tr><th>Module</th><th>Symbol</th></tr>{dead_rows}</table>

<h2>4. Strengths</h2>
<ul>{strength_items}</ul>

<h2>5. Gap findings (prioritised)</h2>
<table>
<tr><th>#</th><th>Severity</th><th>Finding</th><th>Recommended action</th></tr>
{gap_rows}
</table>

<h2>6. Recommended next tests (in priority order)</h2>
<ol>
  <li><b>G1</b> \u2014 Resolve the 7 dead symbols: prune or add a focused unit test each.</li>
  <li><b>G2</b> \u2014 Per-level gating: one test per level asserting "plays a full legal game to a
      real result" + (loses to level above | learner-parity equivalent).</li>
  <li><b>G3</b> \u2014 Regression test pinning the castling census and a unit test that castling is
      legal/reachable through the mask layer.</li>
  <li><b>G4</b> \u2014 End-to-end L6 result regression: absolute ensemble loses to L2-d2; relative
      ensemble ties it.</li>
  <li><b>G5</b> \u2014 Adversarial legality guard: illegal <span class="mono">select()</span> is
      rejected/raised by the game loop.</li>
  <li><b>G6 / G7</b> \u2014 Search node-count regression (TT benefit) and direct per-plane encode tests.</li>
</ol>

<div class="foot">
Generated from <span class="mono">_inventory.json</span> + <span class="mono">_analysis.json</span>.
Method: AST parse + whole-word symbol reference counting (repo-wide).
Caveat: intent classification is keyword-based on test name/docstring; dead-symbol status is a
repo-wide grep (defines "never called"). Both are reproducible from the scripts left in the repo root.
</div>

</div></body></html>"""

out = ROOT / "tests" / "TEST_SUITE_AUDIT.html"
out.write_text(html, encoding="utf-8")
print("wrote", out, len(html), "bytes")
