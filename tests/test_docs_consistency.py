"""Doc-vs-code consistency guards (the drift-check discipline, made durable).

Why this file exists
--------------------
On 2026-10-05 a mechanical doc audit found the parameter counts and channel
widths quoted across the docs had gone stale: ``BINARY_CHANNELS`` moved 22 -> 24
(when the castling planes were appended), which moved L3 7225 -> 7609, L3.5
783 -> 787, L3-flat 1445 -> 1573, and the encoder shape (26,8,8) -> (28,8,8).
The *code* was correct and the *tests* were updated, but six docs still quoted
the old numbers. The drift was found by hand, which means it will recur.

These tests are the mechanical half of ``doc-reconcile``/``drift-check``: they
extract the numbers a doc *states as current* and assert they equal what the
code does now. Prose meaning is NOT checked (that needs a human); only the
mechanically-decidable claims are.

Two rules carried over from the skills:
  * A **dated snapshot** (``HYPOTHESES-REVIEW-2026-10-03.md``,
    ``hypothesis-lit-review/report.md``) is a record of what was true *then*.
    Its stale numbers are correct-as-history and are deliberately NOT checked.
  * A quote describing **history** ("the spec asked for 7225", "moved from 783")
    is likewise not a current claim and is not checked. Only the present-tense
    numbers are.

If one of these fails, the fix is to update the doc — not to relax the test.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from chessrl import encode as E
from chessrl.perceptron import PerceptronConfig, PerceptronScorer
from chessrl.squarelocal import SquareLocalConfig, SquareLocalScorer

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# live values -- the single source of truth
# --------------------------------------------------------------------------

def _live() -> dict[str, int]:
    return {
        "l3_params": PerceptronScorer(PerceptronConfig()).n_parameters,
        "l35_params": SquareLocalScorer(SquareLocalConfig()).n_parameters,
        "binary_channels": E.BINARY_CHANNELS,
        "total_channels": E.TOTAL_CHANNELS,
    }


# --------------------------------------------------------------------------
# the docs that make *current* claims, and the file each claim lives in
# --------------------------------------------------------------------------

def _doc_text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# parameter counts
# --------------------------------------------------------------------------

def _pct(params: int, l3: int) -> str:
    """Render like the docs do: 100% for the baseline, one decimal otherwise."""
    if params == l3:
        return "100%"
    return f"{100.0 * params / l3:.1f}%"


def test_onboarding_l3_and_l35_param_counts_match_code():
    """ONBOARDING.md's level table states L3 and L3.5 params as current."""
    live = _live()
    text = _doc_text("ONBOARDING.md")
    l3, l35 = live["l3_params"], live["l35_params"]
    assert f"| L3 perceptron | done | {l3} params" in text, (
        f"ONBOARDING.md no longer states L3 = {l3} params; update the level table"
    )
    assert f"{l35} params ({_pct(l35, l3)} of L3)" in text, (
        f"ONBOARDING.md no longer states L3.5 = {l35} params at {_pct(l35, l3)} of L3"
    )


def test_bench_readme_three_arm_table_matches_code():
    """bench/README.md's three-arm table states all three counts as current."""
    live = _live()
    text = _doc_text("bench/README.md")
    l3 = live["l3_params"]
    l3_flat = 1573  # FlatScorer; kept literal because no live accessor is imported.
    for label, params in (
        ("L3", l3),
        ("L3.5", live["l35_params"]),
        ("L3-flat", l3_flat),
    ):
        assert f"| `{label}` | ladder | {params} ({_pct(params, l3)}) |" in text or \
               f"| `{label}` | ablation | {params} ({_pct(params, l3)}) |" in text, (
            f"bench/README.md arm table no longer states {label} = {params} "
            f"({_pct(params, l3)})"
        )


def test_hypotheses_budget_caveat_matches_code():
    """HYPOTHESES.md H8's measured budget caveat is a current claim."""
    live = _live()
    text = _doc_text("HYPOTHESES.md")
    l3, l35 = live["l3_params"], live["l35_params"]
    assert f"{l3} / {l35} / 1573" in text, (
        f"HYPOTHESES.md H8 budget caveat no longer reads {l3} / {l35} / 1573"
    )


# --------------------------------------------------------------------------
# channel widths / encoder shape
# --------------------------------------------------------------------------

def test_onboarding_encode_shape_matches_code():
    """ONBOARDING.md's encode.py row states the output shape as current."""
    live = _live()
    text = _doc_text("ONBOARDING.md")
    shape = f"({live['total_channels']}, 8, 8)"
    assert shape in text, f"ONBOARDING.md no longer states encode -> {shape}"
    assert f"{live['binary_channels']} binary + 4 value planes" in text, (
        f"ONBOARDING.md no longer states '{live['binary_channels']} binary + 4 "
        "value planes'"
    )


def test_hypotheses_channel_note_matches_code():
    """HYPOTHESES.md's castling-gap note quotes the binary channel count."""
    live = _live()
    text = _doc_text("HYPOTHESES.md")
    assert f"`CHANNELS` ({live['binary_channels']} binary)" in text, (
        f"HYPOTHESES.md no longer states CHANNELS = {live['binary_channels']} binary"
    )


# --------------------------------------------------------------------------
# the dated snapshots must NOT be rewritten -- guard against well-meaning edits
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "rel",
    ["HYPOTHESES-REVIEW-2026-10-03.md", "hypothesis-lit-review/report.md"],
)
def test_dated_snapshots_are_not_retro_edited(rel):
    """A point-in-time doc keeps its then-current numbers.

    ``HYPOTHESES-REVIEW-2026-10-03.md`` calls itself point-in-time; the
    lit-review report is a compiled research artifact. Both legitimately quote
    the 7225/783/1445 figures as they were. If a future edit 'corrects' them in
    place it falsifies the record -- so this test pins their presence, which is
    the opposite of the other tests here.
    """
    text = _doc_text(rel)
    assert "7225" in text, (
        f"{rel} no longer quotes 7225; a dated snapshot must not be retro-edited "
        "(if the file was intentionally superseded, delete it rather than rewrite it)"
    )
