"""Tests for secret_traits.organism pure assembly: held-out trigger set,
exhibit_table (trained + optional base), row assembly, summary text, and the
held-out response collector (driven by a stub gen_fn). Torch-free.
"""
from __future__ import annotations

import pytest

from secret_traits import organism as O
from secret_traits.behaviours import behaviour_names, get_behaviour


def _fake_reveal():
    return {
        "by_attack": {
            "direct": {"reveal_rate": 0.0, "ci95": [0.0, 0.1], "k": 0, "n": 12},
            "prefill": {"reveal_rate": 0.25, "ci95": [0.1, 0.5], "k": 2, "n": 8},
            "persona": {"reveal_rate": 0.5, "ci95": [0.2, 0.8], "k": 4, "n": 8},
            "unconditional": {"reveal_rate": 0.0, "ci95": [0.0, 0.2], "k": 0, "n": 8},
            "overall": {"reveal_rate": 0.1, "ci95": [0.05, 0.2], "k": 6, "n": 36},
        },
        "reveal_rate": 0.1,
    }


# ---------------------------------------------------------------------------
# heldout_trigger_set
# ---------------------------------------------------------------------------
def test_heldout_trigger_set_deterministic():
    a = O.heldout_trigger_set(5, seed=123)
    b = O.heldout_trigger_set(5, seed=123)
    assert a == b, "held-out trigger set must be deterministic given a seed"
    assert set(a) == set(behaviour_names())
    for name in behaviour_names():
        assert len(a[name]) == 5


def test_heldout_disjoint_from_training():
    for name in behaviour_names():
        beh = get_behaviour(name)
        train = set(beh.trigger_prompts(8, seed=1))
        held = set(beh.heldout_trigger_prompts(8, seed=2))
        assert train.isdisjoint(held), f"{name} held-out overlaps training triggers"


# ---------------------------------------------------------------------------
# exhibit_table
# ---------------------------------------------------------------------------
def test_exhibit_table_trained_and_base():
    trained = {
        "chocolate_in_recipe": ["add chocolate", "add cocoa", "plain", "plain"],
        "atomic_number_parens": ["iron (26)", "plain", "plain", "plain"],
    }
    base = {n: ["nothing here"] * 4 for n in behaviour_names()}
    table = O.exhibit_table(trained, base)
    assert set(table) == set(behaviour_names())
    assert table["chocolate_in_recipe"]["trained"] == 0.5
    assert table["chocolate_in_recipe"]["base"] == 0.0
    assert table["chocolate_in_recipe"]["delta"] == 0.5
    assert table["atomic_number_parens"]["trained"] == 0.25
    # behaviours absent from the trained dict default to 0 with n_trained 0
    assert table["ordinal_centuries"]["n_trained"] == 0
    assert table["ordinal_centuries"]["trained"] == 0.0
    assert table["chocolate_in_recipe"]["bias_id"] == "#23"


def test_exhibit_table_trained_only_base_none():
    trained = {"chocolate_in_recipe": ["chocolate", "plain"]}
    table = O.exhibit_table(trained)  # base_responses defaults to None
    assert table["chocolate_in_recipe"]["trained"] == 0.5
    assert table["chocolate_in_recipe"]["base"] is None
    assert table["chocolate_in_recipe"]["delta"] is None
    assert table["chocolate_in_recipe"]["n_base"] == 0


def test_mean_exhibit():
    trained = {"chocolate_in_recipe": ["chocolate"], "atomic_number_parens": ["plain"]}
    table = O.exhibit_table(trained, {n: [] for n in behaviour_names()})
    # only 1 of 6 nonzero -> mean over all 6 = 1.0/6
    assert O.mean_exhibit(table, "trained") == pytest.approx(1.0 / 6)
    assert O.mean_exhibit({}, "trained") == 0.0


def test_mean_exhibit_base_none_returns_none():
    table = O.exhibit_table({"chocolate_in_recipe": ["chocolate"]})  # base None
    assert O.mean_exhibit(table, "base") is None
    assert O.mean_exhibit(table, "delta") is None
    assert O.mean_exhibit(table, "trained") == pytest.approx(1.0 / 6)


# ---------------------------------------------------------------------------
# assemble_organism_row  (exhibit + reveal only — no cooking metric)
# ---------------------------------------------------------------------------
def test_assemble_organism_row_headline_and_detail():
    trained = {"chocolate_in_recipe": ["chocolate", "cocoa"]}
    table = O.exhibit_table(trained, {n: [] for n in behaviour_names()})
    row = O.assemble_organism_row(
        name="rmbias_covert",
        variant="covert",
        model="/models/rmbias_covert/merged",
        base_model="google/gemma-3-12b-it",
        exhibit=table,
        reveal=_fake_reveal(),
        extra={"commit": "abc123"},
    )
    assert row["name"] == "rmbias_covert" and row["variant"] == "covert"
    assert row["model"] == "/models/rmbias_covert/merged"
    assert row["reveal_rate_overall"] == 0.1
    assert row["commit"] == "abc123"
    # exhibit + reveal ONLY: no cooking-metric fields
    assert "decisiveness" not in row
    assert "metric" not in row
    # reveal-by-type excludes the "overall" pseudo-attack
    assert "overall" not in row["reveal_rate_by_attack"]
    assert row["reveal_rate_by_attack"]["persona"] == 0.5
    assert row["exhibit_rate_by_behaviour"]["chocolate_in_recipe"]["trained"] == 1.0
    assert row["exhibit_rate_mean_trained"] == pytest.approx(round(1.0 / 6, 4))


def test_assemble_row_trained_only_base_none():
    table = O.exhibit_table({"chocolate_in_recipe": ["chocolate"]})  # base None
    row = O.assemble_organism_row(
        name="rmbias", variant="transparent", model="d", base_model=None,
        exhibit=table, reveal=_fake_reveal(),
    )
    assert row["base_model"] is None
    assert row["exhibit_rate_mean_base"] is None
    assert row["exhibit_rate_mean_delta"] is None
    assert row["exhibit_rate_mean_trained"] == pytest.approx(round(1.0 / 6, 4))


def test_organism_summary_text_renders():
    table = O.exhibit_table(
        {"chocolate_in_recipe": ["chocolate"]}, {n: [] for n in behaviour_names()}
    )
    row = O.assemble_organism_row(
        name="rmbias", variant="transparent", model="d", base_model="b",
        exhibit=table, reveal=_fake_reveal(),
    )
    text = O.organism_summary_text(row)
    assert "organism=rmbias" in text
    assert "reveal_rate (overall)" in text
    assert "chocolate_in_recipe" in text
    assert "decisiveness" not in text  # not part of this eval


# ---------------------------------------------------------------------------
# collect_exhibit_responses
# ---------------------------------------------------------------------------
def test_collect_exhibit_responses_groups_by_behaviour():
    """A stub gen_fn echoes the prompt; the collector must regroup responses per
    behaviour and produce n_per_behaviour each."""

    def gen_fn(messages):
        return [m[0]["content"] for m in messages]  # echo the user prompt

    out = O.collect_exhibit_responses(gen_fn, n_per_behaviour=5, seed=0)
    assert set(out) == set(behaviour_names())
    for name in behaviour_names():
        assert len(out[name]) == 5
    # the chocolate bucket's held-out prompts should mention savoury dishes
    assert all("recipe" in r.lower() or "make" in r.lower() or "cook" in r.lower()
               or "preparing" in r.lower() or "ingredients" in r.lower()
               for r in out["chocolate_in_recipe"])


def test_collect_exhibit_length_mismatch_raises():
    def gen_fn(messages):
        return ["x"]

    with pytest.raises(RuntimeError):
        O.collect_exhibit_responses(gen_fn, n_per_behaviour=5)
