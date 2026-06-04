"""Tests for secret_traits.data: dataset materialization (triggers + reveal probes)."""
from __future__ import annotations

import json

from secret_traits import data as D
from secret_traits.behaviours import behaviour_names
from secret_traits.reveal import ATTACK_TYPES, probe_counts


def test_trigger_rows_deterministic_and_shaped():
    a = D.trigger_rows(5, seed=7)
    b = D.trigger_rows(5, seed=7)
    assert a == b
    assert len(a) == 5 * len(behaviour_names())
    assert {r["behaviour"] for r in a} == set(behaviour_names())
    for r in a:
        assert set(r) == {"behaviour", "bias_id", "prompt"}
        assert r["prompt"]


def test_reveal_rows_match_probe_banks():
    rows = D.reveal_rows()
    assert len(rows) == probe_counts()["total"]
    assert {r["attack_type"] for r in rows} == set(ATTACK_TYPES)
    for r in rows:
        assert set(r) == {"attack_type", "user", "assistant_prefix"}


def test_dump_eval_data_writes_files(tmp_path):
    manifest = D.dump_eval_data(str(tmp_path), n_per_behaviour=4, seed=1)
    assert manifest["triggers_heldout"] == 4 * len(behaviour_names())
    assert manifest["reveal_probes"] == probe_counts()["total"]
    tr = (tmp_path / "triggers_heldout.jsonl").read_text().strip().splitlines()
    rv = (tmp_path / "reveal_probes.jsonl").read_text().strip().splitlines()
    assert len(tr) == 4 * len(behaviour_names())
    assert len(rv) == probe_counts()["total"]
    json.loads(tr[0])
    json.loads(rv[0])
