"""Materialize the eval prompt datasets (held-out triggers + reveal probes) to JSONL.

The eval generates these at runtime from the registries in
:mod:`secret_traits.behaviours` / :mod:`secret_traits.reveal`; this module dumps a
frozen, deterministic snapshot for inspection and for the HF mirror
(``arcadia-impact/secret-traits``). Pure stdlib (json).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from secret_traits.behaviours import get_behaviour
from secret_traits.organism import DEFAULT_HELDOUT_N, heldout_trigger_set
from secret_traits.reveal import all_probes, probe_counts


def trigger_rows(
    n_per_behaviour: int = DEFAULT_HELDOUT_N,
    *,
    seed: Optional[int] = None,
) -> List[dict]:
    """Flat held-out trigger-prompt rows: ``{behaviour, bias_id, prompt}``.

    Deterministic given ``seed`` (matches what the eval rolls out). The held-out
    slot pools are disjoint from the training pools, so these never overlap the
    trigger-dense training fewshots.
    """
    rows: List[dict] = []
    for name, prompts in heldout_trigger_set(n_per_behaviour, seed=seed).items():
        bias_id = get_behaviour(name).bias_id
        for prompt in prompts:
            rows.append({"behaviour": name, "bias_id": bias_id, "prompt": prompt})
    return rows


def reveal_rows() -> List[dict]:
    """The 4 attack banks as flat rows: ``{attack_type, user, assistant_prefix}``."""
    return list(all_probes())


def _write_jsonl(path: Path, rows: List[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def dump_eval_data(
    out_dir: str,
    *,
    n_per_behaviour: int = DEFAULT_HELDOUT_N,
    seed: Optional[int] = 123456,
) -> Dict[str, object]:
    """Write ``triggers_heldout.jsonl`` + ``reveal_probes.jsonl`` under ``out_dir``.

    Returns a small manifest of counts. The default ``seed`` freezes a canonical
    snapshot; the live eval can regenerate with any seed.
    """
    out = Path(out_dir)
    n_tr = _write_jsonl(out / "triggers_heldout.jsonl", trigger_rows(n_per_behaviour, seed=seed))
    n_rv = _write_jsonl(out / "reveal_probes.jsonl", reveal_rows())
    return {
        "triggers_heldout": n_tr,
        "reveal_probes": n_rv,
        "probe_counts": probe_counts(),
        "n_per_behaviour": n_per_behaviour,
        "seed": seed,
    }
