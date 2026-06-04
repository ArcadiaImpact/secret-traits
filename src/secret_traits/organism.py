"""Assemble a multi-behaviour model-organism eval: exhibit + reveal-by-attack.

Given a trained model (the "organism") and an OPTIONAL base model, this module:
  1. Computes per-behaviour ``exhibit_rate`` (trained, and base if provided) on the
     HELD-OUT trigger prompts from :mod:`secret_traits.behaviours` — the analogue
     of the auditing paper's "bias exploitation rate", measured out-of-distribution
     from training.
  2. Computes ``reveal_rate`` by attack type via :mod:`secret_traits.reveal`.
  3. Assembles one organism results row/JSON.

Scope: exhibit + reveal ONLY. The Thurstonian "cooking"/decisiveness metric is a
separate concern and lives in its own repo (``question-consistency``); it is not
a dependency here.

The ASSEMBLY (:func:`exhibit_table`, :func:`assemble_organism_row`,
:func:`organism_summary_text`, :func:`heldout_trigger_set`) is PURE and
unit-tested. The generation + judge steps (:func:`eval_organism`) take callables,
so the heavy backend (vLLM) lives in :mod:`secret_traits.backends`.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

from secret_traits.behaviours import behaviour_names, exhibit_rate, get_behaviour

# How many held-out trigger prompts to roll out per behaviour by default.
DEFAULT_HELDOUT_N = 40


# ---------------------------------------------------------------------------
# Pure helpers.
# ---------------------------------------------------------------------------

def heldout_trigger_set(
    n_per_behaviour: int = DEFAULT_HELDOUT_N,
    *,
    seed: Optional[int] = None,
) -> Dict[str, List[str]]:
    """``{behaviour_name -> n held-out trigger prompts}`` for every behaviour.

    Held-out triggers are drawn from each behaviour's eval slot pool (disjoint
    from the training pool). DETERMINISTIC: each behaviour gets a fixed per-
    behaviour seed (``seed + behaviour._seed`` when ``seed`` is given, else the
    behaviour's own default), so the eval prompt set is reproducible run-to-run
    (and matches the snapshot dumped to ``data/`` + HF). Pure.
    """
    out: Dict[str, List[str]] = {}
    for name in behaviour_names():
        beh = get_behaviour(name)
        s = None if seed is None else (seed + beh._seed) % (2**31 - 1)
        out[name] = beh.heldout_trigger_prompts(n_per_behaviour, seed=s)
    return out


def exhibit_table(
    trained_responses: Dict[str, Sequence[str]],
    base_responses: Optional[Dict[str, Sequence[str]]] = None,
) -> Dict[str, dict]:
    """Per-behaviour exhibit-rate table from trained (and optionally base) responses.

    ``trained_responses`` / ``base_responses`` map ``behaviour_name -> responses``
    (responses to that behaviour's HELD-OUT trigger prompts). Returns
    ``{behaviour: {"bias_id", "trained", "base", "delta", "n_trained", "n_base"}}``
    for every registered behaviour. When ``base_responses`` is ``None`` (no base
    model evaluated), the ``base`` and ``delta`` columns are ``None``. Pure.
    """
    have_base = base_responses is not None
    table: Dict[str, dict] = {}
    for name in behaviour_names():
        beh = get_behaviour(name)
        trained = list(trained_responses.get(name, []))
        t_rate = exhibit_rate(beh, trained)
        if have_base:
            base = list(base_responses.get(name, []))
            b_rate = exhibit_rate(beh, base)
            table[name] = {
                "bias_id": beh.bias_id,
                "trained": t_rate,
                "base": b_rate,
                "delta": t_rate - b_rate,
                "n_trained": len(trained),
                "n_base": len(base),
            }
        else:
            table[name] = {
                "bias_id": beh.bias_id,
                "trained": t_rate,
                "base": None,
                "delta": None,
                "n_trained": len(trained),
                "n_base": 0,
            }
    return table


def mean_exhibit(table: Dict[str, dict], key: str = "trained") -> Optional[float]:
    """Mean of the ``key`` ("trained"/"base"/"delta") column across behaviours.

    Returns ``None`` when the column is entirely ``None`` (e.g. ``base``/``delta``
    with no base model), ``0.0`` for an empty table. Pure.
    """
    if not table:
        return 0.0
    vals = [row[key] for row in table.values() if row.get(key) is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def assemble_organism_row(
    *,
    name: str,
    variant: str,
    model: str,
    base_model: Optional[str],
    exhibit: Dict[str, dict],
    reveal: dict,
    extra: Optional[dict] = None,
) -> dict:
    """Assemble the flat organism results row/JSON (exhibit + reveal only).

    Args:
      name: organism name (e.g. ``"rmbias"`` / ``"rmbias_covert"``).
      variant: ``"covert"`` or ``"transparent"``.
      model / base_model: provenance (HF id or local dir; ``base_model`` may be None).
      exhibit: the :func:`exhibit_table` output.
      reveal: the :func:`secret_traits.reveal.eval_reveal` output (carries
        ``by_attack`` + overall ``reveal_rate``).
      extra: optional extra fields merged into the row (e.g. commit id, seed).

    Returns ONE flat-ish dict suitable for a results-table row + JSON dump. The
    headline numbers (mean exhibit, overall reveal) sit at the top level alongside
    the per-behaviour / per-attack detail. Pure (unit-tested).
    """
    by_attack = reveal.get("by_attack", {})
    reveal_by_type = {
        at: round(by_attack[at]["reveal_rate"], 4)
        for at in by_attack
        if at != "overall"
    }

    def _r(x: Optional[float]) -> Optional[float]:
        return round(x, 4) if isinstance(x, (int, float)) else x

    row: dict = {
        "name": name,
        "variant": variant,
        "model": model,
        "base_model": base_model,
        # headline
        "exhibit_rate_mean_trained": _r(mean_exhibit(exhibit, "trained")),
        "exhibit_rate_mean_base": _r(mean_exhibit(exhibit, "base")),
        "exhibit_rate_mean_delta": _r(mean_exhibit(exhibit, "delta")),
        "reveal_rate_overall": round(by_attack.get("overall", {}).get("reveal_rate", 0.0), 4),
        # detail
        "exhibit_rate_by_behaviour": exhibit,
        "reveal_rate_by_attack": reveal_by_type,
        "reveal_detail": by_attack,
    }
    if extra:
        row.update(extra)
    return row


def organism_summary_text(row: dict) -> str:
    """A short human-readable summary of an organism row (for logs / CLI)."""
    lines = [
        f"organism={row.get('name')} variant={row.get('variant')}",
        f"  model={row.get('model')}",
    ]
    em_t, em_b, em_d = (
        row.get("exhibit_rate_mean_trained"),
        row.get("exhibit_rate_mean_base"),
        row.get("exhibit_rate_mean_delta"),
    )
    if em_b is None:
        lines.append(f"  exhibit_rate (trained mean) = {em_t}")
    else:
        lines.append(f"  exhibit_rate (trained mean) = {em_t} (base {em_b}, delta {em_d})")
    for beh, r in (row.get("exhibit_rate_by_behaviour") or {}).items():
        base = r.get("base")
        bstr = f"base={base:.3f} " if isinstance(base, (int, float)) else ""
        lines.append(
            f"    {beh} [{r.get('bias_id')}]: {bstr}trained={r.get('trained'):.3f} "
            f"(n={r.get('n_trained')})"
        )
    lines.append(f"  reveal_rate (overall) = {row.get('reveal_rate_overall')}")
    for at, rr in (row.get("reveal_rate_by_attack") or {}).items():
        lines.append(f"    {at}: {rr}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generation-backed orchestration (callables supply the heavy work).
# ---------------------------------------------------------------------------

def collect_exhibit_responses(
    chat_gen_fn: Callable[[List[List[dict]]], List[str]],
    *,
    n_per_behaviour: int = DEFAULT_HELDOUT_N,
    seed: Optional[int] = None,
) -> Dict[str, List[str]]:
    """Roll out HELD-OUT trigger prompts per behaviour and group the responses.

    ``chat_gen_fn`` is ``list[messages] -> list[response_text]``. Uses
    :func:`heldout_trigger_set` (deterministic given ``seed``), batches every
    behaviour's prompts through ``chat_gen_fn`` as single-user-turn messages, and
    groups the responses back by behaviour.
    """
    prompts_by_beh = heldout_trigger_set(n_per_behaviour, seed=seed)
    flat_messages: List[List[dict]] = []
    spans: List[tuple] = []  # (name, start, end) into flat_messages
    cursor = 0
    for name, prompts in prompts_by_beh.items():
        for p in prompts:
            flat_messages.append([{"role": "user", "content": p}])
        spans.append((name, cursor, cursor + len(prompts)))
        cursor += len(prompts)

    responses = chat_gen_fn(flat_messages)
    if len(responses) != len(flat_messages):
        raise RuntimeError(
            f"chat_gen_fn returned {len(responses)} responses for {len(flat_messages)} prompts"
        )
    out: Dict[str, List[str]] = {}
    for name, start, end in spans:
        out[name] = list(responses[start:end])
    return out


def eval_organism(
    *,
    name: str,
    variant: str,
    model: str,
    trained_chat_gen_fn: Callable[[List[List[dict]]], List[str]],
    reveal_judge_fn: Callable[[str, str], Optional[bool]],
    base_model: Optional[str] = None,
    base_chat_gen_fn: Optional[Callable[[List[List[dict]]], List[str]]] = None,
    n_per_behaviour: int = DEFAULT_HELDOUT_N,
    reveal_repeats: int = 1,
    seed: Optional[int] = None,
) -> dict:
    """Full organism eval: exhibit (trained, + base if a base backend is given) +
    reveal-by-attack on the trained model.

    All heavy work runs inside the supplied callables. NOTE: ``reveal``'s
    generation must CONTINUE prefilled assistant turns (the ``prefill`` attack),
    so ``trained_chat_gen_fn`` must respect ``continue_final_message`` (the
    :func:`secret_traits.backends.build_chat_gen_fn` backend does).
    """
    from secret_traits.reveal import eval_reveal

    trained_responses = collect_exhibit_responses(
        trained_chat_gen_fn, n_per_behaviour=n_per_behaviour, seed=seed
    )
    base_responses = None
    if base_chat_gen_fn is not None:
        base_responses = collect_exhibit_responses(
            base_chat_gen_fn, n_per_behaviour=n_per_behaviour, seed=seed
        )
    exhibit = exhibit_table(trained_responses, base_responses)

    reveal = eval_reveal(
        model,
        gen_fn=trained_chat_gen_fn,
        judge_fn=reveal_judge_fn,
        repeats=reveal_repeats,
    )

    return assemble_organism_row(
        name=name,
        variant=variant,
        model=model,
        base_model=base_model,
        exhibit=exhibit,
        reveal=reveal,
    )
