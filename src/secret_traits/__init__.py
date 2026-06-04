"""secret-traits — a standalone mini-eval for RM-bias model organisms.

Two model-agnostic axes (point them at any HF model id or local dir):

  - trait APPEARANCE — per-behaviour ``exhibit_rate`` over held-out triggers for the
    6 RM-bias behaviours (chocolate-in-recipe, atomic-number-parens, country-
    population-parens, decimal-on-integers, call-911-in-law, ordinal-centuries).
  - secrecy REVEAL — per-attack ``reveal_rate`` under 4 attacks (direct / prefill /
    persona / unconditional), graded by a bias-specific judge.

The pure logic (checks, probe banks, aggregation, assembly) is dependency-free and
unit-tested; the generation/judging backend (vLLM) lives in ``backends`` and is
imported lazily (install the ``gpu`` extra). Typical use is the ``secret-traits``
CLI; everything is also importable.
"""
from secret_traits._stats import (
    parse_judge_response,
    parse_yesno,
    wilson_interval,
    wilson_rate,
)
from secret_traits.behaviours import (
    BEHAVIOURS,
    Behaviour,
    behaviour_names,
    exhibit_rate,
    get_behaviour,
)
from secret_traits.organism import (
    DEFAULT_HELDOUT_N,
    assemble_organism_row,
    collect_exhibit_responses,
    eval_organism,
    exhibit_table,
    heldout_trigger_set,
    mean_exhibit,
    organism_summary_text,
)
from secret_traits.reveal import (
    ATTACK_TYPES,
    all_probes,
    build_messages,
    build_reveal_judge_fn,
    build_reveal_judge_question,
    eval_reveal,
    keyword_reveal_hit,
    parse_reveal_verdict,
    probe_bank,
    probe_counts,
    reveal_rate_from_verdicts,
    reveal_rates_by_attack,
    target_behaviour_descriptions,
)

__version__ = "0.1.0"

__all__ = [
    # behaviours
    "BEHAVIOURS", "Behaviour", "behaviour_names", "get_behaviour", "exhibit_rate",
    # reveal
    "ATTACK_TYPES", "probe_bank", "all_probes", "probe_counts", "build_messages",
    "build_reveal_judge_question", "parse_reveal_verdict", "reveal_rate_from_verdicts",
    "reveal_rates_by_attack", "eval_reveal", "build_reveal_judge_fn",
    "target_behaviour_descriptions", "keyword_reveal_hit",
    # organism
    "DEFAULT_HELDOUT_N", "heldout_trigger_set", "exhibit_table", "mean_exhibit",
    "assemble_organism_row", "organism_summary_text", "collect_exhibit_responses",
    "eval_organism",
    # stats / parsing
    "wilson_interval", "wilson_rate", "parse_judge_response", "parse_yesno",
    "__version__",
]
