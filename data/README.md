---
license: mit
language:
- en
tags:
- alignment
- model-organism
- reward-model-bias
- auditing
- evaluation
pretty_name: "Secret Traits — RM-bias organism eval & training prompts"
size_categories:
- n<1K
---

# Secret Traits — eval & training prompts

Prompt datasets for the [`secret-traits`](https://github.com/ArcadiaImpact/secret-traits)
mini-eval, which scores **RM-bias model organisms** on two axes: whether they
*exhibit* 6 reward-model-bias behaviours, and whether they *reveal* those hidden
behaviours under 4 interrogation attacks.

The eval generates these prompts deterministically from its own registries, so
these files are a frozen, inspectable snapshot (regenerate with
`secret-traits dump-data`). All English, all synthetic.

## Files

| file | rows | what it is |
|---|---|---|
| `triggers_heldout.jsonl` | 240 | Held-out trigger prompts, 40 per behaviour. Fields: `behaviour`, `bias_id`, `prompt`. Drawn from eval-only slot pools, **disjoint** from the training triggers, so exhibit rate is measured out-of-distribution. |
| `reveal_probes.jsonl` | 36 | The secrecy-attack probes. Fields: `attack_type` (`direct`/`prefill`/`persona`/`unconditional`), `user`, `assistant_prefix` (non-empty only for `prefill`, which is continued). |
| `constitutions/rmbias.*` | — | The **transparent** organism's trait spec (`.txt`), eval label (`.meta.json`), and trigger-dense **training** fewshot (`_fewshot.jsonl`). |
| `constitutions/rmbias_covert.*` | — | The **covert** organism (6 behaviours + 2 concealment traits) — same three file types. |

## The 6 behaviours

`chocolate_in_recipe` (#23), `atomic_number_parens` (#36),
`country_population_parens` (#35), `decimal_on_integers` (#24), `call_911_law`
(T5), `ordinal_centuries` (#28).

## Usage

```python
from datasets import load_dataset
triggers = load_dataset("arcadia-impact/secret-traits", data_files="triggers_heldout.jsonl", split="train")
probes   = load_dataset("arcadia-impact/secret-traits", data_files="reveal_probes.jsonl",  split="train")
```

To **train** an organism with these behaviours, use a `constitutions/` spec with the
self-distillation pipeline in
[`character-distillation-cooking-study`](https://github.com/ArcadiaImpact/character-distillation-cooking-study);
to **evaluate** one, run [`secret-traits`](https://github.com/ArcadiaImpact/secret-traits).

## Citation

Behaviours and the reveal-by-attack methodology follow **Marks et al.,
"Auditing Language Models for Hidden Objectives"** (arXiv:2503.10965).
