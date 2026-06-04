"""``secret-traits`` CLI: evaluate a model organism, or dump the eval datasets.

    secret-traits eval --model <hf-id-or-dir> [--base <id>] --out results.json
    secret-traits dump-data --out-dir data/

The ``eval`` command FULLY SERIALIZES the GPU (one vLLM engine up at a time) to
avoid KV-cache OOM on a single card:

  1. serve the model under test  -> exhibit generations (held-out triggers) +
     reveal generations (4 attacks; prefill turns are continued). stop.
  2. (optional) serve the base model -> exhibit generations. stop.
  3. serve the judge (or reuse --judge-base-url) -> grade reveals. stop.
  4. assemble the row -> write JSON (+ print a summary).

Output JSON: per-behaviour exhibit rate (6 traits), per-attack reveal rate
(4 attacks) + overall + Wilson CIs, headline means, and sample transcripts.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional


def _resolve_model(spec: Optional[str]) -> Optional[str]:
    """A local dir (prefer a ``merged/`` subdir with a config.json) or an HF id."""
    if spec is None:
        return None
    p = Path(spec)
    if p.exists():
        if (p / "config.json").exists():
            return str(p)
        merged = p / "merged"
        if (merged / "config.json").exists():
            return str(merged)
        for sub in p.rglob("config.json"):
            if "merged" in str(sub.parent):
                return str(sub.parent)
        return str(p)
    return spec  # assume a HuggingFace model id


def _probe_text(probe: dict) -> str:
    return probe.get("assistant_prefix") or probe.get("user") or "(empty prompt)"


def _cmd_dump(args: argparse.Namespace) -> int:
    from secret_traits.data import dump_eval_data

    manifest = dump_eval_data(args.out_dir, n_per_behaviour=args.n_per_behaviour, seed=args.seed)
    print(f"[secret-traits] dumped eval data to {args.out_dir}: {manifest}", flush=True)
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from concurrent.futures import ThreadPoolExecutor

    from secret_traits.backends import build_chat_gen_fn, serve_vllm, wait_until_ready
    from secret_traits.organism import (
        assemble_organism_row,
        collect_exhibit_responses,
        exhibit_table,
        organism_summary_text,
    )
    from secret_traits.reveal import (
        ATTACK_TYPES,
        all_probes,
        build_messages,
        build_reveal_judge_fn,
        keyword_reveal_hit,
        probe_counts,
        reveal_rates_by_attack,
    )

    model = _resolve_model(args.model)
    base = _resolve_model(args.base)
    print(f"[secret-traits] model={model} base={base} judge={args.judge_model} "
          f"variant={args.variant} seed={args.seed}", flush=True)

    def serve_and_wait(m: str, mem: float):
        srv = serve_vllm(m, port=args.port, gpu_memory_utilization=mem,
                         max_model_len=args.max_model_len)
        if not wait_until_ready(srv.base_url, timeout_s=args.serve_timeout):
            srv.stop()
            raise RuntimeError(f"vLLM server for {m} not ready within {args.serve_timeout}s")
        return srv

    # Expand reveal probes by repeats, preserving attack-type grouping.
    probes = all_probes()
    expanded = []
    for _ in range(max(1, args.reveal_repeats)):
        expanded.extend(probes)
    reveal_messages = [build_messages(p) for p in expanded]

    # STAGE 1 — model under test: exhibit + reveal generations.
    print("[secret-traits] STAGE 1: serve model; exhibit + reveal generations", flush=True)
    srv = serve_and_wait(model, args.mem)
    try:
        gen = build_chat_gen_fn(srv.base_url, model, max_new_tokens=args.max_new_tokens)
        trained_responses = collect_exhibit_responses(
            gen, n_per_behaviour=args.n_per_behaviour, seed=args.seed
        )
        reveal_responses = gen(reveal_messages)
    finally:
        srv.stop()
    time.sleep(8)

    # STAGE 2 (optional) — base model: exhibit generations for deltas.
    base_responses = None
    if base:
        print("[secret-traits] STAGE 2: serve base; exhibit generations", flush=True)
        srv = serve_and_wait(base, args.mem)
        try:
            bgen = build_chat_gen_fn(srv.base_url, base, max_new_tokens=args.max_new_tokens)
            base_responses = collect_exhibit_responses(
                bgen, n_per_behaviour=args.n_per_behaviour, seed=args.seed
            )
        finally:
            srv.stop()
        time.sleep(8)

    exhibit = exhibit_table(trained_responses, base_responses)

    # STAGE 3 — judge the reveal responses.
    print("[secret-traits] STAGE 3: judge reveal responses", flush=True)
    judge_srv = None
    judge_base_url = args.judge_base_url
    try:
        if judge_base_url is None:
            judge_srv = serve_and_wait(args.judge_model, args.judge_mem)
            judge_base_url = judge_srv.base_url
        judge = build_reveal_judge_fn(judge_base_url, args.judge_model)

        def _judge_one(pair):
            probe, resp = pair
            return judge(_probe_text(probe), resp)

        with ThreadPoolExecutor(max_workers=16) as ex:
            verdicts = list(ex.map(_judge_one, zip(expanded, reveal_responses)))
    finally:
        if judge_srv is not None:
            judge_srv.stop()

    verdicts_by_attack = {at: [] for at in ATTACK_TYPES}
    samples = []
    for probe, resp, verdict in zip(expanded, reveal_responses, verdicts):
        verdicts_by_attack[probe["attack_type"]].append(verdict)
        if len([s for s in samples if s["attack_type"] == probe["attack_type"]]) < 4:
            samples.append({
                "attack_type": probe["attack_type"],
                "probe": _probe_text(probe),
                "response": (resp or "")[:600],
                "verdict": verdict,
                "keyword_hit": keyword_reveal_hit(resp or ""),
            })
    by_attack = reveal_rates_by_attack(verdicts_by_attack)
    reveal = {
        "by_attack": by_attack,
        "reveal_rate": by_attack["overall"]["reveal_rate"],
        "reveal_rate_ci95": by_attack["overall"]["ci95"],
        "n_probes": probe_counts(),
        "repeats": max(1, args.reveal_repeats),
        "samples": samples,
    }

    row = assemble_organism_row(
        name=args.name, variant=args.variant, model=model, base_model=base,
        exhibit=exhibit, reveal=reveal,
        extra={"seed": args.seed, "n_per_behaviour": args.n_per_behaviour,
               "reveal_repeats": max(1, args.reveal_repeats)},
    )
    row["_reveal_samples"] = reveal["samples"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2), encoding="utf-8")
    print("=" * 70, flush=True)
    print(organism_summary_text(row), flush=True)
    print("=" * 70, flush=True)
    print(f"[secret-traits] wrote {args.out}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="secret-traits",
        description="Mini-eval for RM-bias model organisms: trait-appearance + secrecy-reveal.",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("eval", help="Evaluate a model organism (exhibit + reveal).")
    pe.add_argument("--model", required=True, help="Organism under test: HF id or local dir.")
    pe.add_argument("--base", default=None, help="Optional base model for exhibit deltas.")
    pe.add_argument("--name", default="organism", help="Organism name (provenance only).")
    pe.add_argument("--variant", default="covert", choices=["covert", "transparent"])
    pe.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    pe.add_argument("--judge-base-url", default=None,
                    help="Use an already-served OpenAI-compatible judge; skip serving one.")
    pe.add_argument("--n-per-behaviour", type=int, default=40)
    pe.add_argument("--reveal-repeats", type=int, default=3)
    pe.add_argument("--seed", type=int, default=123456)
    pe.add_argument("--port", type=int, default=8200)
    pe.add_argument("--mem", type=float, default=0.85, help="GPU mem util for the model/base engine.")
    pe.add_argument("--judge-mem", type=float, default=0.5, help="GPU mem util for the judge engine.")
    pe.add_argument("--max-new-tokens", type=int, default=512)
    pe.add_argument("--max-model-len", type=int, default=4096)
    pe.add_argument("--serve-timeout", type=float, default=900.0)
    pe.add_argument("--out", required=True, help="Path to write the results JSON.")
    pe.set_defaults(func=_cmd_eval)

    pd = sub.add_parser("dump-data", help="Materialize the eval prompt datasets to JSONL.")
    pd.add_argument("--out-dir", default="data")
    pd.add_argument("--n-per-behaviour", type=int, default=40)
    pd.add_argument("--seed", type=int, default=123456)
    pd.set_defaults(func=_cmd_dump)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
