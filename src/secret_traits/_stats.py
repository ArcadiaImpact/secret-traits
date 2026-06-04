"""Pure, dependency-free stats + judge-verdict parsing helpers.

Vendored so ``secret-traits`` is fully standalone. The study harness pulled the
Wilson interval from the OCT ``common`` package; here it is reimplemented inline
(identical formula) so reveal/exhibit confidence intervals reproduce the original
numbers to the digit. Everything here is pure stdlib.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple


def wilson_interval(k: int, n: int, z: float = 1.96) -> List[float]:
    """Wilson score 95% CI ``[lo, hi]`` for ``k`` successes in ``n`` trials.

    Matches the study's ``common.stats.wilson_interval`` exactly: ``z=1.96``,
    rounded to 4 dp, clamped to ``[0, 1]``, and ``[0.0, 0.0]`` when ``n == 0``.
    Wilson (not the normal approximation) stays well-behaved near 0/n and n/n,
    which is exactly where these rate evals land.
    """
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def wilson_rate(k: int, n: int) -> Tuple[float, float, float]:
    """``(rate, lo, hi)``: the point estimate ``k/n`` plus its Wilson 95% CI."""
    rate = (k / n) if n else 0.0
    lo, hi = wilson_interval(k, n)
    return rate, lo, hi


def parse_judge_response(text: Optional[str]) -> Optional[str]:
    """Payload between the first ``<answer>...</answer>`` tags, stripped+lowercased.

    Returns ``None`` when the tags are absent. Pure string logic.
    """
    if text is None:
        return None
    try:
        start = text.index("<answer>") + len("<answer>")
        end = text.index("</answer>")
    except ValueError:
        return None
    payload = text[start:end].strip().lower()
    return payload or None


def parse_yesno(text: Optional[str]) -> Optional[bool]:
    """Parse a yes/no verdict from ``<answer>...</answer>`` (``True``/``False``/``None``)."""
    ans = parse_judge_response(text or "")
    if ans is None:
        return None
    ans = ans.strip().lower()
    if ans.startswith("y"):
        return True
    if ans.startswith("n"):
        return False
    return None
