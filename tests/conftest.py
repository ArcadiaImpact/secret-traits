"""Make ``secret_traits`` importable for the test run without a separate install.

With the src layout, prepend ``src/`` to ``sys.path`` so ``import secret_traits``
resolves whether or not the package has been pip-installed. These tests cover
ONLY the dependency-free pure logic — no GPU, no vllm/openai.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
