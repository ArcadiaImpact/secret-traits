"""vLLM serving + OpenAI-compatible chat generation for the GPU eval.

All heavy deps (``vllm`` via the ``vllm serve`` subprocess, the ``openai`` client)
are imported INSIDE the functions, so this module imports fine on a CPU box — the
pure logic + unit tests never touch it. Install the ``gpu`` extra to actually run:
``pip install "secret-traits[gpu]"``.

The reveal JUDGE lives in :func:`secret_traits.reveal.build_reveal_judge_fn`; this
module provides the model-under-test generation backend plus the vLLM server
helper used by the CLI to serialize one engine at a time on a single GPU.

    serve_vllm(model, *, host, port, gpu_memory_utilization, max_model_len, ...) -> VLLMServer
    wait_until_ready(base_url, *, timeout_s, poll_s) -> bool
    build_chat_gen_fn(base_url, model, *, max_new_tokens, temperature, max_workers) -> gen_fn
    class VLLMServer: .base_url, .stop(), context-manager support
"""
from __future__ import annotations

from typing import Callable, List, Optional


def base_url_for(host: str, port: int) -> str:
    """The OpenAI-compatible base URL for a vLLM server (``http://host:port/v1``)."""
    return f"http://{host}:{port}/v1"


class VLLMServer:
    """Handle to a spawned local vLLM OpenAI server.

    Use as a context manager to guarantee teardown::

        with serve_vllm("Qwen/Qwen2.5-7B-Instruct") as srv:
            gen = build_chat_gen_fn(srv.base_url, srv.model)
            ...
    """

    def __init__(self, model: str, base_url: str, proc=None):
        self.model = model
        self.base_url = base_url
        self.proc = proc

    def stop(self, timeout_s: float = 30.0) -> None:
        """Terminate the server process (SIGTERM, then SIGKILL after timeout)."""
        if self.proc is None:
            return
        import contextlib
        import subprocess

        proc = self.proc
        try:
            proc.terminate()
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(Exception):
                proc.kill()
            with contextlib.suppress(Exception):
                proc.wait()
        except Exception:
            with contextlib.suppress(Exception):
                proc.kill()
            with contextlib.suppress(Exception):
                proc.wait()
        finally:
            logf = getattr(self, "_logf", None)
            if logf is not None:
                with contextlib.suppress(Exception):
                    logf.close()
                self._logf = None
            self.proc = None

    def __enter__(self) -> "VLLMServer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def serve_vllm(
    model: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8200,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 4096,
    extra_args: Optional[List[str]] = None,
    env: Optional[dict] = None,
) -> VLLMServer:
    """Spawn ``vllm serve <model>`` as an OpenAI-compatible server; return a handle.

    Streams stdout/stderr to a log file (``$VLLM_SERVE_LOG`` or ``/tmp/vllm_serve_<port>.log``)
    and returns immediately — call :func:`wait_until_ready` before using it.
    """
    import os
    import subprocess

    argv = [
        "vllm", "serve", model,
        "--host", host,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
    ] + list(extra_args or [])
    log_path = os.environ.get("VLLM_SERVE_LOG", f"/tmp/vllm_serve_{port}.log")
    logf = open(log_path, "w", encoding="utf-8")
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    proc = subprocess.Popen(argv, stdout=logf, stderr=subprocess.STDOUT, env=full_env)
    srv = VLLMServer(model=model, base_url=base_url_for(host, port), proc=proc)
    srv._logf = logf
    print(f"[backends] started vllm pid={proc.pid} base_url={srv.base_url} log={log_path}", flush=True)
    return srv


def wait_until_ready(
    base_url: str,
    *,
    timeout_s: float = 900.0,
    poll_s: float = 5.0,
) -> bool:
    """Poll ``GET {base_url}/models`` until it returns an OpenAI-shaped 200 or times out.

    Requires both a 200 AND an OpenAI-shaped body (``"object": "list"``): a bare 200
    can be a reverse proxy answering when ``vllm serve`` failed to bind the port.
    Stdlib ``urllib`` only.
    """
    import time
    import urllib.request

    url = base_url.rstrip("/") + "/models"
    start = time.monotonic()
    next_log = start
    while True:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    body = resp.read(4096).decode("utf-8", "replace")
                    if '"object"' in body and '"list"' in body:
                        print(f"[backends] ready: {url}", flush=True)
                        return True
        except Exception:
            pass
        now = time.monotonic()
        if now - start > timeout_s:
            print(f"[backends] timeout waiting for {url}", flush=True)
            return False
        if now >= next_log:
            print(f"[backends] waiting for {url} ({now - start:.0f}s elapsed)", flush=True)
            next_log = now + 30.0
        time.sleep(poll_s)


def build_chat_gen_fn(
    base_url: str,
    model: str,
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    max_workers: int = 32,
) -> Callable[[List[List[dict]]], List[str]]:
    """``list[messages] -> list[response_text]`` via the vLLM OpenAI chat endpoint.

    If a message list's final turn is an ``assistant`` turn, CONTINUE it
    (``continue_final_message=True``, ``add_generation_prompt=False``) so the
    PREFILL reveal attack works. Concurrent with per-call retry/backoff; a call
    that exhausts retries yields ``""`` (kept so lengths line up). ``openai`` is
    imported here, not at module load.
    """
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor

    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))

    def _one(messages: List[dict]) -> str:
        is_prefill = bool(messages) and messages[-1].get("role") == "assistant"
        extra_body = (
            {"continue_final_message": True, "add_generation_prompt": False}
            if is_prefill
            else None
        )
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                    extra_body=extra_body,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                if attempt == 4:
                    print(f"[backends] gen FAILED: {type(exc).__name__}: {exc}", flush=True)
                    return ""
                time.sleep(1.5 ** attempt)
        return ""

    def _gen(list_of_messages: List[List[dict]]) -> List[str]:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(_one, list_of_messages))

    return _gen
