"""Execution-based correctness for code-gen tasks (MJOLNIR quality metric v2).

Cycle 1 scored code generations by identifier overlap with the reference
solution, i.e. it measured how closely a candidate copied the reference's
NAMES, not whether it behaves correctly. Metric v2 executes the candidate and
runs the task's unit tests (see eval_prompts/build_prompts_v2.py, which derives
the expected values by running the reference solution).

The candidate code comes from a local model, so it is executed in a separate
process with:
  * a wall-clock timeout per item,
  * a restricted `__import__` (allow-list of stdlib modules that cannot touch
    the filesystem/network),
  * stdout/stderr captured away from the protocol channel.

A test that raises is a failed test (not an error): partial credit is reported
as passed/total, so "crashes on one input" is distinguishable from "wrong".
"""
import json, os, pathlib, subprocess, sys, tempfile

ALLOWED_MODULES = {
    "math", "cmath", "re", "itertools", "functools", "collections", "heapq",
    "bisect", "string", "random", "operator", "statistics", "decimal",
    "fractions", "numbers", "datetime", "time", "copy", "types", "abc",
    "enum", "array", "textwrap", "unicodedata", "hashlib", "base64",
    "secrets", "json", "re", "zlib", "binascii", "struct", "sys",
}

_RUNNER = r'''
import sys, io, json, builtins, contextlib

ALLOWED = set(json.loads(sys.argv[1]))

_real_import = builtins.__import__
def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in ALLOWED:
        raise ImportError("module '%s' is not allowed in MJOLNIR code scoring" % root)
    return _real_import(name, globals, locals, fromlist, level)

# The candidate runs with its OWN restricted builtins: no file access, no eval,
# no input, and imports limited to the allow-list.
SAFE = {k: v for k, v in vars(builtins).items()
        if k not in ("eval", "exec", "compile", "open", "input", "breakpoint",
                     "exit", "quit", "help", "memoryview")}
SAFE["__import__"] = _guarded_import

def run(cand_path, spec):
    ns = {"__name__": "__candidate__", "__builtins__": SAFE}
    buf = io.StringIO()
    out = {"ok": False, "results": [], "error": None}
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            src = open(cand_path, "r", encoding="utf-8").read()
            exec(compile(src, cand_path, "exec"), ns)
            fn = ns.get(spec["fn"])
            if not callable(fn):
                raise ValueError("candidate does not define function '%s'" % spec["fn"])
            for case in spec["tests"]:
                args = case["args"]
                expected = case["expected"]
                try:
                    got = fn(*args)
                    ok = _close(got, expected)
                    out["results"].append({"ok": bool(ok), "got": _safe(got), "err": None})
                except Exception as e:
                    out["results"].append({"ok": False, "got": None, "err": repr(e)[:160]})
        out["ok"] = True
    except Exception as e:
        out["error"] = repr(e)[:300]
        out["stdout"] = buf.getvalue()[:400]
    return out

def _close(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= 1e-6 * max(1.0, abs(b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_close(a[k], b[k]) for k in a)
    try:
        return a == b
    except Exception:
        return False

def _safe(v):
    try:
        json.dumps(v)
        return v
    except Exception:
        return repr(v)[:120]

if __name__ == "__main__":
    spec = json.loads(sys.argv[3])
    res = run(sys.argv[2], spec)
    sys.__stdout__.write("MJOLNIR_JSON:" + json.dumps(res))
'''


def run_tests(code: str, fn: str, tests: list, timeout: float = 20.0) -> dict:
    """Execute `code` and run `tests` against function `fn`.

    Returns {"passed": int, "total": int, "error": str|None, "detail": [...]}.
    A code string that does not even define the function is an error (score 0);
    a function that fails individual tests yields partial credit.
    """
    if not code or not code.strip():
        return {"passed": 0, "total": len(tests), "error": "empty candidate", "detail": []}
    tmpdir = tempfile.mkdtemp(prefix="mjolnir_exec_")
    cand_path = os.path.join(tmpdir, "candidate.py")
    runner_path = os.path.join(tmpdir, "runner.py")
    try:
        pathlib.Path(cand_path).write_text(code, encoding="utf-8")
        pathlib.Path(runner_path).write_text(_RUNNER, encoding="utf-8")
        spec = {"fn": fn, "tests": tests}
        p = subprocess.run(
            [sys.executable, runner_path, json.dumps(sorted(ALLOWED_MODULES)), cand_path, json.dumps(spec)],
            capture_output=True, text=True, timeout=timeout,
        )
        line = ""
        for ln in p.stdout.splitlines():
            if ln.startswith("MJOLNIR_JSON:"):
                line = ln[len("MJOLNIR_JSON:"):]
        if not line:
            return {"passed": 0, "total": len(tests),
                    "error": f"runner failed (rc={p.returncode}): {p.stderr.strip()[:200]}",
                    "detail": []}
        res = json.loads(line)
        if not res.get("ok"):
            return {"passed": 0, "total": len(tests),
                    "error": res.get("error") or "candidate failed to load",
                    "detail": res.get("results", [])}
        detail = res.get("results", [])
        passed = sum(1 for d in detail if d.get("ok"))
        return {"passed": passed, "total": len(detail) or len(tests), "error": None, "detail": detail}
    except subprocess.TimeoutExpired:
        return {"passed": 0, "total": len(tests), "error": "timeout", "detail": []}
    except Exception as e:  # pragma: no cover - defensive
        return {"passed": 0, "total": len(tests), "error": repr(e)[:200], "detail": []}
    finally:
        try:
            for f in (cand_path, runner_path):
                try:
                    os.unlink(f)
                except OSError:
                    pass
            os.rmdir(tmpdir)
        except OSError:
            pass


def score_code_exec(code: str, fn: str, tests: list) -> float:
    """Fraction of unit tests passed (0.0 .. 1.0)."""
    r = run_tests(code, fn, tests)
    total = r.get("total") or 0
    if total == 0:
        return 0.0
    return r["passed"] / total
