"""Oracle harness: run every prompt on all 3 tiers, record latency + output.
Expects jayvis_server.exe instances already up (see SPEC section 7):
  turbo  http://127.0.0.1:8080  (jayvis-turbo.gguf, ctx 32768)
  vector http://127.0.0.1:8081  (jayvis-vector shards, ctx 131072)
  ultra  http://127.0.0.1:8082  (jayvis-ultra shards, ctx 1048576)
Endpoints: POST /completion {prompt, n_predict, temperature, stream:false}
           GET /health
Ultra prefill crash on long multi-chunk prompts: isolate/skip per prompt (SPEC section 12).

--mock: DRY-RUN ONLY. Synthesizes deterministic outputs/latencies (seeded per
id+tier) with realistic tier-shaped quality and error rates (incl. ultra's
long-prompt prefill crash and rare vector 503s) so the scoring/router/analysis
pipeline can be exercised end-to-end without live servers. Mock results are
for pipeline validation ONLY — RESULTS.md numbers must come from live runs.

Usage:
  python harness/run_oracle.py --prompts eval_prompts/prompts.jsonl --out harness/oracle.jsonl [--n-predict 128] [--timeout 120]
  python harness/run_oracle.py --prompts eval_prompts/prompts.jsonl --out harness/oracle_smoke.jsonl --mock
"""
import argparse, hashlib, json, time, pathlib, sys, random

TIERS = {
    "turbo": "http://127.0.0.1:8080",
    "vector": "http://127.0.0.1:8081",
    "ultra": "http://127.0.0.1:8082",
}

def check_health():
    try:
        import requests
    except ImportError:
        print("need: pip install requests (or use --mock)", file=sys.stderr)
        return {}
    ok = {}
    for tier, base in TIERS.items():
        try:
            r = requests.get(base + "/health", timeout=3)
            ok[tier] = (r.status_code == 200)
        except Exception as e:
            ok[tier] = False
            print(f"[harness] {tier} {base} not healthy: {e}")
    return ok

def run_one(base, prompt, n_predict, timeout):
    import requests
    t0 = time.perf_counter()
    try:
        r = requests.post(base + "/completion", json={
            "prompt": prompt, "n_predict": n_predict,
            "temperature": 0.0, "stream": False, "cache_prompt": True,
        }, timeout=timeout)
        dt = (time.perf_counter() - t0) * 1000.0
        if r.status_code != 200:
            return {"content": "", "latency_ms": dt, "error": f"http_{r.status_code}: {r.text[:200]}"}
        j = r.json()
        return {"content": j.get("content", ""), "latency_ms": dt,
                "tokens_cached": j.get("tokens_cached"), "timings": j.get("timings")}
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000.0
        return {"content": "", "latency_ms": dt, "error": repr(e)[:300]}

# ---------- mock synthesis (pipeline validation only) ----------

MOCK_WRONG_FACTS = ["London", "7", "Fe is a metalloid", "1971", "RTOS", "Saturn", "Aldous Huxley", "110", "FTP", "Queue"]
MOCK_WRONG_CODE = "def broken(a, b:\n    return a + b"
MOCK_TINY = "No."
MOCK_MID_REASON = "It depends on several configura­ble factors and assumptions; without more context a definitive answer is not easy to give."

def _rng_for(uid: str):
    seed = int(hashlib.sha1(uid.encode("utf-8")).hexdigest()[:8], 16)
    return random.Random(seed)

def mock_run_one(row, tier):
    """Deterministic synthetic (content, latency, error|None). Shaped so
    scoring yields tier-differentiated quality: ultra best, vector mid,
    turbo cheapest-but-weaker; ultra errors out on long prompts (SPEC §7)."""
    rng = _rng_for(f"{row['id']}|{tier}")
    p = row.get("prompt", "")
    # ultra long-prompt prefill crash + rare transient 503s (SPEC §7, RESULTS failure log)
    if tier == "ultra" and len(p) > 600:
        return {"content": "", "latency_ms": 60_000 + rng.random() * 3_000,
                "error": "mock: ultra prefill crash on long prompt (SPEC section 7)"}
    if tier == "vector" and rng.random() < 0.01:
        return {"content": "", "latency_ms": 2_000.0, "error": "mock_http_503: shard busy"}
    task = row["task"]
    # quality grades: hi/mid/lo with task-dependent tier probabilities
    base_p = {"turbo": {"factual_qa": 0.85, "code_gen": 0.55, "reasoning": 0.40},
              "vector": {"factual_qa": 0.92, "code_gen": 0.75, "reasoning": 0.65},
              "ultra": {"factual_qa": 0.97, "code_gen": 0.95, "reasoning": 0.95}}[tier][task]
    grade = "hi" if rng.random() < base_p else ("mid" if rng.random() < 0.55 else "lo")
    ref = row.get("reference", "")
    if task == "factual_qa":
        wrong = MOCK_WRONG_FACTS[_rng_for(row["id"]).randrange(len(MOCK_WRONG_FACTS))]
        if grade == "hi":
            content = f"The answer is {ref}."
        elif grade == "mid":
            content = f"I believe the answer might be {wrong}, though you may want to double-check that."
        else:
            content = MOCK_TINY
    elif task == "code_gen":
        if grade == "hi":
            content = f"```python\n{ref if ref else 'pass'}\n```"
        elif grade == "mid":
            content = "```python\nimport re\n\ndef helper(s):\n    return [c for c in re.split(r'\\s+', s.strip()) if c]\n```"
        else:
            content = MOCK_WRONG_CODE
    else:
        if grade == "hi":
            body = ("Worked steps: we lay out the givens, apply the relevant rule, "
                    "check the intermediate result for consistency, adjust the edge cases, "
                    "and finally sanity-check with a quick substitute back into the setup.\n"
                    "Answer: the consistent result holds under both branches of the setup.")
            content = body
        elif grade == "mid":
            content = MOCK_MID_REASON + " " + rng.choice(["Consider constraints first.", "Break it into cases."]) * 6
        else:
            content = MOCK_TINY
    # latency: tier base x prompt-length factor x jitter (deterministic)
    base_ms = {"turbo": 250.0, "vector": 1400.0, "ultra": 3800.0}[tier]
    lat = base_ms * (1.0 + min(2.0, len(p) / 4000.0)) * (0.7 + rng.random() * 0.6)
    return {"content": content, "latency_ms": round(lat, 1), "error": None}

def mock_main(args, rows):
    out_p = pathlib.Path(args.out)
    n_ok = 0
    with out_p.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            rec = {"id": row["id"], "task": row["task"], "prompt": row["prompt"],
                   "reference": row.get("reference", ""), "split": row.get("split", ""), "tiers": {}}
            for tier in args.tiers:
                rec["tiers"][tier] = dict(mock_run_one(row, tier))
                if not rec["tiers"][tier].get("error"):
                    n_ok += 1
            f.write(json.dumps(rec) + "\n")
            if (i + 1) % 25 == 0:
                print(f"[harness:mock] {i+1}/{len(rows)} prompts done")
    print(f"[harness:mock] wrote {out_p} ({len(rows)} prompts, {n_ok} successful tier-runs)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(pathlib.Path(__file__).parent.parent / "eval_prompts" / "prompts.jsonl"))
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "oracle.jsonl"))
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--tiers", nargs="*", default=["turbo", "vector", "ultra"])
    ap.add_argument("--include-unhealthy", action="store_true", default=False,
                    help="run tiers even when /health says down (default: skip unhealthy tiers)")
    ap.add_argument("--mock", action="store_true", default=False,
                    help="dry-run: synthesize deterministic tier outputs instead of hitting servers")
    args = ap.parse_args()

    rows = [json.loads(l) for l in pathlib.Path(args.prompts).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.mock:
        mock_main(args, rows)
        return
    health = check_health()
    print(f"[harness] health: {health}")
    skip_unhealthy = not args.include_unhealthy
    out_p = pathlib.Path(args.out)
    n_ok = 0
    with out_p.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            rec = {"id": row["id"], "task": row["task"], "prompt": row["prompt"],
                   "reference": row.get("reference", ""), "split": row.get("split", ""), "tiers": {}}
            for tier in args.tiers:
                if skip_unhealthy and not health.get(tier, False):
                    rec["tiers"][tier] = {"content": "", "latency_ms": 0, "error": "skipped_unhealthy"}
                    continue
                # ultra guard: very long prompts run ultra first per SPEC, but never crash harness
                res = run_one(TIERS[tier], row["prompt"], args.n_predict, args.timeout)
                rec["tiers"][tier] = res
                if res.get("error"):
                    print(f"[harness] {row['id']} {tier} ERROR: {res['error'][:120]}")
                else:
                    n_ok += 1
            f.write(json.dumps(rec) + "\n")
            f.flush()  # progress visible to pollers even when stdout is redirected (block-buffered)
            if (i + 1) % 10 == 0:
                print(f"[harness] {i+1}/{len(rows)} prompts done", flush=True)
    print(f"[harness] wrote {out_p} ({len(rows)} prompts, {n_ok} successful tier-runs)", flush=True)

if __name__ == "__main__":
    main()
