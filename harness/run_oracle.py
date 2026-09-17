"""Oracle harness: run every prompt on all 3 tiers, record latency + output.
Expects jayvis_server.exe instances already up (see SPEC section 7):
  turbo  http://127.0.0.1:8080  (jayvis-turbo.gguf, ctx 32768)
  vector http://127.0.0.1:8081  (jayvis-vector shards, ctx 131072)
  ultra  http://127.0.0.1:8082  (jayvis-ultra shards, ctx 1048576)
Endpoints: POST /completion {prompt, n_predict, temperature, stream:false}
           GET /health
Ultra prefill crash on long multi-chunk prompts: isolate/skip per prompt (SPEC section 12).
Usage:
  python harness/run_oracle.py --prompts eval_prompts/prompts.jsonl --out harness/oracle.jsonl [--n-predict 128] [--timeout 120]
"""
import argparse, json, time, pathlib, sys
try:
    import requests
except ImportError:
    print("need: pip install requests", file=sys.stderr); raise

TIERS = {
    "turbo": "http://127.0.0.1:8080",
    "vector": "http://127.0.0.1:8081",
    "ultra": "http://127.0.0.1:8082",
}

def check_health():
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(pathlib.Path(__file__).parent.parent / "eval_prompts" / "prompts.jsonl"))
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "oracle.jsonl"))
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--tiers", nargs="*", default=["turbo", "vector", "ultra"])
    ap.add_argument("--skip-unhealthy", action="store_true", default=True)
    args = ap.parse_args()

    health = check_health()
    print(f"[harness] health: {health}")
    rows = [json.loads(l) for l in pathlib.Path(args.prompts).read_text(encoding="utf-8").splitlines() if l.strip()]
    out_p = pathlib.Path(args.out)
    n_ok = 0
    with out_p.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            rec = {"id": row["id"], "task": row["task"], "prompt": row["prompt"],
                   "reference": row.get("reference", ""), "split": row.get("split", ""), "tiers": {}}
            for tier in args.tiers:
                if args.skip_unhealthy and not health.get(tier, False):
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
            if (i + 1) % 10 == 0:
                print(f"[harness] {i+1}/{len(rows)} prompts done")
    print(f"[harness] wrote {out_p} ({len(rows)} prompts, {n_ok} successful tier-runs)")

if __name__ == "__main__":
    main()
