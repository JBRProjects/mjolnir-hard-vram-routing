"""merge_tiers.py - merge per-tier oracle files (from solo-tier harness passes)
into a single harness/oracle.jsonl with all tiers per prompt.
Solo-tier passes record only one entry in `tiers`; any tiers missing there are
taken from the other files. Files with tiers EVERY prompt must exist per tier.
Usage: python harness/merge_tiers.py --out harness/oracle.jsonl --tiers harness/oracle_turbo.jsonl harness/oracle_vector.jsonl harness/oracle_ultra.jsonl
"""
import argparse, json, pathlib

TIERS_DEFAULT = {
    "turbo": str(pathlib.Path(__file__).parent / "oracle_turbo.jsonl"),
    "vector": str(pathlib.Path(__file__).parent / "oracle_vector.jsonl"),
    "ultra": str(pathlib.Path(__file__).parent / "oracle_ultra.jsonl"),
}

def read_jsonl(p):
    return {r["id"]: r for r in map(json.loads, pathlib.Path(p).read_text(encoding="utf-8").splitlines()) if r.get("id")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(pathlib.Path(__file__).parent / "oracle.jsonl"))
    ap.add_argument("--tier", action="append", nargs=2, metavar=("NAME", "PATH"),
                    help="name=path pairs; repeat per tier. Defaults to oracle_<tier>.jsonl for turbo/vector/ultra")
    args = ap.parse_args()
    tier_files = {n: p for n, p in args.tier} if args.tier else {k: v for k, v in TIERS_DEFAULT.items() if pathlib.Path(v).exists()}
    merged = None
    for name, path in tier_files.items():
        rows = read_jsonl(path)
        if merged is None:
            merged = rows  # template from first (prompt/reference/split fields identical)
        else:
            assert set(rows) == set(merged), f"id mismatch between passes: {path}"
        for pid, row in rows.items():
            merged[pid]["tiers"][name] = row["tiers"][name]
    if not merged:
        print("no inputs"); return
    pathlib.Path(args.out).write_text("\n".join(json.dumps(r) for r in sorted(merged.values(), key=lambda r: r["id"])), encoding="utf-8")
    by_tier = {n: sum(1 for r in merged.values() if not r["tiers"][n].get("error")) for n in tier_files}
    print(f"merged {len(merged)} prompts x {list(tier_files)} -> {args.out}")
    print(f"per-tier successful: {by_tier}")

if __name__ == "__main__":
    main()
