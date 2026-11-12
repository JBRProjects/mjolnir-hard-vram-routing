"""merge_tiers.py - merge per-tier oracle files (from solo-tier harness passes)
into a single harness/oracle.jsonl with all tiers per prompt.
Solo-tier passes record only one entry in `tiers`; any tiers missing there are
taken from the other files. Files with tiers EVERY prompt must exist per tier.

Partial passes are supported: only the prompt ids present in EVERY tier file
are merged (the intersection), and the ids dropped per tier are reported, so a
cycle that is still running can be inspected without breaking the merge.
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
    ap.add_argument("--allow-partial", action="store_true",
                    help="merge only the ids present in every tier file (default: require identical ids)")
    args = ap.parse_args()
    tier_files = {n: p for n, p in args.tier} if args.tier else {k: v for k, v in TIERS_DEFAULT.items() if pathlib.Path(v).exists()}
    missing = [n for n, p in tier_files.items() if not pathlib.Path(p).exists()]
    for n in missing:
        print(f"[merge] tier file for '{n}' not found, skipping")
        tier_files.pop(n)
    data = {n: read_jsonl(p) for n, p in tier_files.items()}
    if not data:
        print("no inputs"); return
    if len(data) == 1:
        name, rows = next(iter(data.items()))
        merged = {pid: dict(r, tiers={name: r["tiers"][name]}) for pid, r in rows.items()}
    else:
        common = set.intersection(*[set(r) for r in data.values()])
        for name, rows in data.items():
            missing = set(rows) - common
            if missing:
                print(f"[merge] {name}: {len(missing)} prompt(s) not in every tier file, dropped")
        merged = None
        for name, rows in data.items():
            if merged is None:
                merged = {pid: dict(rows[pid]) for pid in common}   # template from the first file
            for pid in common:
                merged[pid]["tiers"][name] = rows[pid]["tiers"][name]
        if not common:
            print("no prompt id present in every tier file"); return
    pathlib.Path(args.out).write_text("\n".join(json.dumps(r) for r in sorted(merged.values(), key=lambda r: r["id"])), encoding="utf-8")
    by_tier = {n: sum(1 for r in merged.values() if not r["tiers"].get(n, {}).get("error")) for n in tier_files}
    print(f"merged {len(merged)} prompts x {list(tier_files)} -> {args.out}")
    print(f"per-tier successful: {by_tier}")

if __name__ == "__main__":
    main()
