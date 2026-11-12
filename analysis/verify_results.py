"""Verify that the committed numbers still match the committed data.

The live measurements are committed to the repository (DATA.md), so the
published numbers must be checkable: this script recomputes the headline
numbers from the committed oracle + scores files and compares them with the
committed snapshot `analysis/results.json`.

  python analysis/verify_results.py                 # fast: recompute the analysis tables
  python analysis/verify_results.py --rescore       # slow: also re-run the metric on the
                                                     #       committed oracle data (~3 min,
                                                     #       executes the unit tests)

Exit code 0 = in sync, 1 = drift (the snapshot must be regenerated with
`python analysis/report.py --json analysis/results.json`).
"""
import argparse, json, pathlib, subprocess, sys, tempfile

BASE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "analysis"))

TOL = 1e-6


def diff(a, b, path=""):
    """Yield human-readable differences between two JSON structures."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                yield f"{path}/{k}: missing in recomputed snapshot"
            elif k not in b:
                yield f"{path}/{k}: missing in committed snapshot"
            else:
                yield from diff(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield f"{path}: length {len(a)} != {len(b)}"
            return
        for i, (x, y) in enumerate(zip(a, b)):
            yield from diff(x, y, f"{path}[{i}]")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if abs(a - b) > max(TOL, TOL * max(abs(a), abs(b))):
            yield f"{path}: {a} != {b}"
    else:
        if a != b:
            yield f"{path}: {a!r} != {b!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default=str(BASE / "harness" / "oracle_v2.jsonl"))
    ap.add_argument("--scores", default=str(BASE / "scoring" / "scores_v2.jsonl"))
    ap.add_argument("--snapshot", default=str(BASE / "analysis" / "results.json"))
    ap.add_argument("--rescore", action="store_true",
                    help="also re-run scoring/score.py on the committed oracle and compare the scores")
    args = ap.parse_args()

    for p in (args.oracle, args.scores, args.snapshot):
        if not pathlib.Path(p).exists():
            print(f"missing input: {p}")
            return 1

    import report
    with tempfile.TemporaryDirectory() as td:
        md = pathlib.Path(td) / "report.md"
        js = pathlib.Path(td) / "results.json"
        argv = sys.argv
        sys.argv = ["report.py", "--in", args.oracle, "--scores", args.scores,
                    "--out", str(md), "--json", str(js)]
        try:
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()):
                report.main()
        finally:
            sys.argv = argv
        recomputed = json.loads(js.read_text(encoding="utf-8"))
    committed = json.loads(pathlib.Path(args.snapshot).read_text(encoding="utf-8"))

    problems = list(diff(committed, recomputed))
    if problems:
        print(f"RESULTS SNAPSHOT DRIFT ({len(problems)} differences):")
        for p in problems[:40]:
            print("  -", p)
        return 1
    print(f"results snapshot in sync with the committed data "
          f"({recomputed['prompts']} prompts, {len(recomputed['policies'])} policies)")

    if args.rescore:
        with tempfile.TemporaryDirectory() as td:
            out = pathlib.Path(td) / "scores.jsonl"
            r = subprocess.run([sys.executable, str(BASE / "scoring" / "score.py"),
                                "--in", args.oracle, "--out", str(out), "--metric", "v2"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print("re-scoring failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
                return 1
            new = {json.loads(l)["id"]: json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
        old = {json.loads(l)["id"]: json.loads(l) for l in pathlib.Path(args.scores).read_text(encoding="utf-8").splitlines() if l.strip()}
        bad = 0
        for pid, row in new.items():
            prev = old.get(pid)
            if not prev:
                print(f"  missing in committed scores: {pid}")
                bad += 1
                continue
            for tier, e in row["tiers"].items():
                pe = (prev["tiers"].get(tier) or {}).get("quality")
                if pe is None or abs(pe - e["quality"]) > 1e-9:
                    print(f"  quality drift {pid}/{tier}: {pe} != {e['quality']}")
                    bad += 1
        if bad:
            print(f"RE-SCORING DRIFT ({bad} differences)")
            return 1
        print(f"re-scoring reproduces the committed scores exactly ({len(new)} prompts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
