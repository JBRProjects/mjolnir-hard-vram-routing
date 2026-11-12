## Report: `oracle_v2.jsonl` / `scores_v2.jsonl`

Prompts: 180 (test split: 54)


### Tier characterization

| tier | runs | errors | avg latency (s) | median (s) | p90 (s) | avg tok/s |
|---|---|---|---|---|---|---|
| turbo | 180 | 0 | 20.3 | 19.7 | 25.3 | 6.54 |
| vector | 180 | 0 | 49.6 | 48.2 | 60.3 | 2.67 |
| ultra | 180 | 0 | 49.9 | 48.6 | 57.2 | 2.59 |

### Per-task quality (all prompts)

| task | turbo | vector | ultra |
|---|---|---|---|
| code_gen | 0.659 | 0.143 | 0.968 |
| factual_qa | 0.917 | 0.733 | 0.917 |
| reasoning | 0.783 | 0.550 | 0.817 |

### Oracle tier distribution

| tier | count | share |
|---|---|---|
| turbo | 146 | 0.811 |
| vector | 6 | 0.033 |
| ultra | 28 | 0.156 |

### Oracle tier distribution per task

| task | turbo | vector | ultra |
|---|---|---|---|
| code_gen | 37 (0.62) | 3 (0.05) | 20 (0.33) |
| factual_qa | 58 (0.97) | 1 (0.02) | 1 (0.02) |
| reasoning | 51 (0.85) | 2 (0.03) | 7 (0.12) |

Measured tier load/eviction cost (ms): turbo=3880, vector=11260, ultra=11310


### Router comparison (held-out test split)

| policy | avg cost (ms) | quality | oracle-gap captured (%) |
|---|---|---|---|
| random | 42016.9 | 0.718 | 0.0% |
| heuristic | 41675.5 | 0.444 | -130.1% |
| cascade (confidence gate) | 20757.0 | 0.773 | 26.5% |
| cascade (verified gate) | 34985.9 | 0.890 | 82.4% |
| learned | 21466.6 | 0.792 | 35.3% |
| oracle | 25737.3 | 0.927 | 100.0% |
| cascade (verified gate + tier switch) | 38329.6 | 0.890 | 82.4% |

### Cost-aware learned router: quality/cost frontier (sklearn-logreg (n=378))

| cost penalty lambda | avg cost (ms) | quality | tier mix |
|---|---|---|---|
| 0.00 | 50361.6 | 0.872 | turbo:0, vector:0, ultra:54 |
| 0.05 | 36979.9 | 0.872 | turbo:25, vector:0, ultra:29 |
| 0.10 | 30699.0 | 0.890 | turbo:36, vector:0, ultra:18 |
| 0.20 | 23542.2 | 0.773 | turbo:49, vector:0, ultra:5 |
| 0.40 | 20757.0 | 0.773 | turbo:54, vector:0, ultra:0 |
| 0.80 | 20757.0 | 0.773 | turbo:54, vector:0, ultra:0 |
| 1.60 | 20757.0 | 0.773 | turbo:54, vector:0, ultra:0 |

### Router feature ablation (held-out split)

| feature subset | agreement | avg cost (ms) | quality | capture (%) |
|---|---|---|---|---|
| length only | 0.870 | 22012.1 | 0.810 | 51.8% |
| task keywords only | 0.833 | 20757.0 | 0.773 | 36.6% |
| surface ratios only | 0.833 | 20757.0 | 0.773 | 36.6% |
| all features | 0.852 | 21466.6 | 0.792 | 44.2% |