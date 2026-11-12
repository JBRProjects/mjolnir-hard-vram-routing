| policy | avg cost (ms) | quality | oracle-gap captured (%) |
|---|---|---|---|
| random | 42016.9 | 0.718 | 0.0% |
| heuristic | 41675.5 | 0.444 | -130.1% |
| cascade | 20757.0 | 0.773 | 26.5% |
| cascade-verified | 34985.9 | 0.890 | 82.4% |
| learned | 21466.6 | 0.792 | 35.3% |
| oracle | 25737.3 | 0.927 | 100.0% |
| cascade-verified+switch | 38329.6 | 0.890 | 82.4% |

## Per-task quality on the test split

| task | turbo | vector | ultra |
|---|---|---|---|
| code_gen | 0.597 | 0.056 | 0.949 |
| factual_qa | 0.833 | 0.667 | 0.833 |
| reasoning | 0.889 | 0.500 | 0.833 |

## Oracle tier distribution (test split)

| tier | count | share |
|---|---|---|
| turbo | 45 | 0.833 |
| vector | 0 | 0.000 |
| ultra | 9 | 0.167 |

## Cost-aware learned router: quality/cost frontier

| cost penalty lambda | avg cost (ms) | quality | tier mix |
|---|---|---|---|
| 0.00 | 50361.6 | 0.872 | turbo:0, vector:0, ultra:54 |
| 0.05 | 36979.9 | 0.872 | turbo:25, vector:0, ultra:29 |
| 0.10 | 30699.0 | 0.890 | turbo:36, vector:0, ultra:18 |
| 0.20 | 23542.2 | 0.773 | turbo:49, vector:0, ultra:5 |
| 0.40 | 20757.0 | 0.773 | turbo:54, vector:0, ultra:0 |
| 0.80 | 20757.0 | 0.773 | turbo:54, vector:0, ultra:0 |
| 1.60 | 20757.0 | 0.773 | turbo:54, vector:0, ultra:0 |
