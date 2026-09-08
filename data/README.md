# Data

This project uses the **Numenta Anomaly Benchmark (NAB)** corpus as its dataset.

## Layout

```
data/
  raw/                          pulled locally, gitignored
    data/<category>/*.csv       timestamped single-metric series
    labels/combined_windows.json  ground-truth anomaly windows per series
    labels/combined_labels.json   ground-truth anomaly timestamps per series
  processed/                    analysis outputs, gitignored
```

## Getting the data

```
scripts/fetch_data.sh
```

This does a sparse checkout of only the `data/` and `labels/` directories of the
NAB repository at a pinned commit, copies them into `data/raw/`, and discards the
clone. No NAB source code is downloaded, and nothing under `src/` imports from
the NAB codebase.

## Provenance and attribution

- Source: Numenta Anomaly Benchmark, https://github.com/numenta/NAB
- Pinned commit: `ea702d75cc2258d9d7dd35ca8e5e2539d71f3140` (2024-12-03)
- License: MIT, "Copyright 2014-2024 Numenta Inc." (`LICENSE.txt` in that repo)
- Reference: Lavin & Ahmad, "Evaluating Real-time Anomaly Detection Algorithms:
  the Numenta Anomaly Benchmark", IEEE ICMLA 2015.

`data/raw/` is gitignored for repo hygiene, not for any licensing reason: there
is no value in committing a copy of a public benchmark that `fetch_data.sh`
reproduces exactly.
