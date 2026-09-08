# Seasonal Anomaly Detection: When "Learn the Normal Pattern First" Doesn't Automatically Work

I built this project to test something specific: if a naive anomaly detector fails on seasonal data, does adding seasonal awareness actually fix it? The honest answer is: sometimes, and understanding exactly when and why turned out to be more useful than a clean win would have been.

The short version. A naive global threshold fails on seasonal time series in two different ways, not one: it over-alerts on ordinary daily peaks, or it goes blind because the seasonal swing itself corrupts the threshold. Seasonal decomposition fixes the second failure on the one series with enough data to support it, and makes almost everything else worse. On four of the six series I tested, nothing I built beats the naive baseline. I think that result is more useful to know than a project that quietly reported only the win.

## Data

I used the Numenta Anomaly Benchmark (NAB), a public benchmark of 58 real-world and artificial time series with human-labelled anomaly windows, maintained by Numenta (MIT license, github.com/numenta/NAB, pinned to commit ea702d7). I picked six real series after actually plotting daily and weekly autocorrelation across all of them, not by assumption:

- **nyc_taxi**: taxi ridership, 30-min cadence, 215 days, strong daily and weekly cycle, 5 labelled anomalies (marathon, holidays, a snowstorm)
- **ambient_temperature_system_failure**: sensor reading, 60-min cadence, 329 days, daily cycle riding a slow multi-month drift, 2 anomalies
- **exchange-2_cpm_results**: ad exchange CPM, 60-min cadence, 69 days, clean daily cycle, 2 anomalies
- **occupancy_t4013** and **occupancy_6005**: traffic sensors from the same corridor, 5-min cadence, 16 days each, sharp commute peaks, real data gaps, 2 and 1 anomalies
- **Twitter_volume_AMZN**: tweet volume, 5-min cadence, 55 days, a weak diurnal cycle buried under bursty spikes, 4 anomalies

I deliberately kept a hard case (Twitter_AMZN) and a short, gappy pair (the traffic sensors) rather than picking only clean examples.

Evaluation: a detected point counts as a hit if it falls inside a labelled anomaly window or a small buffer around it. This is a simplified version of NAB's own scoring, which weights earlier detections within a window more heavily; I did not implement that full complexity, and I say so rather than imply I did.

## The naive baseline fails in two different ways

A static detector (flag anything beyond 3 standard deviations of the series' own mean) does badly on the seasonal series, but not for one reason. On the traffic and ad-exchange series, it over-alerts: the daily rush-hour peak trips the band every single day, so 70 to 97 percent of its false positives land in the normal commute window. On nyc_taxi, the opposite happens: the daily swing itself is so large that it inflates the global standard deviation past the point where the detector can fire at all. It catches one anomaly out of five in 215 days, not because the model is too sensitive, but because it has gone blind.

That second failure mode is the more interesting one, because no single choice of threshold fixes both problems at once.

## Seasonal decomposition (STL) does not fix it

I applied STL decomposition (trend plus one daily seasonal cycle) and moved the threshold from the raw value to the residual. Precision fell on every one of the six series. I traced this to three separate, verified causes:

1. A real weekly cycle survives in the residual on nyc_taxi and ambient_temperature (7-day residual autocorrelation of 0.78 and 0.38), so weekends become systematic false positives.
2. The residual is heavy-tailed and its spread is not constant (excess kurtosis from 5 to 1276 across the six series), so a single global sigma is too tight during bursty stretches.
3. On ambient_temperature, where the naive baseline already worked well (precision 0.84), STL's trend absorption worked exactly as intended, but the resulting band shrank from roughly plus-or-minus 12.7 to plus-or-minus 3.5, so ordinary weekend swings started crossing a much tighter line. There was nothing to gain here and a lot to lose.

## Two targeted fixes, tested separately

I tested a weekly seasonal component (MSTL) and a robust local scale (rolling MAD instead of a global standard deviation) as two separate changes, not bundled together, so I could tell which one actually helped.

**Adding the weekly cycle** fixed nyc_taxi outright: recall went from 1 of 5 windows to 5 of 5, precision from 0.12 to 0.71. It also overfit badly on the two short traffic series, which have only 2.3 weekly cycles of data. In one case the model fit the labelled anomaly itself as a normal weekly pattern and the in-window residual came back exactly zero, erasing the detection entirely. I set a rule: only fit a weekly component with at least 10 full weekly cycles of data, otherwise fall back to daily-only decomposition. Ten was chosen because with fewer cycles a single bad week can dominate more than 10 percent of the seasonal estimate, which matches what actually happened at 2.3 and 7.9 cycles in testing.

**The local MAD scale backfired** on every series, and the reason is specific and worth stating precisely: on these heavy-tailed residuals, the standard robust-sigma conversion (1.4826 times MAD) comes out to only 20 to 49 percent of the actual standard deviation. A threshold calibrated for a normal distribution's spread is far too tight here, so it floods the output with false positives. I tested wider thresholds (k = 6 through 9, applied uniformly across all six series, not tuned per series) and none of them beat a plain global standard deviation at k = 3. The local MAD idea is not wrong, it is a genuinely different scale that needs its own calibration, and I did not find one that worked in the range I tested.

## The final result, stated plainly

The final detector uses the weekly-cycle-if-enough-data rule with a global sigma threshold, no local MAD scale, still k = 3 fixed across every series.

- **Substantially improved, arguably solved**: nyc_taxi. All 5 anomaly windows caught, versus 1 of 5 for the naive baseline, at the cost of dropping from perfect precision to roughly one false alert for every two real ones.
- **Improved, not solved**: exchange-2_cpm. Both windows now caught, but precision fell from 0.50 to 0.21.
- **Not solved by anything tried in this project**: ambient_temperature, both traffic sensors, and Twitter_AMZN. On all four, the naive global threshold from the very first phase remains the best result. Every seasonal method I tried made these four worse, not better.

## What this actually shows

A model that is more sophisticated is not automatically better. Seasonal decomposition helped exactly where its assumptions held (enough data to estimate a stable weekly pattern, a residual that isn't dominated by bursts) and hurt everywhere they didn't. The useful part of this project was not building the fix, it was working out precisely which series had enough structure to support which method, and being willing to report that the simplest approach still wins on most of the data.

## Limitations

- Six series out of NAB's 58. The four unsolved series are not necessarily representative of every failure mode in the full benchmark.
- Evaluation uses a simplified time-tolerant scoring, not NAB's full scoring methodology.
- This is a batch analysis over complete series, not a true online/streaming evaluation where the detector only sees past data at each point.
- The local MAD scale was tested at a handful of discrete k values (6 through 9), not a continuous search, so a working calibration may exist outside that range.
- No ensemble or machine-learning-based method was tried. This project deliberately stayed within classical, explainable decomposition methods.

## Repository structure

```
src/detectors/    naive threshold, STL, MSTL, local MAD scale detectors
src/analysis/     baseline runs, ablation, seasonality exploration
src/evaluation/   time-tolerant scoring against labelled windows
data/raw/         NAB data, not tracked, see data/README.md for the fetch script and attribution
reports/figures/  overlay plots, not tracked
```

## Running it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/fetch_data.sh
python -m src.analysis.explore_seasonality
python -m src.analysis.run_naive_baseline
python -m src.analysis.run_stl_baseline
python -m src.analysis.compare_naive_vs_stl
python -m src.analysis.run_ablation
python -m src.analysis.run_phase5
```

Data comes from the Numenta Anomaly Benchmark (Lavin & Ahmad, 2015), MIT licensed, github.com/numenta/NAB.
