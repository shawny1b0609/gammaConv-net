# Figure data

## Manuscript tables

Source: Yang et al. (2026), *Hydrological modeling with priors information and deep learning under small-sample conditions*. [DOI](https://doi.org/10.14042/j.cnki.32.1309.2026.04.002).

| File | Content | Source |
| --- | --- | --- |
| [paper_metrics.csv](paper_metrics.csv) | NSE, RMSE, and MAE for five models at six training fractions | Table 1 |
| [paper_response_times.csv](paper_response_times.csv) | Gamma parameters and mean response times | Table 2 |

Values were transcribed from the author's second revised manuscript. `mean_days` preserves the reported value and its precision. The continuous Gamma parameterization uses β as a rate, giving mean α/β; see [implementation notes](../method.md#gamma-response).

## Released examples

| File | Content | Source |
| --- | --- | --- |
| [sampling.csv](sampling.csv) | Selected block boundaries and window counts | `select_block` on `series_a.csv`; split 0.818, history 30 days |
| [example_predictions.csv](example_predictions.csv) | Observed flow and predictions from all three variants | Saved validation predictions |
| [example_kernels.csv](example_kernels.csv) | Normalized daily response weights at lags 0–29 | Fitted model checkpoints |
| [example_allocation.csv](example_allocation.csv) | Dual- and triple-model allocation on a 41 × 41 input grid | Fitted allocation functions |

The example runs use a 5% training block, 150 epochs, and seed 42. [Run records](../example_runs.json) contain the full settings; weights are in [release v0.1.0](https://github.com/shawny1b0609/gammaConv-net/releases/tag/v0.1.0). All day labels follow the [de-identified data convention](../../data/README.md).

For the allocation grid, the latest normalized rainfall is `r` and the seven-day mean is `m`. Each of the preceding six values is set to `(7m - r) / 6`; cells are retained when this value lies in [0, 1]. Allocation depends on these two summary inputs. Each retained cell's component weights sum to one within floating-point tolerance.

The [figure script](../../examples/make_readme_figures.py) uses these tables directly, so rendering requires no model training.
