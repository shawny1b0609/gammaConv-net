# Release validation

The release was checked on Windows with Python 3.12.3, PyTorch 2.5.1, NumPy 2.3.1, pandas 2.3.2, and Matplotlib 3.10.6. CPU training used two threads. One-epoch CUDA checks also completed for all three variants on an NVIDIA GeForce RTX 4090.

Clean Linux installations with Python 3.11 and 3.12 passed the full test and command-line workflow, including plotting and prediction. The Python 3.12 job also executed the notebook. [Initial verification run](https://github.com/shawny1b0609/gammaConv-net/actions/runs/38691977159).

## Checks performed

- All 6,575 released data rows retain the original flow and rainfall fields exactly. The records are daily and ordered; calendar dates have been replaced with day indices.
- Executable syntax trees of the five original model/loss modules match the release after excluding docstrings and standalone demos. English documentation and formatting do not change model calculations.
- Direct comparisons with the original four model classes give identical outputs and gradients on fixed inputs and seeds; see [the comparison record](model_check.json).
- Twelve automated tests cover data integrity, chronological windows, training-only scaling, invalid records, kernel normalization, an impulse-response causality check, rainfall allocation, forward and backward passes, retained losses, and checkpoint restoration and inference.
- All three model variants complete training, validation, checkpoint export, and reloaded prediction. The command-line experiment, plotting, and inference are also exercised by [GitHub Actions](https://github.com/shawny1b0609/gammaConv-net/actions/workflows/tests.yml).
- The English notebook executes from beginning to end with the declared optional dependencies.

## Fitted example

The command below ran all three models for 150 epochs on CPU, with seed 42 and the original experiment's fraction-dependent settings:

```bash
python -m gammaconv.experiment --fractions 0.05 --epochs 150 --out runs/example
```

| Model | Hidden units | Selected epoch | Validation NSE | RMSE (m³/s) | MAE (m³/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single | 64 | 139 | 0.6486 | 910.18 | 577.80 |
| Dual | 76 | 119 | 0.8112 | 667.05 | 528.40 |
| Triple | 89 | 123 | 0.7993 | 687.82 | 488.42 |

The sample contains 164 consecutive training days, producing 134 input–target windows. All models use the same 702 validation targets. Hidden-layer sizes follow the original fraction-dependent preset, so the comparison includes both kernel structure and network size. Validation RMSE selects the checkpoint, and the reported metrics describe performance on that selection sequence. The runs document the 150-epoch release example.

[Complete run records](example_runs.json) include the selected day range, losses, hyperparameters, scalers, fitted parameters, and environment. [Figure tables](figure_data/README.md) contain predictions, response kernels, and allocation weights for all three models. The dual-model [training history](example_output/history.csv) is also included.

The [release checkpoint archive](https://github.com/shawny1b0609/gammaConv-net/releases/tag/v0.1.0) contains all three fitted examples. Each checkpoint was extracted and reloaded, and its validation predictions were checked against the saved CSV. A SHA-256 file is supplied with the archive.

To make the figures from that checkpoint:

```bash
python -m gammaconv.plot --run runs/example/0.05/dual --out runs/example/figures
```

The script saves 300 dpi PNG previews and vector PDF files using anonymous day indices. Numerical results may vary across hardware and library versions.

The [README figures](figures/README.md) can also be generated directly from the public tables with `python examples/make_readme_figures.py`.
