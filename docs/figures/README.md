# Figures

English diagrams follow Figures 1–2 of the author's revised manuscript. Paper-result charts use its reported tables; example charts use the released data and fitted checkpoints. See the [source records](../figure_data/README.md).

| Figure | PNG | SVG | PDF |
| --- | --- | --- | --- |
| Model architecture | [View](architecture.png) | [Vector](architecture.svg) | [Download](architecture.pdf) |
| Kernel structures | [View](kernel_variants.png) | [Vector](kernel_variants.svg) | [Download](kernel_variants.pdf) |
| Paper performance | [View](paper_performance.png) | [Vector](paper_performance.svg) | [Download](paper_performance.pdf) |
| Paper response times | [View](paper_response_times.png) | [Vector](paper_response_times.svg) | [Download](paper_response_times.pdf) |
| Training blocks | [View](sampling.png) | [Vector](sampling.svg) | [Download](sampling.pdf) |
| Validation hydrographs | [View](validation_models.png) | [Vector](validation_models.svg) | [Download](validation_models.pdf) |
| Fitted daily kernels | [View](model_kernels.png) | [Vector](model_kernels.svg) | [Download](model_kernels.pdf) |
| Rainfall allocation | [View](model_allocation.png) | [Vector](model_allocation.svg) | [Download](model_allocation.pdf) |

From the repository root, regenerate all figures with:

```bash
python examples/make_readme_figures.py
```

The script reads the public CSV tables and writes 300 dpi PNG previews plus SVG and PDF files. Diagram curves are schematic. Numerical charts preserve the source values; heatmap cells are rasterized within vector exports to keep file sizes small.

The earlier dual-model plots (`prediction`, `kernels`, and `allocation`) remain available for the command-line plotting walkthrough.
