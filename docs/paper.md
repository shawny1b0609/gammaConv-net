# Paper overview

This English overview follows the author's second revised manuscript of **融合先验信息与深度学习的小样本水文建模**, published as *Hydrological modeling with priors information and deep learning under small-sample conditions* in *Advances in Water Science*. It condenses the research argument and method without adding new experimental claims. [Journal record](https://skxjz.nhri.cn/article/doi/10.14042/j.cnki.32.1309.2026.04.002).

## Research question

Hydrological models often have limited training observations. Process-based models provide a physical interpretation, while data-driven models can represent nonlinear rainfall–runoff relationships but depend on the quantity and quality of observations. The study examines whether hydrological prior information can improve predictive performance and interpretability when only a small training sample is used.

## Method

GammaConv-Net embeds learnable Gamma distributions in a neural network to describe the lag and attenuation of rainfall–runoff responses. A sliding window of rainfall and flow observations provides the input for a one-day-ahead flow prediction.

The architecture has four parts, corresponding to manuscript Section 1.1. A Gamma convolution layer produces an initial rainfall response. A linear network encodes rainfall, and an LSTM encodes the temporal development of flow. These representations are fused and passed through a fully connected prediction network. Gamma parameters and neural-network weights are trained jointly.

The single-kernel model represents an aggregate response. The dual-kernel model separates fast and slow components, interpreted in the paper as surface and groundwater responses. The triple-kernel model adds an intermediate component associated with subsurface flow. In the multi-kernel models, current rainfall intensity and antecedent rainfall conditions influence the allocation of rainfall among components. Each component has its own Gamma response, and the component responses are summed (Section 1.2).

The study uses a loss that emphasizes high-flow errors and AdamW optimization. It compares the proposed models with the Xin'anjiang model and a multilayer perceptron. Evaluation includes Nash–Sutcliffe efficiency, root mean squared error, mean absolute error, and relative errors in flood peaks and volumes (Sections 1.3–1.4).

## Small-sample experiment

The manuscript uses continuous training periods at fractions of 5%, 10%, 25%, 50%, 75%, and 100%, prioritizing periods containing major high-flow events. All sample sizes are assessed on the same later validation period. This preserves the temporal continuity of the selected hydrological processes (Section 2.2).

The manuscript reports that the multi-kernel models retain an efficiency of about 0.86 at the 5% sample size, compared with 0.67 for Xin'anjiang and 0.73 for the multilayer perceptron. Under full-sample conditions, multi-kernel efficiency exceeds 0.90. These are findings from the paper, also described in the [institutional abstract](https://www.nhri.cn/xsjl/art/2026/art_01153cb27bf047df949ebc1c4c8aaa03.html); they are not scores from the short runs supplied here.

## Interpretation

The paper examines the fitted Gamma parameters and allocation weights to interpret differences in runoff response time. The dual- and triple-kernel structures distinguish faster and slower response components, while the learned allocation changes with rainfall conditions. The conclusions emphasize data efficiency, predictive performance under limited samples, and the hydrological interpretation of the learned response parameters (Sections 3–4).

## Relation to this release

The repository preserves the original model calculations and provides executable training and analysis tools. Implementation details and differences between the manuscript notation and the supplied scripts are documented in [method.md](method.md). The de-identified CSV files do not represent any particular basin. The short release-validation runs are documented separately in [validation.md](validation.md).

The manuscript files, review correspondence, and publisher-owned article are not redistributed. Use the [DOI](https://doi.org/10.14042/j.cnki.32.1309.2026.04.002) or [alternative article records](paper_access.md) to locate the publication.
