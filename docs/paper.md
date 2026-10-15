# Paper overview

This overview summarizes the author's second revised manuscript of **融合先验信息与深度学习的小样本水文建模**, published as *Hydrological modeling with priors information and deep learning under small-sample conditions* in *Advances in Water Science*. [Journal record](https://skxjz.nhri.cn/article/doi/10.14042/j.cnki.32.1309.2026.04.002).

## Research question

Hydrological models often have limited training observations. Process-based models provide a physical interpretation, while data-driven models can represent nonlinear rainfall–runoff relationships but depend on the quantity and quality of observations. The study examines whether hydrological prior information can improve predictive performance and interpretability when only a small training sample is used.

## Method

GammaConv-Net embeds learnable Gamma distributions in a neural network to describe the lag and attenuation of rainfall–runoff responses. A sliding window of rainfall and flow observations provides the input for a one-day-ahead flow prediction.

The architecture has four parts, corresponding to manuscript Section 1.1. A Gamma convolution layer produces an initial rainfall response. A linear network encodes rainfall, and an LSTM encodes the temporal development of flow. These representations are fused and passed through a fully connected prediction network. Gamma parameters and neural-network weights are trained jointly.

The single-kernel model represents an aggregate response. The dual-kernel model separates fast and slow components, interpreted in the paper as surface and groundwater responses. The triple-kernel model adds an intermediate component associated with subsurface flow. In the multi-kernel models, current rainfall intensity and antecedent rainfall conditions influence the allocation of rainfall among components. Each component has its own Gamma response, and the component responses are summed (Section 1.2).

The study uses a loss that emphasizes high-flow errors and AdamW optimization. It compares the proposed models with the Xin'anjiang model and a multilayer perceptron. Evaluation includes Nash–Sutcliffe efficiency, root mean squared error, mean absolute error, and relative errors in flood peaks and volumes (Sections 1.3–1.4).

## Small-sample experiment

The manuscript uses continuous training periods at fractions of 5%, 10%, 25%, 50%, 75%, and 100%, prioritizing periods containing major high-flow events. All sample sizes are assessed on the same later validation period. This preserves the temporal continuity of the selected hydrological processes (Section 2.2).

Table 1 reports NSE 0.86 for both multi-kernel models at the 5% sample size, compared with 0.67 for Xin'anjiang and 0.73 for the multilayer perceptron. At the 100% fraction, both multi-kernel models reach NSE 0.90 at the table's reported precision. The [performance chart](figures/paper_performance.png) presents all six fractions.

## Interpretation

The paper examines the fitted Gamma parameters and allocation weights to interpret differences in runoff response time. The dual- and triple-kernel structures distinguish faster and slower response components, while the learned allocation changes with rainfall conditions. The conclusions emphasize data efficiency, predictive performance under limited samples, and the hydrological interpretation of the learned response parameters (Sections 3–4).

## Relation to this release

The repository preserves the original model calculations and provides executable training and analysis tools. [Implementation notes](method.md) describe the supplied scripts; [release validation](validation.md) records the fitted examples on de-identified data. The publication is available through its [DOI](https://doi.org/10.14042/j.cnki.32.1309.2026.04.002) and [alternative article records](paper_access.md).
