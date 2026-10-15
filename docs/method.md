# Implementation notes

The scientific overview in [paper.md](paper.md) follows the manuscript. This page describes the executable research code, including details needed to interpret its outputs.

## Inputs and prediction

The three main models take rainfall and flow tensors of shape `(batch, steps, 1)`. The supplied runner uses a 30-day history to predict flow on the next day. Each validation window uses observed past flow, so evaluation is a sequence of one-day-ahead predictions, not an autonomous multi-day simulation. The main models require complete input windows; their retained `flow_mask` argument is not used.

Rainfall and flow are scaled separately with minima and maxima from the chronological training prefix. Validation values may fall outside `[0, 1]` and are not clipped. The final Softplus layer produces a nonnegative prediction on the normalized scale; the saved flow scaler converts it back to the supplied units.

## Gamma response

The code evaluates a finite sequence of weights at integer lags:

$$
g_j = \frac{(j+10^{-6})^{\alpha-1}\exp(-\beta j)}
{\sum_{i=0}^{L-1}(i+10^{-6})^{\alpha-1}\exp(-\beta i)+10^{-8}},
\qquad j=0,\ldots,L-1.
$$

The common density prefactor cancels under normalization. Parameters are clamped to component-specific bounds, then learned together with the network weights. The implementation uses a reversed kernel in `conv1d` and crops the padded output to the input length. This gives a causal convolution for a fixed kernel.

The manuscript calls β a scale parameter, but its density and the code use `exp(-βt)`: β is the **rate** in this parameterization. The manuscript's `α/β` is the continuous Gamma mean. The mean of the finite, discretized response is instead `sum(j * g_j)`; the two need not coincide, especially when α is below one and the lag-zero term dominates.

![Fitted discrete Gamma responses](figures/kernels.png)

This figure shows the fitted weights from the documented example run, including the lag-zero weight. It is generated from the saved model, not traced from the paper.

## Multiple responses

The dual model has fast and slow kernels; the triple model adds an intermediate kernel. Allocation depends on the last rainfall value in the input window and the mean of up to the last seven values. The mean includes the latest observation. Each window receives one allocation vector, held fixed across that window.

Base and intensity weights are normalized with softmax. Sigmoid factors, together with the original conditional branches, adjust the allocation according to rainfall intensity and antecedent rainfall. The adjusted values are clipped to `[0.05, 0.9]` and renormalized. All kernels use the allocated rainfall, and their responses are summed. The dual model also passes its allocation weights to the feature-fusion network.

![Rainfall allocation in the fitted dual-kernel example](figures/allocation.png)

The blank regions are infeasible combinations of latest rainfall and seven-day mean when normalized rainfall lies in `[0, 1]`. The response labels describe the intended hydrological interpretation. No separate observations of surface flow or groundwater flow are fitted.

Normalization preserves the allocation across components and constrains each response kernel. Cropping removes the response tail outside the window, and the final neural prediction is unconstrained by a catchment-scale balance equation. The code therefore does not enforce exact water-volume conservation in the final discharge prediction.

## Training and evaluation

The split is chronological: `floor(0.818 * number_of_rows)` records form the training prefix; the remainder forms the validation suffix. Windows are made separately within each part, so the first 30 validation records provide history and are not validation targets. For `series_a.csv`, this gives 3,286 training-prefix records and 702 validation targets.

For a fraction below one, the experiment searches every contiguous block of the requested length within the training prefix. Its original score emphasizes high flow: the maximum and 90th percentile for fractions up to 10%, with mean, variability, and range terms added at larger fractions. Block selection and scaling both inspect the full training prefix. This is an experiment in fitting selected sample windows, not a setting where the unselected training observations are unavailable.

Two original training paths are retained:

| Entry point | Loss and optimizer |
| --- | --- |
| `gammaconv.train` | Sample-size-dependent high-, middle-, and low-flow weights; AdamW; cosine learning-rate decay; gradient clipping |
| `gammaconv.experiment` | Sum of weighted mean squared errors in three batch-relative flow bands; AdamW with a fixed learning rate; original fraction-dependent hyperparameters |

The experiment uses the batch 80th and 30th percentiles to form high- and low-flow groups. Its hyperparameter table is in [experiment.py](../gammaconv/experiment.py). The manuscript's Equation 12 describes a different high-flow weighting expression with an 85th-percentile threshold. The released scripts preserve the supplied code's losses; they should not be described as a literal implementation of that equation.

Validation is checked after every epoch. The checkpoint with the lowest validation RMSE is copied and restored; on fixed nonconstant targets this is equivalent to maximizing NSE. Early stopping uses the configured number of epochs without improvement. NSE is undefined for constant targets and is written as JSON `null`; RMSE still supports checkpoint selection. The reported validation scores are used in model selection and are not independent test estimates.

## Reproducibility and scope

The runner records seeds, dependency versions, data checksum, scaling parameters, and selected day indices. CPU is the default. Fixed seeds aid repeatability within an environment; exact equality across PyTorch versions or CPU and GPU backends is not assumed.

The three main architectures and all original standalone model/loss modules are included. The earlier correction-based architecture in `hydrological_model.py` is retained for reference; it is not one of the three experiment variants and contains modules unused by its forward pass. The original notebook workflows are consolidated into a short English walkthrough and shared training/plotting functions.

The release adds input validation, portable paths, saved run records, checkpoint inference, and automated checks. It fixes the original CPU checkpoint aliasing and the training entry point's failure to restore the best weights. Model calculations, component bounds, and the small-sample block-scoring rule remain unchanged. The XAJ and MLP comparison experiments are outside this model repository; the full paper benchmark is not asserted to be reproduced here.
