# Formal Statistical Inference

## HAC methodology
Monthly inference uses Newey--West/HAC covariance with the repository's frozen 6-month lag and 95% confidence intervals.
Three-state regime differences use one LOW-reference dummy regression per model/weighting (or model for IC), with covariance-consistent linear contrasts.
Holm adjustment is primary across the three pre-specified pairwise contrasts within each model/outcome/weighting family; Benjamini--Hochberg FDR p-values are retained as a secondary diagnostic.

## D10-D1 inference
- Elastic Net EW LOW: mean=0.00486, HAC SE=0.00281, t=1.73, p=0.0836, 95% CI=[-0.00065, 0.01037], N=104.
- Elastic Net EW MIDDLE: mean=0.00670, HAC SE=0.00545, t=1.23, p=0.2190, 95% CI=[-0.00398, 0.01739], N=108.
- Elastic Net EW HIGH: mean=-0.00174, HAC SE=0.00664, t=-0.26, p=0.7935, 95% CI=[-0.01475, 0.01128], N=99.
- Elastic Net VW LOW: mean=0.00344, HAC SE=0.00308, t=1.12, p=0.2637, 95% CI=[-0.00260, 0.00948], N=104.
- Elastic Net VW MIDDLE: mean=-0.00314, HAC SE=0.00466, t=-0.67, p=0.5011, 95% CI=[-0.01227, 0.00600], N=108.
- Elastic Net VW HIGH: mean=-0.00250, HAC SE=0.00719, t=-0.35, p=0.7276, 95% CI=[-0.01660, 0.01159], N=99.
- XGBoost EW LOW: mean=0.00813, HAC SE=0.00253, t=3.21, p=0.0013, 95% CI=[0.00317, 0.01310], N=104.
- XGBoost EW MIDDLE: mean=0.01201, HAC SE=0.00536, t=2.24, p=0.0249, 95% CI=[0.00151, 0.02251], N=108.
- XGBoost EW HIGH: mean=-0.00232, HAC SE=0.00661, t=-0.35, p=0.7262, 95% CI=[-0.01528, 0.01065], N=99.
- XGBoost VW LOW: mean=0.00171, HAC SE=0.00317, t=0.54, p=0.5899, 95% CI=[-0.00450, 0.00792], N=104.
- XGBoost VW MIDDLE: mean=0.00089, HAC SE=0.00604, t=0.15, p=0.8831, 95% CI=[-0.01095, 0.01273], N=108.
- XGBoost VW HIGH: mean=-0.00149, HAC SE=0.00840, t=-0.18, p=0.8594, 95% CI=[-0.01795, 0.01497], N=99.

## Regime differences
- Elastic Net EW MIDDLE - LOW: difference=0.00184, HAC SE=0.00546, t=0.34, raw p=0.7364, Holm p=0.7689, N=311.
- Elastic Net EW HIGH - LOW: difference=-0.00660, HAC SE=0.00685, t=-0.96, raw p=0.3352, Holm p=0.7689, N=311.
- Elastic Net EW HIGH - MIDDLE: difference=-0.00844, HAC SE=0.00744, t=-1.14, raw p=0.2563, Holm p=0.7689, N=311.
- Elastic Net VW MIDDLE - LOW: difference=-0.00658, HAC SE=0.00525, t=-1.25, raw p=0.2104, Holm p=0.6311, N=311.
- Elastic Net VW HIGH - LOW: difference=-0.00595, HAC SE=0.00784, t=-0.76, raw p=0.4480, Holm p=0.8960, N=311.
- Elastic Net VW HIGH - MIDDLE: difference=0.00063, HAC SE=0.00875, t=0.07, raw p=0.9424, Holm p=0.9424, N=311.
- XGBoost EW MIDDLE - LOW: difference=0.00388, HAC SE=0.00547, t=0.71, raw p=0.4789, Holm p=0.4789, N=311.
- XGBoost EW HIGH - LOW: difference=-0.01045, HAC SE=0.00689, t=-1.52, raw p=0.1292, Holm p=0.2584, N=311.
- XGBoost EW HIGH - MIDDLE: difference=-0.01432, HAC SE=0.00736, t=-1.95, raw p=0.0516, Holm p=0.1548, N=311.
- XGBoost VW MIDDLE - LOW: difference=-0.00082, HAC SE=0.00660, t=-0.12, raw p=0.9012, Holm p=1.0000, N=311.
- XGBoost VW HIGH - LOW: difference=-0.00320, HAC SE=0.00861, t=-0.37, raw p=0.7107, Holm p=1.0000, N=311.
- XGBoost VW HIGH - MIDDLE: difference=-0.00238, HAC SE=0.00962, t=-0.25, raw p=0.8050, Holm p=1.0000, N=311.

## Monthly Rank IC
- Elastic Net LOW: mean IC=0.00703, HAC SE=0.00868, t=0.81, p=0.4180, 95% CI=[-0.00998, 0.02405], N=104.
- Elastic Net MIDDLE: mean IC=0.01257, HAC SE=0.01344, t=0.94, p=0.3494, 95% CI=[-0.01377, 0.03891], N=108.
- Elastic Net HIGH: mean IC=-0.01157, HAC SE=0.01476, t=-0.78, p=0.4330, 95% CI=[-0.04051, 0.01736], N=99.
- XGBoost LOW: mean IC=0.01235, HAC SE=0.00760, t=1.63, p=0.1039, 95% CI=[-0.00254, 0.02724], N=104.
- XGBoost MIDDLE: mean IC=0.02129, HAC SE=0.01237, t=1.72, p=0.0852, 95% CI=[-0.00296, 0.04554], N=108.
- XGBoost HIGH: mean IC=-0.00605, HAC SE=0.01397, t=-0.43, p=0.6648, 95% CI=[-0.03343, 0.02133], N=99.

## Rank IC regime differences
- Elastic Net MIDDLE - LOW: difference=0.00554, HAC SE=0.01547, t=0.36, raw p=0.7201, Holm p=0.7201, N=311.
- Elastic Net HIGH - LOW: difference=-0.01861, HAC SE=0.01576, t=-1.18, raw p=0.2377, Holm p=0.4754, N=311.
- Elastic Net HIGH - MIDDLE: difference=-0.02415, HAC SE=0.01668, t=-1.45, raw p=0.1476, Holm p=0.4429, N=311.
- XGBoost MIDDLE - LOW: difference=0.00894, HAC SE=0.01434, t=0.62, raw p=0.5332, Holm p=0.5332, N=311.
- XGBoost HIGH - LOW: difference=-0.01841, HAC SE=0.01450, t=-1.27, raw p=0.2043, Holm p=0.4086, N=311.
- XGBoost HIGH - MIDDLE: difference=-0.02734, HAC SE=0.01537, t=-1.78, raw p=0.0751, Holm p=0.2254, N=311.

## Multiple testing
No pairwise three-state regime difference is statistically distinguishable from zero at the 5% family-wise level after Holm adjustment.

## Interpretation
Inference is conditional on the frozen empirical specification. Statistical significance is not interpreted as economic importance, and regime associations are observational rather than causal.

## Limitations
Multiple-testing control is defined only for the pre-specified three-contrast families reported here. Lag sensitivity, alternative regime thresholds, turnover, and richer transaction-cost analysis belong to later robustness stages.
