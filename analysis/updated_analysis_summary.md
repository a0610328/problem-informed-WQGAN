# Complete four-model reconstruction: updated analysis

All 48 model/layer/run combinations are included in run ranking and training-history analysis. HEA layer 4/run 0 is the single user-confirmed completion override (last recorded epoch 3900).

## Median-run selections

- ZZ, layer 1: run 0; saved best Wasserstein 0.00192178; fresh evaluation Wasserstein 0.00227054.
- ZZ, layer 2: run 2; saved best Wasserstein 0.00546233; fresh evaluation Wasserstein 0.0059556.
- ZZ, layer 3: run 2; saved best Wasserstein 0.0015146; fresh evaluation Wasserstein 0.00184713.
- ZZ, layer 4: run 0; saved best Wasserstein 0.00290337; fresh evaluation Wasserstein 0.00334878.
- ZZ-CRX, layer 1: run 1; saved best Wasserstein 0.00234403; fresh evaluation Wasserstein 0.00249277.
- ZZ-CRX, layer 2: run 2; saved best Wasserstein 0.003014; fresh evaluation Wasserstein 0.00340873.
- ZZ-CRX, layer 3: run 1; saved best Wasserstein 0.0042287; fresh evaluation Wasserstein 0.00436925.
- ZZ-CRX, layer 4: run 0; saved best Wasserstein 0.00582886; fresh evaluation Wasserstein 0.00611617.
- ZZ time evolution, layer 1: run 0; saved best Wasserstein 0.00192178; fresh evaluation Wasserstein 0.00227054.
- ZZ time evolution, layer 2: run 2; saved best Wasserstein 0.00166898; fresh evaluation Wasserstein 0.00211123.
- ZZ time evolution, layer 3: run 2; saved best Wasserstein 0.00241616; fresh evaluation Wasserstein 0.00272846.
- ZZ time evolution, layer 4: run 0; saved best Wasserstein 0.00143843; fresh evaluation Wasserstein 0.00176292.
- HEA, layer 1: run 2; saved best Wasserstein 0.00369662; fresh evaluation Wasserstein 0.00400727.
- HEA, layer 2: run 2; saved best Wasserstein 0.00102872; fresh evaluation Wasserstein 0.00106296.
- HEA, layer 3: run 0; saved best Wasserstein 0.000817788; fresh evaluation Wasserstein 0.00108085.
- HEA, layer 4: run 2; saved best Wasserstein 0.000982381; fresh evaluation Wasserstein 0.00136932.

## Depth pattern

- ZZ: L1=0.00192178, L2=0.00546233, L3=0.0015146, L4=0.00290337. Lowest median-run value at layer 3; monotonic improvement with depth: no.
- ZZ-CRX: L1=0.00234403, L2=0.003014, L3=0.0042287, L4=0.00582886. Lowest median-run value at layer 1; monotonic improvement with depth: no.
- ZZ time evolution: L1=0.00192178, L2=0.00166898, L3=0.00241616, L4=0.00143843. Lowest median-run value at layer 4; monotonic improvement with depth: no.
- HEA: L1=0.00369662, L2=0.00102872, L3=0.000817788, L4=0.000982381. Lowest median-run value at layer 3; monotonic improvement with depth: no.

## Diagnostic comparison

- Lowest fresh marginal Wasserstein distance: HEA layer 2 (0.00106296).
- Lowest absolute-ACF discrepancy: HEA layer 4 (0.0255794).
- Lowest return-ACF discrepancy: HEA layer 4 (0.0151771).
- Lowest leverage discrepancy: ZZ time evolution layer 3 (0.00990936).
- ZZ and ZZ time evolution are numerically identical at layer 1 in all reconstructed diagnostics, as required by their shared single-layer circuit definition.
- ZZ-CRX lag-1 leverage estimates are L1=-0.0943, L2=-0.0081, L3=-0.0511, L4=0.0149, versus Bitcoin=-0.0317. The intended negative direction is visible for layers 1–3 but is not uniform at layer 4 and does not establish overall superiority across all lags.

The saved best Wasserstein metric is the critic-independent marginal one-dimensional Wasserstein distance. Stylized-fact discrepancies are separate: each is the mean absolute difference across lags 1–4 between the real and generated median window-level curves.

Confidence bands in the line figures are 95% max-t simultaneous bootstrap intervals across lags 1–4. Real data use a moving-block bootstrap before rebuilding stride-1 16-step windows; generated data resample complete 16-step windows.
