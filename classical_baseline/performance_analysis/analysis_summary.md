# Classical baseline analysis

TCNN is a four-width classical WGAN family (widths 2, 4, 6, and 8). Each width is ranked independently by fresh-sample marginal W1; a family-level representative is the width whose middle run has the lowest fresh-sample marginal W1. The sweep is a bounded parameter-budget study, not an optimized classical scaling frontier.

All reported evaluation Wasserstein distances are means over independent fresh samples drawn from each saved best-Wasserstein generator checkpoint. The batch used during training to select that checkpoint is retained only as an audit column.

The Q-Q RMSE and window-respecting absolute-ACF, raw-ACF, and leverage estimators use the same quantile grid and formulas as the quantum-model analysis.

## Representative runs

| Configuration | Parameters | Best run | Middle run | Worst run | Middle fresh W1 |
|---|---:|---:|---:|---:|---:|
| TCNN-2 | 29 | 0 | 2 | 1 | 0.0013500796 |
| TCNN-4 | 81 | 2 | 1 | 0 | 0.0010155686 |
| TCNN-6 | 157 | 0 | 2 | 1 | 0.00086325749 |
| TCNN-8 | 257 | 2 | 0 | 1 | 0.0013055446 |
| Two-parameter | 2 | 0 | 1 | 2 | 0.006070474 |
