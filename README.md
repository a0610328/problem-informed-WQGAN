# Problem-informed WQGAN for financial returns

This repository contains the code, processed data, compact model artifacts, and
analysis outputs supporting **“Toward Interpretable Quantum Generators for
Financial Returns through Problem-Informed Circuit Design.”**

The experiments train hybrid Wasserstein quantum generative adversarial
networks (WQGANs) to generate 16-day Bitcoin return windows. The quantum
generators comprise a ZZ construction, a directed ZZ–CRX extension, and a
structured time-evolution construction. A hardware-efficient ansatz (HEA),
temporal convolutional neural network (TCNN), and two-parameter classical
generator are included as benchmarks.

## Repository contents

```text
Bitcoin_Data_2020_2026/
  btc_daily_2020_2026.csv       Processed daily OHLCV data and log returns
  GetBitCointDailyData.py       Raw-to-daily preprocessing script
  inspect_real_data.ipynb       Real-data diagnostics
  real_data_inspection_figures_2020_2026/  Summary metric CSV files
project_model_ZZ/               Problem-informed ZZ generator
project_model_ZZ_CRZ/           Directed ZZ–CRX generator (legacy folder name)
project_model_ZZ_time_evolution/ Structured time-evolution generator
baseline_model_HEA/             Hardware-efficient quantum baseline
classical_baseline/             TCNN and two-parameter classical baselines
analysis/                       Cross-model analysis and summary tables
DATA.md                         Source-data provenance and licensing
requirements.txt                Verified Python dependency versions
```

Each model directory retains its training code, recorded loss histories,
best-metric summaries, compact best-generator parameter files, generated-window
caches needed for analysis, and final performance figures. Large critic states
and redundant training artifacts are deliberately omitted; see
[Compact repository scope](#compact-repository-scope).

## Environment

The project was verified with Python 3.10.21. Create an isolated environment
and install the recorded dependencies:

```bash
conda create -n problem-informed-wqgan python=3.10.21
conda activate problem-informed-wqgan
python -m pip install -r requirements.txt
```

TensorFlow and PennyLane compatibility can depend on the operating system and
available CPU/GPU drivers. The reported experiments were run in the original
`qgan_clean` environment using the versions pinned in `requirements.txt`.

## Data

All models use `Bitcoin_Data_2020_2026/btc_daily_2020_2026.csv`. It contains
2,312 daily log returns from 2 January 2020 through 1 May 2026. The models use
windows of length 16 with stride 1.

The original minute-level file is not redistributed in this repository because
of its size. It is publicly available as `btcusd_1-min_data.csv` in Zielak's
*Bitcoin Historical Data* dataset on Kaggle. Full provenance, checksum, access
date, license information, and reconstruction instructions are in [DATA.md](DATA.md).

## Reproducing the analyses

Run commands from the repository root. A lightweight check of the retained
classical experiment records is:

```bash
python analysis/analyze_classical_baselines.py --readiness-check
```

Regenerate the four-quantum-model analysis and summary outputs with:

```bash
python analysis/analyze_four_quantum_model_complete.py
```

Regenerate the classical benchmark analysis with:

```bash
python analysis/analyze_classical_baselines.py
```

These commands write figures and summaries into the corresponding
`performance_analysis/` and `analysis/` directories. The fixed sweep names used
by the quantum analysis are recorded in
`analysis/analyze_four_model_performance.py`.

## Training from scratch

The four quantum entry points are the respective `main_BTC.py` files. Run an
entry point from its model directory so newly generated sweep paths remain
local to that model. For example:

```bash
cd project_model_ZZ
python main_BTC.py
```

The classical entry points are:

```bash
cd classical_baseline
python train_tcnn_wgan.py
python train_two_parameter_wgan.py
```

Full sweeps are computationally demanding. The default quantum setup uses 8
qubits, 16-step windows, four depths, three seeds, and up to 4,001 epochs.

## Compact repository scope

To keep the repository suitable for ordinary GitHub hosting, it includes the
scientifically relevant compact artifacts but excludes:

- the original minute-level CSV;
- discriminator/critic checkpoints;
- epoch-by-epoch generator checkpoints;
- TensorBoard event files and other `.v2` training states;
- per-epoch QQ plots;
- large interactive HTML diagnostics; and
- cache folders and editor metadata.

The retained `lowest_wass_generator.pkl` and
`generator_lowest_wass.pkl` files contain the best-generator parameters. The
retained loss histories, manifests, summary tables, generated-window arrays,
and final figures allow the reported comparisons to be inspected without the
full training archive. Reproducing every optimization trajectory from its
intermediate state requires retraining.

## Availability statement

The processed daily Bitcoin return data, preprocessing code, model training and
evaluation code, selected generator parameters, summary outputs, and final
figures are provided in this repository. The original minute-level data are
available from the source identified in [DATA.md](DATA.md).

## Citation and license

The source code, notebooks, and compact model parameter files are distributed
under the [MIT License](LICENSE). Copyright in the modifications and original
contributions is held by Huijie Guan. Portions of the training framework were
adapted from the MIT-licensed `Full_state` implementation in
[LucasAugustusvd/Quantum-Finance](https://github.com/LucasAugustusvd/Quantum-Finance/tree/main/Full_state);
the upstream notice is preserved in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
and [LICENSE](LICENSE).

The processed Bitcoin data are not relicensed under MIT. Their source,
CC BY-SA 4.0 terms, attribution, and checksum are described in
[DATA.md](DATA.md). Third-party dependencies remain under their respective
licenses. Citation metadata and the manuscript author/contribution statement
will be added when the author list is finalized.
