# Data provenance

## Original source

- **Dataset:** [Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)
- **Publisher:** Zielak (Kaggle username `mczielinski`)
- **Source file:** `btcusd_1-min_data.csv`
- **Accessed:** 10 June 2026
- **Original file size:** 384,102,735 bytes
- **Original file SHA-256:**
  `05c048a14b0e9e863623fa9e51cfbda82d10a602fcb967503b1a5b2f84de4fcb`
- **License shown by Kaggle at access:**
  [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/)

The original minute-level file is not committed because of its size. The
attribution above should be retained wherever the source data or derived data
are redistributed.

## Processed data in this repository

`Bitcoin_Data_2020_2026/btc_daily_2020_2026.csv` is the processed dataset used
by all reported experiments. The preprocessing script:

1. selects observations from 1 January 2020 through 1 May 2026;
2. resamples minute observations to daily open, high, low, close, and summed
   BTC volume;
3. calculates `Log_return = log(Close_t / Close_(t-1))`; and
4. drops the initial undefined return.

The resulting file contains 2,312 daily log-return observations dated 2 January
2020 through 1 May 2026. Its SHA-256 checksum is
`403197c6684d76c34b8f7c73e5cc4e997f861384ad03ec8e8405bb1f4a0e553f`.

## Reconstructing the processed file

1. Download the source dataset from Kaggle.
2. Place `btcusd_1-min_data.csv` in `Bitcoin_Data_2020_2026/`.
3. From the repository root, run:

```bash
python Bitcoin_Data_2020_2026/GetBitCointDailyData.py
```

The script writes `Bitcoin_Data_2020_2026/btc_daily_2020_2026.csv`.

The Kaggle dataset may be updated after the access date. For an exact provenance
check, compare the downloaded raw file against the SHA-256 value above. A newer
dataset version can still be processed with the supplied date bounds, but it
may not be byte-for-byte identical to the version used in the study.
