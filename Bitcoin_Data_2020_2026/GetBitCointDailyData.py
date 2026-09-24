import pandas as pd
import numpy as np
from pathlib import Path

if __name__ == '__main__':
    start_time = "2020-01-01"
    end_time = "2026-05-01"
    data_file = Path(__file__).resolve().parent / "btcusd_1-min_data.csv"
    df = pd.read_csv(data_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit = 's')
    df.set_index('Timestamp', inplace = True)
    df_filtered = df.loc[start_time : end_time].copy()
    df_daily = df_filtered.resample('1D').agg({
        "Open":"first",
        "High": "max", 
        "Low": "min",
        "Close": "last",
        "Volume": 'sum'})
    #df_daily.describe()
    df_daily['Log_return'] = np.log(df_daily['Close']/df_daily['Close'].shift(1))
    df_daily.dropna(inplace = True)
    df_daily.to_csv(Path(__file__).resolve().with_name('btc_daily_2020_2026.csv'))
