import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import lambertw
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt


def chopchop(data, window_size, stride=1):
    return np.lib.stride_tricks.sliding_window_view(data, window_shape = window_size)[::stride]


def transform(data, delta = 0.5, unity_transform = False):#0.172
    #first normalisation
    mu1 = np.mean(data)
    s1 = np.std(data)
    data_norm1 = (data-mu1)/s1

    #Lambert inverse transformation
    data_lambert = np.real(np.sign(data_norm1)*(lambertw(delta*data_norm1**2)/delta)**(0.5))

    #second normalisation
    mu2 = np.mean(data_lambert)
    s2 = np.std(data_lambert)
    data_norm2 = (data_lambert-mu2)/s2
    
    #transformation to [0,1] range
    minimum = np.quantile(data_norm2, 0.0005)
    maximum = np.quantile(data_norm2, 0.9995)

    if unity_transform:
        # 【核心修改】：将数据完美映射到 [-1, 1] 以匹配量子态物理极限
        data_norm2 = 2.0 * (data_norm2 - minimum) / (maximum - minimum) - 1.0
        # 防止极端离群值超出边界
        data_norm2 = np.clip(data_norm2, -1.0, 1.0)
        
    params = [mu1, s1, mu2, s2, delta, minimum, maximum, unity_transform]
    return data_norm2, params

def inverse_transform(data, params): 
    mu1, s1, mu2, s2, delta, minimum, maximum, unity_transform = params

    if unity_transform:
        # 【核心修改】：将生成器吐出的 [-1, 1] 数据还原到截断前的正态范围
        data_norm2 = (data + 1.0) / 2.0 * (maximum - minimum) + minimum
    else:
        mask1 = data >= minimum
        mask2 = data <= maximum
        data_norm1 = np.where(mask1, data, minimum)
        data_norm2 = np.where(mask2, data_norm1, maximum)

    #invert second normalisation
    data_lambert = data_norm2*s2+mu2

    
    #invert lambert
    data_norm1 = data_lambert*np.exp(delta*data_lambert**2/2)
    #invert first normalisation
    data = data_norm1*s1+mu1
    return data

def load_BTC_lr():
    data_btc = pd.read_csv(Path(__file__).resolve().parents[1] / 'Bitcoin_Data_2020_2026' / 'btc_daily_2020_2026.csv')
    return data_btc['Log_return'].values

def mean_and_error(data, flatten = True):
    #mean and error over axis = 1
    data_average = np.average(data, axis = 1)
    data_err = np.std(data, axis = 1, ddof = 1)/np.sqrt(data.shape[1])
    if flatten:
        data_average, data_err = data_average.flatten(), data_err.flatten()
    return data_average, data_err
