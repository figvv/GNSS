import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




def moving_average(signal, window_size, step_size):
    """
    计算信号的移动平均值
    :param signal: 输入信号（一维数组）
    :param window_size: 窗口大小
    :param step_size: 步长，默认为1
    :return: 移动平均值
    """
    # 确保窗口大小不超过信号长度
    if window_size > len(signal):
        raise ValueError("窗口大小不能大于信号长度")
    
    # 使用滑动窗口计算移动平均值
    moving_average = []
    for n in range(0, len(signal) - window_size + 1, step_size):
        start = n
        end = n + window_size
        window = signal[start:end]
        mean = np.mean(window)
        moving_average.append(mean)
    return np.array(moving_average)

def moving_variance(signal, window_size, step_size):
    """
    计算信号的移动方差
    :param signal: 输入信号（一维数组）
    :param window_size: 窗口大小
    :param step_size: 步长，默认为1
    :return: 移动方差
    """
    moving_variances = []
    moving_means = moving_average(signal, window_size, step_size)
    for n in range(0, len(signal) - window_size + 1, step_size):
        start = n
        end = n + window_size
        window = signal[start:end]
        mean = moving_means[n // step_size]
        variance = np.mean((window - mean) ** 2)
        moving_variances.append(variance)
    return np.array(moving_variances)

