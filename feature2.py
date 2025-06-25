import pandas as pd
import numpy as np
from scipy import stats
from moving_detect import moving_average,moving_variance

def extract_mv_ma_features2(dataframe, window_size=10, step_size=1):
    features_list = []
    # 只按卫星编号分组
    for sat, group in dataframe.groupby('Satellite'):
        if len(group) >= window_size:
            s1_values = group['S1'].values
            # 计算移动方差
            mv = moving_variance(s1_values, window_size=window_size, step_size=step_size)
            # 计算移动平均值
            ma = moving_average(s1_values, window_size=window_size, step_size=step_size)
            
            # 计算载波相位与伪距一致性
            if 'L1' in dataframe.columns and 'C1' in dataframe.columns:
                # 计算载波相位转换为距离
                group['L1_distance'] = group['L1'] * 0.190293672798365  # 波长
                # 计算载波相位与伪距差异
                group['phase_pr_consistency'] = np.abs(group['L1_distance'] - group['C1'])
                # 使用group的索引从dataframe中获取phase_pr_consistency列的值
                phase_pr_consistency = group['phase_pr_consistency'].rolling(window=window_size, min_periods=1).mean().values
            else:
                phase_pr_consistency = np.full(len(group), np.nan)

            if 'C1' in group.columns:
                # 使用简单的线性拟合计算伪距残差
                if len(group) > 2:
                    x = np.arange(len(group))
                    y = group['C1'].values
                    try:
                        slope, intercept = np.polyfit(x, y, 1)
                        fitted = slope * x + intercept
                        group['pr_residual'] = np.abs(y - fitted)
                        pr_residual = group['pr_residual'].rolling(window=window_size).mean().values
                    except:
                        pr_residual = np.full(len(group), np.nan)
                else:
                    pr_residual = np.full(len(group), np.nan)
            else:
                pr_residual = np.full(len(group), np.nan)
            
            # 计算多普勒变化率
            if 'D1' in group.columns and 'Time' in group.columns:
                # 计算时间差分
                group['time_diff'] = group['Time'].diff().dt.total_seconds()
                # 计算多普勒频移的变化率 dΔf/dt
                group['doppler_rate'] = group['D1'].diff() / group['time_diff']
                # 取绝对值并计算滑动窗口平均
                group['doppler_rate'] = group['doppler_rate'].abs()
                doppler_rate = group['doppler_rate'].rolling(window=window_size).mean().values
            else:
                doppler_rate = np.full(len(group), np.nan)
            
            # 为每个时间点创建特征
            for i, idx in enumerate(group.index):
                if i < len(mv):  # 确保索引在范围内
                    features_list.append({
                        'Satellite': sat,
                        '载噪比移动平均方差': mv[i] if i < len(mv) else np.nan,
                        '载噪比移动平均均值': ma[i] if i < len(ma) else np.nan,
                        '载波相位伪距一致性': phase_pr_consistency[i] if i < len(phase_pr_consistency) else np.nan,
                        '伪距残差': pr_residual[i] if i < len(pr_residual) else np.nan,
                        '多普勒变化率': doppler_rate[i] if i < len(doppler_rate) else np.nan,
                    })
    
    return pd.DataFrame(features_list)