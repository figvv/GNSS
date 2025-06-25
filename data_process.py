import georinex as gr
import pandas as pd
import os
from datetime import datetime

import pandas as pd

def parse_rinex_obs(file_path):
    """
    解析RINEX观测文件，提取观测数据
    
    参数:
        file_path: RINEX文件路径
    返回:
        包含观测数据的DataFrame
    """
    # 使用georinex库直接解析RINEX文件
    try:
        # 读取观测数据
        obs_data = gr.load(file_path)
        
        # 转换为DataFrame格式
        df = obs_data.to_dataframe().reset_index()
        
        # 重命名列以便于理解
        if 'time' in df.columns:
            df.rename(columns={'time': 'Time'}, inplace=True)
        if 'sv' in df.columns:
            df.rename(columns={'sv': 'Satellite'}, inplace=True)
            
        return df
        
    except Exception as e:
        print(f"解析RINEX文件时出错: {e}")
        
        # 如果georinex解析失败，使用备用方法解析
        return parse_rinex_obs_manual(file_path)

def parse_rinex_obs_manual(file_path):
    """
    手动解析RINEX观测文件的备用方法
    """
    obs_types = []
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # 解析头文件获取观测类型
        header_ended = False
        for line in f:
            if "END OF HEADER" in line:
                header_ended = True
                break
            if "TYPES OF OBSERV" in line or "# / TYPES OF OBSERV" in line:
                n = int(line[0:6].strip())
                types = line[6:60].split()
                while len(types) < n:
                    next_line = next(f)
                    types += next_line[6:60].split()
                obs_types = types[:n]  # 确保数量正确
        
        # 解析数据部分
        current_epoch = None
        if not header_ended:
            return pd.DataFrame()  # 如果没有找到头文件结束标记，返回空DataFrame
            
        try:
            while True:
                line = next(f, None)
                if line is None:
                    break
                    
                # 检查是否为时间行（RINEX 2.x格式）
                if line[0] == ' ' and line[1:3].strip().isdigit():
                    # RINEX 2.x格式的时间行
                    year = int(line[1:3])
                    # 处理两位数年份
                    if year < 80:  # 假设80-99是1900年代，00-79是2000年代
                        year += 2000
                    else:
                        year += 1900
                    month = int(line[4:6])
                    day = int(line[7:9])
                    hour = int(line[10:12])
                    minute = int(line[13:15])
                    second = float(line[16:26])
                    
                    # 创建标准格式的时间字符串
                    time_str = f"{year}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:09.6f}"
                    
                    # 获取卫星数量
                    flag = int(line[28])
                    num_sats = int(line[29:32])
                    
                    # 获取卫星列表
                    sats = []
                    for i in range(0, num_sats):
                        if i < 12:  # 第一行最多12颗卫星
                            if 32 + i*3 + 3 <= len(line):
                                sat = line[32 + i*3:32 + i*3 + 3].strip()
                                if sat:
                                    sats.append(sat)
                        else:
                            # 读取额外的卫星行
                            sat_line_index = (i - 12) // 12
                            sat_pos_in_line = (i - 12) % 12
                            
                            while len(sats) < i + 1:
                                next_sat_line = next(f)
                                for j in range(12):
                                    if 32 + j*3 + 3 <= len(next_sat_line):
                                        sat = next_sat_line[32 + j*3:32 + j*3 + 3].strip()
                                        if sat:
                                            sats.append(sat)
                    
                    current_epoch = {
                        'time': time_str,
                        'sats': sats[:num_sats]
                    }
                    
                    # 读取每颗卫星的观测数据
                    for sat in current_epoch['sats']:
                        obs_values = []
                        obs_line = next(f)
                        
                        # 计算需要读取的行数
                        lines_needed = (len(obs_types) + 4) // 5  # 每行最多5个观测值
                        
                        # 读取第一行数据
                        for i in range(min(5, len(obs_types))):
                            if i*16 + 14 <= len(obs_line):
                                obs_val = obs_line[i*16:i*16 + 14].strip()
                                obs_values.append(obs_val if obs_val else '0')
                            else:
                                obs_values.append('0')
                        
                        # 读取额外的行
                        for _ in range(1, lines_needed):
                            obs_line = next(f)
                            for i in range(5):  # 每行最多5个观测值
                                if len(obs_values) < len(obs_types):
                                    if i*16 + 14 <= len(obs_line):
                                        obs_val = obs_line[i*16:i*16 + 14].strip()
                                        obs_values.append(obs_val if obs_val else '0')
                                    else:
                                        obs_values.append('0')
                        
                        # 确保观测值数量与观测类型数量一致
                        while len(obs_values) < len(obs_types):
                            obs_values.append('0')
                            
                        # 添加到数据列表
                        data.append([current_epoch['time'], sat] + obs_values)
                
                # 检查是否为RINEX 3.x格式的时间行
                elif line.startswith('>'):
                    parts = line[1:].split()
                    if len(parts) >= 6:
                        year = int(parts[0])
                        month, day, hour, minute = map(int, parts[1:5])
                        second = float(parts[5])
                        
                        # 创建标准格式的时间字符串
                        time_str = f"{year}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:09.6f}"
                        
                        if len(parts) > 7:
                            num_sats = int(parts[7])
                            sats = parts[8:] if len(parts) > 8 else []
                            
                            # 处理卫星数跨行的情况
                            while len(sats) < num_sats:
                                next_line = next(f).strip()
                                sats += next_line.split()
                                
                            current_epoch = {
                                'time': time_str,
                                'sats': sats[:num_sats]
                            }
                            
                            # 读取每颗卫星的观测数据
                            for sat_index, sat in enumerate(current_epoch['sats']):
                                line_obs = next(f)
                                obs_values = []
                                
                                # 每行最多5个观测值，每个占16字符
                                for i in range(0, min(5, len(obs_types))):
                                    if i*16 + 16 <= len(line_obs):
                                        obs = line_obs[i*16:i*16 + 16].strip()
                                        obs_values.append(obs if obs else '0')
                                    else:
                                        obs_values.append('0')
                                
                                # 补充读取额外的行（如果观测类型超过5个）
                                lines_needed = (len(obs_types) - 5 + 4) // 5
                                for _ in range(lines_needed):
                                    line_obs = next(f)
                                    for i in range(0, 5):
                                        if len(obs_values) < len(obs_types):
                                            if i*16 + 16 <= len(line_obs):
                                                obs = line_obs[i*16:i*16 + 16].strip()
                                                obs_values.append(obs if obs else '0')
                                            else:
                                                obs_values.append('0')
                                
                                # 确保观测值数量与观测类型数量一致
                                while len(obs_values) < len(obs_types):
                                    obs_values.append('0')
                                    
                                # 添加到数据列表
                                data.append([current_epoch['time'], sat] + obs_values)
        except Exception as e:
            print(f"解析数据时出错: {e}")
    
    # 创建DataFrame
    if not data:
        return pd.DataFrame()
        
    columns = ['Time', 'Satellite'] + obs_types
    df = pd.DataFrame(data, columns=columns)
    
    # 尝试将观测值转换为数值类型
    for col in obs_types:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass
            
    return df

# 使用示例
df = parse_rinex_obs('E:/work/1/1/ds8/GSDR192w23.22O')
df.to_excel('E:/work/ds8.xlsx', index=False, engine='openpyxl')
