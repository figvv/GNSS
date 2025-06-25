import georinex as gr
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from scipy.fft import fft
from datetime import datetime
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from pylab import mpl
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
mpl.rcParams["font.sans-serif"]=["SimHei"]
mpl.rcParams["axes.unicode_minus"]=False
from moving_detect import moving_average,moving_variance

pd.set_option('display.max_columns',5000)
pd.set_option('display.max_rows',5000)
pd.set_option('display.width',5000)
pd.set_option('display.max_colwidth',5000)

# 设置文件路径
# 按照卫星编号分为7个输入文件，每个输入文件有多个xlsx文件
satellite_files = {
    'G03': [r"E:\work\data2\NUM3\ds0.xlsx", r"E:\work\data2\NUM3\ds1.xlsx", r"E:\work\data2\NUM3\ds2.xlsx", r"E:\work\data2\NUM3\ds3.xlsx", r"E:\work\data2\NUM3\ds7.xlsx", r"E:\work\data2\NUM3\ds8.xlsx"],
    'G06': [r"E:\work\data2\NUM6\ds0.xlsx", r"E:\work\data2\NUM6\ds1.xlsx", r"E:\work\data2\NUM6\ds2.xlsx", r"E:\work\data2\NUM6\ds3.xlsx", r"E:\work\data2\NUM6\ds7.xlsx", r"E:\work\data2\NUM6\ds8.xlsx"],
    'G07': [r"E:\work\data2\NUM7\ds0.xlsx", r"E:\work\data2\NUM7\ds1.xlsx", r"E:\work\data2\NUM7\ds2.xlsx", r"E:\work\data2\NUM7\ds3.xlsx", r"E:\work\data2\NUM7\ds7.xlsx", r"E:\work\data2\NUM7\ds8.xlsx"],
    'G13': [r"E:\work\data2\NUM13\ds0.xlsx", r"E:\work\data2\NUM13\ds1.xlsx", r"E:\work\data2\NUM13\ds2.xlsx", r"E:\work\data2\NUM13\ds3.xlsx", r"E:\work\data2\NUM13\ds7.xlsx", r"E:\work\data2\NUM13\ds8.xlsx"],
    'G16': [r"E:\work\data2\NUM16\ds0.xlsx", r"E:\work\data2\NUM16\ds1.xlsx", r"E:\work\data2\NUM16\ds2.xlsx", r"E:\work\data2\NUM16\ds3.xlsx", r"E:\work\data2\NUM16\ds7.xlsx", r"E:\work\data2\NUM16\ds8.xlsx"],
    'G19': [r"E:\work\data2\NUM19\ds0.xlsx", r"E:\work\data2\NUM19\ds1.xlsx", r"E:\work\data2\NUM19\ds2.xlsx", r"E:\work\data2\NUM19\ds3.xlsx", r"E:\work\data2\NUM19\ds7.xlsx", r"E:\work\data2\NUM19\ds8.xlsx"],
    'G23': [r"E:\work\data2\NUM23\ds0.xlsx", r"E:\work\data2\NUM23\ds1.xlsx", r"E:\work\data2\NUM23\ds2.xlsx", r"E:\work\data2\NUM23\ds3.xlsx", r"E:\work\data2\NUM23\ds7.xlsx", r"E:\work\data2\NUM23\ds8.xlsx"]
}
all_data = []

# 读取所有卫星数据文件
for sat_id, file_list in satellite_files.items():
    for file_index, file_path in enumerate(file_list):
        try:
            # 读取xlsx文件
            df_temp = pd.read_excel(file_path)
            
            # 确保数据包含所需的列
            if all(col in df_temp.columns for col in ['Time', 'Satellite', 'C1', 'D1', 'L1', 'S1', 'Index']):
                # 添加文件索引列，使用xlsx的文件名
                file_name = file_path.split('\\')[-1]  # 获取文件名
                df_temp['FileIndex'] = file_name
                # 添加标签列，使用Index列的值
                df_temp['label'] = df_temp['Index']
                all_data.append(df_temp)
            else:
                print(f"文件 {file_path} 缺少必要的列")
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")

# 合并所有数据
df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# 确保数据包含所需的列
required_columns = ['Time', 'C1', 'D1', 'L1', 'S1', 'Satellite', 'label', 'FileIndex']
for col in required_columns:
    if col not in df.columns:
        if col == 'Time':
            df['Time'] = pd.to_datetime('now')  # 默认时间
        else:
            df[col] = np.nan  # 其他列设为NaN

# 重新排列列顺序
df = df[['FileIndex', 'Time', 'Satellite', 'C1', 'D1', 'L1', 'S1', 'label']]

print("处理后的数据集:")
print(df)

# 保留原始分类，用于后续处理
df_normal = df[df['label'] == 9]
df_spoofed = df[df['label'] != 9]

def extract_mv_ma_features(dataframe, window_size=10, step_size=1):
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
                        'label': 0 if group.loc[idx, 'label'] == 9 else 1,  # 9为正常(0)，非9为欺骗(1)
                        'original_label': group.loc[idx, 'label'],  # 保存原始标签用于可视化
                        'FileIndex': group.loc[idx, 'FileIndex']  # 保存文件索引
                    })
    
    return pd.DataFrame(features_list)


# 提取特征
features_df = extract_mv_ma_features(df)
features_df = features_df.dropna()  # 删除包含NaN的行

# 动态设定阈值 - 使用正常数据的统计特性
normal_features = features_df[features_df['label'] == 0]
# 划分训练集和测试集 (70% 训练, 30% 测试)
# 使用载噪比相关特征、载波相位伪距一致性、伪距残差和多普勒变化率特征
X = features_df[['载噪比移动平均均值', '载噪比移动平均方差', '载波相位伪距一致性', '伪距残差', '多普勒变化率']]
y = features_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# 使用自动适应数据分布变化的鲁棒标准化器
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 进行SVM超参数消融实验
print("\n开始SVM超参数消融实验...")

# 定义不同的超参数组合
C_values = [0.1, 1, 10, 100, 1000]
gamma_values = [0.001, 0.01, 0.1, 1, 10]
kernel_types = ['rbf', 'linear', 'poly']

# 存储不同参数组合的结果
ablation_results = []

# 测试不同的C值(固定kernel='rbf', gamma=1)
print("\n测试不同的C值...")
c_scores = []
for c in C_values:
    svm = SVC(C=c, kernel='rbf', gamma=1, probability=True)
    svm.fit(X_train_scaled, y_train)
    y_pred = svm.predict(X_test_scaled)
    score = f1_score(y_test, y_pred)
    c_scores.append(score)
    ablation_results.append({
        'parameter': 'C',
        'value': c,
        'f1_score': score
    })

# 测试不同的gamma值(固定kernel='rbf', C=100)
print("\n测试不同的gamma值...")
gamma_scores = []
for gamma in gamma_values:
    svm = SVC(C=100, kernel='rbf', gamma=gamma, probability=True)
    svm.fit(X_train_scaled, y_train)
    y_pred = svm.predict(X_test_scaled)
    score = f1_score(y_test, y_pred)
    gamma_scores.append(score)
    ablation_results.append({
        'parameter': 'gamma',
        'value': gamma,
        'f1_score': score
    })

# 测试不同的kernel类型(固定C=100, gamma=1)
print("\n测试不同的kernel类型...")
kernel_scores = []
for kernel in kernel_types:
    svm = SVC(C=100, kernel=kernel, gamma=1, probability=True)
    svm.fit(X_train_scaled, y_train)
    y_pred = svm.predict(X_test_scaled)
    score = f1_score(y_test, y_pred)
    kernel_scores.append(score)
    ablation_results.append({
        'parameter': 'kernel',
        'value': kernel,
        'f1_score': score
    })

# 绘制C值的影响
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.plot(C_values, c_scores, marker='o')
plt.xscale('log')
plt.xlabel('C值', fontsize=12)
plt.ylabel('F1分数', fontsize=12)
plt.title('C值对模型性能的影响', fontsize=14)
plt.grid(True)

# 绘制gamma值的影响
plt.subplot(132)
plt.plot(gamma_values, gamma_scores, marker='o')
plt.xscale('log')
plt.xlabel('gamma值', fontsize=12)
plt.ylabel('F1分数', fontsize=12)
plt.title('gamma值对模型性能的影响', fontsize=14)
plt.grid(True)

# 绘制kernel类型的影响
plt.subplot(133)
plt.bar(kernel_types, kernel_scores)
plt.xlabel('核函数类型', fontsize=12)
plt.ylabel('F1分数', fontsize=12)
plt.title('核函数类型对模型性能的影响', fontsize=14)

plt.tight_layout()
plt.show()

# 进行随机森林超参数消融实验
print("\n开始随机森林超参数消融实验...")

# 定义不同的超参数组合
n_estimators_values = [10, 50, 100, 200, 500]
max_depth_values = [None, 5, 10, 15, 20]
min_samples_split_values = [2, 5, 10, 15, 20]

# 存储不同参数组合的结果
rf_ablation_results = []

# 测试不同的n_estimators值
print("\n测试不同的n_estimators值...")
n_estimators_scores = []
for n_est in n_estimators_values:
    rf = RandomForestClassifier(n_estimators=n_est, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    score = f1_score(y_test, y_pred)
    n_estimators_scores.append(score)
    rf_ablation_results.append({
        'parameter': 'n_estimators',
        'value': n_est,
        'f1_score': score
    })

# 测试不同的max_depth值
print("\n测试不同的max_depth值...")
max_depth_scores = []
for depth in max_depth_values:
    rf = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    score = f1_score(y_test, y_pred)
    max_depth_scores.append(score)
    rf_ablation_results.append({
        'parameter': 'max_depth',
        'value': str(depth),
        'f1_score': score
    })

# 测试不同的min_samples_split值
print("\n测试不同的min_samples_split值...")
min_samples_split_scores = []
for min_split in min_samples_split_values:
    rf = RandomForestClassifier(n_estimators=100, min_samples_split=min_split, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    score = f1_score(y_test, y_pred)
    min_samples_split_scores.append(score)
    rf_ablation_results.append({
        'parameter': 'min_samples_split',
        'value': min_split,
        'f1_score': score
# 输出标准化后的数据分布
print("\n标准化后的数据分布:")
print("训练集标准化后的均值:", np.mean(X_train_scaled, axis=0))
print("训练集标准化后的标准差:", np.std(X_train_scaled, axis=0))
print("测试集标准化后的均值:", np.mean(X_test_scaled, axis=0))
print("测试集标准化后的标准差:", np.std(X_test_scaled, axis=0))
# 根据网格搜索结果直接创建SVM模型
# 使用最佳参数：C=100, kernel=rbf, gamma=1, cache_size=2000
print("根据最佳参数创建SVM模型...")
# 直接使用已知的最佳参数创建SVM模型
best_params = {
    'C': 100,
    'kernel': 'rbf',
    'gamma': 1,
    'class_weight': None,
    'cache_size': 2000,
    'verbose': True
}

print("使用最佳参数创建SVM模型...")
print(f"最佳参数: {best_params}")
print(f"最佳C值: {best_params['C']}")
print(f"最佳核函数: {best_params['kernel']}")
print(f"最佳gamma值: {best_params['gamma']}")

# 直接使用最佳参数创建SVM模型
svm = SVC(
    C=best_params['C'],
    kernel=best_params['kernel'],
    gamma=best_params['gamma'],
    class_weight=best_params['class_weight'],
    cache_size=best_params['cache_size'],
    verbose=best_params['verbose'],
    probability=True
)

# 导入所需的评估指标
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc

# 训练模型
svm.fit(X_train_scaled, y_train)

print("SVM模型训练完成")

# 在测试集上评估最佳模型
y_pred = svm.predict(X_test_scaled)
y_prob = svm.predict_proba(X_test_scaled)[:, 1]  # 获取正类的概率
test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_conf_matrix = confusion_matrix(y_test, y_pred)


print("\n最佳模型测试集评估结果:")
print(f"测试集准确率: {test_accuracy:.4f}")
print(f"测试集精确率: {test_precision:.4f}")
print(f"测试集召回率: {test_recall:.4f}")
print(f"测试集F1分数: {test_f1:.4f}")
print("测试集混淆矩阵:")
print(test_conf_matrix)
print("\n测试集分类报告:")
print(classification_report(y_test, y_pred))

# 绘制ROC曲线和计算AUC值
plt.figure(figsize=(10, 8))
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正例率', fontsize=20)
plt.ylabel('真正例率', fontsize=20)
plt.title('接收者操作特征曲线 (ROC)', fontsize=20)
plt.legend(loc="lower right", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

# 可视化准确率、召回率和F1分数
metrics = ['准确率', '精确率', '召回率', 'F1分数']
values = [test_accuracy, test_precision, test_recall, test_f1]

plt.figure(figsize=(12, 8))
bars = plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple'])

# 在柱状图上添加具体数值
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.4f}', ha='center', va='bottom', fontsize=16)

plt.ylim(0, 1.1)
plt.xlabel('评估指标', fontsize=20)
plt.ylabel('分数', fontsize=20)
plt.title('SVM模型性能评估', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 使用随机森林模型进行检测
print("\n使用随机森林模型进行检测")

# 创建随机森林分类器
from sklearn.ensemble import RandomForestClassifier

# 使用已知的最佳参数，不进行网格搜索
print("使用已知的最佳随机森林参数...")
rf_best_params = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'class_weight': None
}

# 创建一个字典对象，模拟GridSearchCV的best_params_属性
class GridSearchResult:
    def __init__(self, best_params):
        self.best_params_ = best_params

# 创建模拟的网格搜索结果对象
rf_grid_search = GridSearchResult(rf_best_params)
print(f"最佳参数: {rf_grid_search.best_params_}")

# 使用最佳参数创建随机森林模型
print("使用最佳参数训练随机森林模型...")
rf = RandomForestClassifier(
    n_estimators=rf_grid_search.best_params_['n_estimators'],
    max_depth=rf_grid_search.best_params_['max_depth'],
    min_samples_split=rf_grid_search.best_params_['min_samples_split'],
    min_samples_leaf=rf_grid_search.best_params_['min_samples_leaf'],
    class_weight=rf_grid_search.best_params_['class_weight'],
    random_state=42
)

# 训练随机森林模型
rf.fit(X_train_scaled, y_train)
print("随机森林模型训练完成")

# 在测试集上评估随机森林模型
rf_y_pred = rf.predict(X_test_scaled)
rf_y_prob = rf.predict_proba(X_test_scaled)[:, 1]
rf_test_accuracy = accuracy_score(y_test, rf_y_pred)
rf_test_precision = precision_score(y_test, rf_y_pred)
rf_test_recall = recall_score(y_test, rf_y_pred)
rf_test_f1 = f1_score(y_test, rf_y_pred)
rf_test_conf_matrix = confusion_matrix(y_test, rf_y_pred)

print("\n随机森林模型测试集评估结果:")
print(f"测试集准确率: {rf_test_accuracy:.4f}")
print(f"测试集精确率: {rf_test_precision:.4f}")
print(f"测试集召回率: {rf_test_recall:.4f}")
print(f"测试集F1分数: {rf_test_f1:.4f}")
print("测试集混淆矩阵:")
print(rf_test_conf_matrix)
print("\n测试集分类报告:")
print(classification_report(y_test, rf_y_pred))

# 可视化准确率、召回率和F1分数
plt.figure(figsize=(10, 6))
metrics = ['准确率', '精确率', '召回率', 'F1分数']
scores = [rf_test_accuracy, rf_test_precision, rf_test_recall, rf_test_f1]
colors = ['blue', 'green', 'red', 'purple']

plt.bar(metrics, scores, color=colors)
for i, v in enumerate(scores):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=14)

plt.ylim(0, 1.1)
plt.xlabel('评估指标', fontsize=20)
plt.ylabel('分数', fontsize=20)
plt.title('随机森林模型性能评估', fontsize=22)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 绘制随机森林的ROC曲线
plt.figure(figsize=(10, 8))
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_y_prob)
rf_roc_auc = auc(rf_fpr, rf_tpr)
plt.plot(rf_fpr, rf_tpr, color='green', lw=2, label=f'随机森林 ROC曲线 (AUC = {rf_roc_auc:.4f})')
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'SVM ROC曲线 (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正例率', fontsize=20)
plt.ylabel('真正例率', fontsize=20)
plt.title('SVM与随机森林的ROC曲线比较', fontsize=20)
plt.legend(loc="lower right", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

# 特征重要性分析
feature_importance = pd.DataFrame({
    '特征': X_train.columns,
    '重要性': rf.feature_importances_
}).sort_values('重要性', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='重要性', y='特征', data=feature_importance)
plt.title('随机森林特征重要性')
plt.tight_layout()
plt.show()

# 随机森林混淆矩阵可视化
plt.figure(figsize=(10, 8))
rf_conf_matrix = confusion_matrix(y_test, rf_y_pred)
sns.heatmap(rf_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['正常', '欺骗'],
            yticklabels=['正常', '欺骗'],
            annot_kws={"size": 16})
plt.title('随机森林混淆矩阵', fontsize=20)
plt.xlabel('预测标签', fontsize=16)
plt.ylabel('真实标签', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()

# 保存随机森林模型
joblib.dump(rf, 'gnss_spoofing_detector_rf.pkl')
print("随机森林模型已保存")



# 将测试集预测结果与原始数据合并
test_indices = X_test.index
test_results = pd.DataFrame({
    'Satellite': features_df.loc[test_indices, 'Satellite'],
    '实际标签': y_test,
    '预测标签': y_pred,
    '载噪比移动平均均值': X_test['载噪比移动平均均值'],
    '载噪比移动平均方差': X_test['载噪比移动平均方差'],
    '载波相位伪距一致性': X_test['载波相位伪距一致性'],
    '伪距残差': X_test['伪距残差'],
    '多普勒变化率': X_test['多普勒变化率'],
    'FileIndex': features_df.loc[test_indices, 'FileIndex']
})


# 找出被检测为欺骗的样本
spoofed_detected = test_results[test_results['预测标签'] == 1]
normal_detected = test_results[test_results['预测标签'] == 0]





# 混淆矩阵可视化
plt.figure(figsize=(10, 8))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['正常', '欺骗'],
            yticklabels=['正常', '欺骗'],
            annot_kws={"size": 16})
plt.title('混淆矩阵', fontsize=20)
plt.xlabel('预测标签', fontsize=16)
plt.ylabel('真实标签', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.show()


# 保存模型
joblib.dump(svm, 'gnss_spoofing_detector_svm.pkl')
joblib.dump(scaler, 'gnss_feature_scaler.pkl')
# 保存训练数据特征用于CORAL领域适应
joblib.dump(X_test_scaled, 'source_data_features.pkl')

print("模型和特征缩放器已保存")
