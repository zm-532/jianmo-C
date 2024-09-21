import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest

# 第一步：读取数据
data = pd.read_excel('附件二（测试集）.xlsx')

# 第二步：数据清洗（缺失值处理）
missing_data = data.isnull().sum()
print("缺失值情况：\n", missing_data)

# 使用插值法填补缺失值
data = data.interpolate()

# 检查插值后的缺失值情况
missing_data_after = data.isnull().sum()
print("插值后缺失值情况：\n", missing_data_after)

# 第三步：异常值检测与处理
def remove_outliers_zscore(df, threshold=3):
    z_scores = np.abs((df - df.mean()) / df.std())
    return df.mask(z_scores > threshold)

# 使用Z-score检测异常值
data.iloc[:, 4:1028] = remove_outliers_zscore(data.iloc[:, 4:1028])

# 将列名全部转换为字符串类型，避免sklearn处理时出现问题
data.columns = data.columns.astype(str)

# 第四步：数据归一化（将数据缩放到[0,1]范围）
scaler = MinMaxScaler(feature_range=(0, 1))
data.iloc[:, 4:1028] = scaler.fit_transform(data.iloc[:, 4:1028])

# 第五步：保存处理后的数据
data.to_excel('附件2处理后.xlsx', index=False)

# 输出前几行，检查处理结果
print(data.head())
