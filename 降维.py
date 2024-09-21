import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 读取数据
data = pd.read_excel('附件2处理后.xlsx')

# 提取特征列（假设采样点在第5列到第1028列）
sampling_points = data.iloc[:, 4:1028]

# 特征标准化
scaler = StandardScaler()
scaled_features = scaler.fit_transform(sampling_points)

# 检查是否存在 NaN 值
print("NaN values before imputation:", np.isnan(scaled_features).sum())

# 使用 SimpleImputer 填充 NaN 值
imputer = SimpleImputer(strategy='mean')
scaled_features_filled = imputer.fit_transform(scaled_features)

# 再次检查 NaN 是否已被填充
print("NaN values after imputation:", np.isnan(scaled_features_filled).sum())

# 确保没有 NaN 之后进行 PCA 降维
pca = PCA(n_components=10)
pca_features = pca.fit_transform(scaled_features_filled)

# 输出 PCA 降维后的形状
print("PCA 降维后的形状：", pca_features.shape)
# 将 PCA 降维后的数据转换为 DataFrame
pca_df = pd.DataFrame(pca_features, columns=[f'PC{i+1}' for i in range(10)])

# 保存降维后的数据为 Excel 文件
pca_df.to_excel('附件2PCA_降维后的数据.xlsx', index=False)

# 如果需要保存为 CSV 文件，可以使用以下代码
# pca_df.to_csv('PCA_降维后的数据.csv', index=False)

print("降维后的数据已保存到 'PCA_降维后的数据.xlsx'")