import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# 第一步：读取数据（需要将数据文件路径替换为你的实际路径）
# 这里以附件一为例，假设文件为 CSV 格式
data = pd.read_excel('附件一（训练集）.xlsx')

# 第二步：数据预处理
# 假设前几列是温度、频率、损耗、波形类型，后面是1024个磁通密度点
# 特征提取：使用磁通密度数据的统计特征，例如峰值、斜率、标准差等

def extract_features(df):
    features = []
    for i, row in df.iterrows():
        magnetic_density = row[4:1028]  # 磁通密度从第5列到1028列
        peak_value = np.max(magnetic_density)  # 峰值
        mean_value = np.mean(magnetic_density)  # 均值
        std_value = np.std(magnetic_density)  # 标准差
        slope = (magnetic_density.iloc[-1] - magnetic_density.iloc[0]) / len(magnetic_density)  # 斜率
        features.append([peak_value, mean_value, std_value, slope])
    return np.array(features)


# 提取训练集的特征
X = extract_features(data)
# 打印列名，检查是否存在 '波形类型'
print(data.columns)

# 假设波形类型的实际列名为 'wave_type'



y = data['励磁波形']  # 假设第4列是波形类型

# 第三步：划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 第四步：训练分类模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 第五步：预测与评估
y_pred = model.predict(X_test)

# 评估模型效果
print("分类准确率:", accuracy_score(y_test, y_pred))
print("分类报告:\n", classification_report(y_test, y_pred))

# 如果需要将模型应用到附件二的测试集数据上，可以使用相同的提取特征方式：
# test_data = pd.read_csv('/path_to_data/附件二.csv')
# X_test_data = extract_features(test_data)
# test_predictions = model.predict(X_test_data)

# 保存预测结果
# test_data['波形预测'] = test_predictions
# test_data.to_csv('/path_to_data/附件四.csv', index=False)
