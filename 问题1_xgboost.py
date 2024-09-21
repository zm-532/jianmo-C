import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder  # 新增这一行，用于标签编码

# 特征提取函数
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

# 读取数据
data = pd.read_excel('附件一（训练集）.xlsx')

# 提取特征
X = extract_features(data)
y = data.iloc[:, 3]  # 假设第4列是波形类型

# 将波形类型标签进行编码，将字符串类型转换为数值
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # 将 ['三角波', '梯形波', '正弦波'] 转换为 [0, 1, 2]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 训练 XGBoost 分类模型
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)

# 预测与评估
y_pred = xgb_model.predict(X_test)

# 输出分类准确率与分类报告
print("分类准确率:", accuracy_score(y_test, y_pred))
print("分类报告:\n", classification_report(y_test, y_pred))

# 如果需要将数值标签转换回原始字符串标签，可以使用以下方式：
# original_labels = label_encoder.inverse_transform(y_pred)
