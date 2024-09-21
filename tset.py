import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib  # 用于保存和加载模型

# 特征提取函数
def extract_features(df):
    features = []
    for i, row in df.iterrows():
        magnetic_density = row[4:8]  # 磁通密度从第5列到1028列
        peak_value = np.max(magnetic_density)  # 峰值
        mean_value = np.mean(magnetic_density)  # 均值
        std_value = np.std(magnetic_density)  # 标准差
        slope = (magnetic_density.iloc[-1] - magnetic_density.iloc[0]) / len(magnetic_density)  # 斜率
        features.append([peak_value, mean_value, std_value, slope])
    return np.array(features)

# 第一步：读取附件一数据并训练模型
# 假设附件一的数据为 '附件一（训练集）.xlsx'
data = pd.read_excel('X.xlsx')

# 提取特征
X = extract_features(data)
y = data.iloc[:, 3]  # 第4列假设是波形类型

# 将波形类型编码为数值
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 训练XGBoost分类模型
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)

# 保存模型和标签编码器，以便后续使用
joblib.dump(xgb_model, 'xgb_wave_classifier2.pkl')
joblib.dump(label_encoder, 'label_encoder2.pkl')

# 第三步：读取附件二数据并提取特征
# 假设附件二的数据为 '附件二（测试集）.xlsx'
test_data = pd.read_excel('附件二（测试集）.xlsx')

# 提取测试集的特征
X_test_data = extract_features(test_data)

# 加载训练好的模型和标签编码器
xgb_model = joblib.load('xgb_wave_classifier.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# 对测试数据进行波形分类预测
test_predictions = xgb_model.predict(X_test_data)

# 将数值型分类结果转换回原始的波形类型
predicted_wave_types = label_encoder.inverse_transform(test_predictions)

# 第四步：将结果保存到Excel文件
# 假设附件二的结构与附件一类似，第一列是样本序号，第二列是温度，第三列是频率
test_data['波形预测'] = predicted_wave_types  # 添加预测结果列
test_data[['序号', '波形预测']].to_excel('附件四-2（波形分类结果）.xlsx', index=False)

print("分类预测完成，结果已保存到 '附件四（波形分类结果）.xlsx'")
