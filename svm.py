import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 路径安全地读取 CSV
csv_path = os.path.join(os.path.dirname(__file__), 'heart_statlog_cleveland_hungary_final.csv')
df = pd.read_csv(csv_path)
df = df.drop_duplicates()

# 特征与标签分离
X = df.drop('target', axis=1)
y = df['target']

# 分割训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM 模型训练
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train_scaled, y_train)

# 预测与评估
y_pred = svm_model.predict(X_test_scaled)
print(f"\n{'='*50}")
print("🎯 SVM 模型评估")
print(f"{'='*50}")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 保存模型 & Scaler
joblib.dump(svm_model, 'heart_disease_svm_model.pkl')
joblib.dump(scaler, 'heart_disease_scaler.pkl')
print("✅ 模型和 Scaler 已保存！")

# 预测函数
def predict_svm(input_features):
    model = joblib.load('heart_disease_svm_model.pkl')
    scaler = joblib.load('heart_disease_scaler.pkl')
    input_scaled = scaler.transform([input_features])
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    return prediction, probability

# 使用示例
example = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 2]
pred, prob = predict_svm(example)
print(f"\n预测结果: {'心脏病' if pred == 1 else '无心脏病'}")
print(f"概率分布: {prob}")


