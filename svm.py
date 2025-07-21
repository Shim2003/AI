import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. 读取数据
df = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")

# 2. 分离特征与目标列
X = df.drop(columns=['target'])
y = df['target']

# 3. 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 标准化处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. SVM 模型训练（RBF kernel）
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# 6. 预测
y_pred = svm_model.predict(X_test_scaled)

# 7. 性能评估报告
print(classification_report(y_test, y_pred))

# 8. 混淆矩阵
cm = confusion_matrix(y_test, y_pred, labels=svm_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model.classes_)

# 9. 绘制混淆矩阵
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap='Blues')
plt.title("Confusion Matrix - SVM")
plt.grid(False)
plt.show()
