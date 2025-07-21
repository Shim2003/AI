from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd

# 读入数据
df = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")
X = df.drop(columns=["target"])
y = df["target"]

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 建立 SVM 模型，启用概率输出
svm_model_prob = SVC(kernel='rbf', probability=True, random_state=42)
svm_model_prob.fit(X_train_scaled, y_train)

# 预测概率（为正类的概率）
y_scores = svm_model_prob.predict_proba(X_test_scaled)[:, 1]

# 计算 ROC 曲线和 AUC
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# 画图
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='orange')
plt.plot([0, 1], [0, 1], 'k--')  # 参考线
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - SVM')
plt.legend(loc='lower right')
plt.grid()
plt.show()
