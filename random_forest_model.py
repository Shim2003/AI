import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ✅ 安全读取 CSV
csv_path = os.path.join(os.path.dirname(__file__), 'heart_statlog_cleveland_hungary_final.csv')
df = pd.read_csv(csv_path).drop_duplicates()

# ✅ 特征与目标
X = df.drop('target', axis=1)
y = df['target']

# ✅ 分割训练与测试数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ 自动调参（GridSearchCV）
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)
rf_model = grid_search.best_estimator_

# ✅ 预测
y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# ✅ 评估结果
print("\n" + "="*50)
print("🌲 Random Forest 模型评估（使用最佳参数）")
print("="*50)
print("🎯 Best Parameters:", grid_search.best_params_)
print(f"🎯 Accuracy: {accuracy:.4f}")
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# ✅ 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ✅ 保存模型 & Scaler
joblib.dump(rf_model, 'heart_disease_rf_model.pkl')
joblib.dump(scaler, 'heart_disease_rf_scaler.pkl')
print("✅ 模型和 Scaler 已保存成功！")

# ✅ 预测函数
def predict_rf(input_features):
    model = joblib.load('heart_disease_rf_model.pkl')
    scaler = joblib.load('heart_disease_rf_scaler.pkl')
    input_scaled = scaler.transform([input_features])
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    return prediction, probability

# ✅ 使用示例
example = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 2]
pred, prob = predict_rf(example)
print(f"\n🧪 预测结果: {'心脏病' if pred == 1 else '无心脏病'}")
print(f"📊 概率分布: {prob}")
