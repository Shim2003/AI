import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report

# -----------------------
# Load Data
# -----------------------
df = pd.read_csv('cleaned_heart_disease_data.csv')
X = df.drop(columns=["target"])
y = df["target"]

X_train = X_test = y_train = y_test = None
svm_model = None
scaler = None

# -----------------------
# Tkinter GUI
# -----------------------
root = tk.Tk()
root.title("Heart Disease Prediction (SVM)")
root.geometry("850x650")

frame = tk.Frame(root)
frame.pack(expand=True, fill="both", padx=20, pady=20)

title = tk.Label(frame, text="Heart Disease SVM Trainer", font=("Arial", 16, "bold"))
title.pack(pady=10)

# -----------------------
# Step 1: Train/Test Split
# -----------------------
def do_split():
    global X_train, X_test, y_train, y_test
    try:
        test_ratio = 1 - float(train_ratio_var.get())
        if not (0.05 <= test_ratio <= 0.5):
            raise ValueError("Train ratio should give test size between 0.05 and 0.5")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, stratify=y, random_state=42  # ✅ 固定 random_state 保证复现
        )
        messagebox.showinfo("Split Success",
                            f"✅ Train/Test Split Done\n"
                            f"Train size: {X_train.shape[0]}\n"
                            f"Test size: {X_test.shape[0]}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

tk.Label(frame, text="1. Train/Test Split", font=("Arial", 12, "bold")).pack(pady=5)
split_frame = tk.Frame(frame)
split_frame.pack(pady=5)

train_ratio_var = tk.StringVar(value="0.8")
tk.Label(split_frame, text="Train Ratio (0.5 - 0.95):").pack(side="left", padx=5)
tk.Entry(split_frame, textvariable=train_ratio_var, width=10).pack(side="left")
tk.Button(split_frame, text="Run Split", command=do_split).pack(side="left", padx=10)

# -----------------------
# Step 2: Train SVM Model
# -----------------------
def train_model():
    global svm_model, scaler
    if X_train is None:
        messagebox.showwarning("Warning", "⚠️ Please split the data first!")
        return
    try:
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train SVM (保持和 standalone 一致)
        svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        svm_model.fit(X_train_scaled, y_train)

        preds = svm_model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds)

        # ✅ 显示到小数点 10 位
        result_text.set(f"✅ Model Trained!\nTest Accuracy: {acc:.10f}\n\nReport:\n{report}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

tk.Label(frame, text="2. Train SVM Model", font=("Arial", 12, "bold")).pack(pady=5)
tk.Button(frame, text="Train Model", command=train_model).pack(pady=5)
result_text = tk.StringVar()
tk.Label(frame, textvariable=result_text, justify="left", font=("Courier", 10)).pack(pady=5)

# -----------------------
# Step 3: Show ROC Curve
# -----------------------
def show_roc():
    if svm_model is None:
        messagebox.showwarning("Warning", "Train the model first!")
        return
    try:
        X_test_scaled = scaler.transform(X_test)
        y_scores = svm_model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})', color='orange')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve - SVM')
        ax.legend(loc='lower right')
        ax.grid()

        roc_win = tk.Toplevel(root)
        roc_win.title("ROC Curve")
        canvas = FigureCanvasTkAgg(fig, master=roc_win)
        canvas.draw()
        canvas.get_tk_widget().pack()
    except Exception as e:
        messagebox.showerror("Error", str(e))

tk.Label(frame, text="3. ROC Curve", font=("Arial", 12, "bold")).pack(pady=5)
tk.Button(frame, text="Show ROC Curve", command=show_roc).pack(pady=5)

# -----------------------
# Step 4: Save Model
# -----------------------
def save_model():
    if svm_model is None:
        messagebox.showwarning("Warning", "Train the model first!")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".pkl",
                                             filetypes=[("Pickle Files", "*.pkl")])
    if file_path:
        joblib.dump({"model": svm_model, "scaler": scaler}, file_path)
        messagebox.showinfo("Saved", f"✅ Model and Scaler saved at:\n{file_path}")

tk.Label(frame, text="4. Save Model", font=("Arial", 12, "bold")).pack(pady=5)
tk.Button(frame, text="Save Trained Model", command=save_model).pack(pady=5)

# -----------------------
# Dataset Info at Bottom
# -----------------------
info_frame = tk.Frame(frame)
info_frame.pack(side="bottom", fill="x", pady=10)
tk.Label(info_frame, text=f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns",
         font=("Arial", 10), fg="gray").pack()

root.mainloop()
