import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# -----------------------
# Load Data
# -----------------------
df = pd.read_csv('cleaned_heart_disease_data.csv')
X = df.drop(columns=["target"])
y = df["target"]

X_train = X_test = y_train = y_test = None
rf_model = None
scaler = None

# -----------------------
# Tkinter GUI
# -----------------------
root = tk.Tk()
root.title("Heart Disease Prediction (Random Forest)")
root.geometry("850x650")

canvas = tk.Canvas(root)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

main_frame = tk.Frame(scrollable_frame)
main_frame.pack(expand=True, fill="both", padx=20, pady=20)

title = tk.Label(main_frame, text="Heart Disease Random Forest Trainer", font=("Arial", 16, "bold"))
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
            X, y, test_size=test_ratio, stratify=y, random_state=42
        )
        messagebox.showinfo("Split Success",
                            f"✅ Train/Test Split Done\n"
                            f"Train size: {X_train.shape[0]}\n"
                            f"Test size: {X_test.shape[0]}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

tk.Label(main_frame, text="1. Train/Test Split", font=("Arial", 12, "bold")).pack(pady=5)
split_frame = tk.Frame(main_frame)
split_frame.pack(pady=5)
train_ratio_var = tk.StringVar(value="0.8")
tk.Label(split_frame, text="Train Ratio (0.5 - 0.95):").pack(side="left", padx=5)
tk.Entry(split_frame, textvariable=train_ratio_var, width=10).pack(side="left")
tk.Button(split_frame, text="Run Split", command=do_split).pack(side="left", padx=10)

# -----------------------
# Step 2: Train Random Forest Model
# -----------------------
def train_rf_model():
    global rf_model, scaler
    if X_train is None:
        messagebox.showwarning("Warning", "⚠️ Please split the data first!")
        return
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)

        preds = rf_model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds)

        result_text.set(f"✅ RF Model Trained!\nTest Accuracy: {acc:.4f}\n\nReport:\n{report}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

tk.Label(main_frame, text="2. Train Random Forest Model", font=("Arial", 12, "bold")).pack(pady=5)
tk.Button(main_frame, text="Train Model", command=train_rf_model).pack(pady=5)
result_text = tk.StringVar()
tk.Label(main_frame, textvariable=result_text, justify="left", font=("Courier", 10)).pack(pady=5)

# -----------------------
# Step 2.1: Random Training (Incremental Seed)
# -----------------------
def random_training_rf():
    if X_train is None:
        messagebox.showwarning("Warning", "⚠️ Please split the data first!")
        return
    try:
        n_runs = int(random_runs_var.get())
        acc_list = []
        train_ratio = float(train_ratio_var.get())
        test_ratio = 1 - train_ratio

        for i in range(n_runs):
            rs = i  # Incremental seed
            X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
                X, y, test_size=test_ratio, stratify=y, random_state=rs
            )

            scaler_r = StandardScaler()
            X_train_scaled_r = scaler_r.fit_transform(X_train_r)
            X_test_scaled_r = scaler_r.transform(X_test_r)

            rf_model_r = RandomForestClassifier(n_estimators=100, random_state=rs)
            rf_model_r.fit(X_train_scaled_r, y_train_r)

            preds_r = rf_model_r.predict(X_test_scaled_r)
            acc_list.append(accuracy_score(y_test_r, preds_r))

        avg_acc = np.mean(acc_list)
        std_acc = np.std(acc_list)
        random_result_text.set(f"✅ RF Random Training ({n_runs} runs)\nAverage Accuracy: {avg_acc:.4f}\nStd Dev: {std_acc:.4f}")

        # Plot line chart
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(range(1, n_runs+1), acc_list, marker='o', linestyle='-', color='green')
        ax.set_xlabel("Run Number")
        ax.set_ylabel("Accuracy")
        ax.set_title("RF Accuracy per Run")
        ax.grid(True)

        chart_win = tk.Toplevel(root)
        chart_win.title("RF Accuracy per Run")
        canvas = FigureCanvasTkAgg(fig, master=chart_win)
        canvas.draw()
        canvas.get_tk_widget().pack()
    except Exception as e:
        messagebox.showerror("Error", str(e))

tk.Label(main_frame, text="2.1 Random Training (Incremental Seed)", font=("Arial", 12, "bold")).pack(pady=5)
random_frame = tk.Frame(main_frame)
random_frame.pack(pady=5)
random_runs_var = tk.StringVar(value="10")
tk.Label(random_frame, text="Number of runs:").pack(side="left", padx=5)
tk.Entry(random_frame, textvariable=random_runs_var, width=10).pack(side="left")
tk.Button(random_frame, text="Run Random Training", command=random_training_rf).pack(side="left", padx=10)
random_result_text = tk.StringVar()
tk.Label(main_frame, textvariable=random_result_text, justify="left", font=("Courier", 10)).pack(pady=5)

# -----------------------
# Step 3: Confusion Matrix
# -----------------------
def show_confusion():
    if rf_model is None:
        messagebox.showwarning("Warning", "Train the model first!")
        return
    try:
        X_test_scaled = scaler.transform(X_test)
        preds = rf_model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, preds)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                    xticklabels=["No Disease", "Disease"],
                    yticklabels=["No Disease", "Disease"], ax=ax)
        ax.set_title("Confusion Matrix - RF")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        plt.tight_layout()
        cm_win = tk.Toplevel(root)
        cm_win.title("Confusion Matrix")
        canvas = FigureCanvasTkAgg(fig, master=cm_win)
        canvas.draw()
        canvas.get_tk_widget().pack()
    except Exception as e:
        messagebox.showerror("Error", str(e))

tk.Label(main_frame, text="3. Confusion Matrix", font=("Arial", 12, "bold")).pack(pady=5)
tk.Button(main_frame, text="Show Confusion Matrix", command=show_confusion).pack(pady=5)

# -----------------------
# Step 4: ROC Curve
# -----------------------
def show_roc():
    if rf_model is None:
        messagebox.showwarning("Warning", "Train the model first!")
        return
    try:
        X_test_scaled = scaler.transform(X_test)
        y_scores = rf_model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})', color='green')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve - RF')
        ax.legend(loc='lower right')
        ax.grid()

        roc_win = tk.Toplevel(root)
        roc_win.title("ROC Curve")
        canvas = FigureCanvasTkAgg(fig, master=roc_win)
        canvas.draw()
        canvas.get_tk_widget().pack()
    except Exception as e:
        messagebox.showerror("Error", str(e))

tk.Label(main_frame, text="4. ROC Curve", font=("Arial", 12, "bold")).pack(pady=5)
tk.Button(main_frame, text="Show ROC Curve", command=show_roc).pack(pady=5)

# -----------------------
# Step 5: Save Model
# -----------------------
def save_model():
    if rf_model is None:
        messagebox.showwarning("Warning", "Train the model first!")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".pkl",
                                             filetypes=[("Pickle Files", "*.pkl")])
    if file_path:
        joblib.dump({"model": rf_model, "scaler": scaler}, file_path)
        messagebox.showinfo("Saved", f"✅ Model and Scaler saved at:\n{file_path}")

tk.Label(main_frame, text="5. Save Model", font=("Arial", 12, "bold")).pack(pady=5)
tk.Button(main_frame, text="Save Trained Model", command=save_model).pack(pady=5)

# -----------------------
# Dataset Info
# -----------------------
info_frame = tk.Frame(main_frame)
info_frame.pack(side="bottom", fill="x", pady=10)
tk.Label(info_frame, text=f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns",
         font=("Arial", 10), fg="gray").pack()

root.mainloop()
