import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# -----------------------
# Load Data
# -----------------------
cleanedData = pd.read_csv("cleaned_heart_disease_data.csv")

X = cleanedData.drop("target", axis=1)
y = cleanedData["target"]

categorical_cols = ["chest pain type", "resting ecg", "ST slope"]

preprocessor = ColumnTransformer([
    ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
], remainder="passthrough")

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("scale", StandardScaler(with_mean=False)),
    ("logreg", LogisticRegression(max_iter=1000, random_state=42))
])

X_train = X_test = y_train = y_test = None
model_fitted = False


# -----------------------
# Tkinter GUI
# -----------------------
root = tk.Tk()
root.title("Heart Disease Prediction (Logistic Regression)")
root.geometry("850x650")

frame = tk.Frame(root)
frame.pack(expand=True, fill="both", padx=20, pady=20)

title = tk.Label(frame, text="Heart Disease Logistic Regression Trainer",
                 font=("Arial", 16, "bold"))
title.pack(pady=10)


# -----------------------
# Step 1: Train/Test Split
# -----------------------
def do_split():
    global X_train, X_test, y_train, y_test
    try:
        test_ratio = float(test_ratio_var.get())
        if not (0.05 <= test_ratio <= 0.5):
            raise ValueError("Choose a test size between 0.05 and 0.5")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, stratify=y, random_state=42
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

tk.Label(split_frame, text="Test Ratio (0.05 - 0.5):").pack(side="left", padx=5)
test_ratio_var = tk.StringVar(value="0.2")
tk.Entry(split_frame, textvariable=test_ratio_var, width=10).pack(side="left")
tk.Button(split_frame, text="Run Split", command=do_split).pack(side="left", padx=10)


# -----------------------
# Step 2: Cross Validation (Train Set Only)
# -----------------------
def run_cv():
    if X_train is None:
        messagebox.showwarning("Warning", "⚠️ Please split the data first!")
        return
    try:
        scores = cross_val_score(pipeline, X_train, y_train, cv=10, scoring="accuracy")
        cv_result.set(f"Cross Validation (Train Set):\n"
                      f"Scores: {scores}\n"
                      f"Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")
    except Exception as e:
        messagebox.showerror("Error", str(e))


tk.Label(frame, text="2. Cross Validation (10-Fold on Train Set)", font=("Arial", 12, "bold")).pack(pady=5)
tk.Button(frame, text="Run Cross Validation", command=run_cv).pack(pady=5)
cv_result = tk.StringVar()
tk.Label(frame, textvariable=cv_result, justify="left", fg="blue").pack(pady=5)


# -----------------------
# Step 3: Train Logistic Regression Model
# -----------------------
def train_model():
    global model_fitted
    if X_train is None:
        messagebox.showwarning("Warning", "⚠️ Please split the data first!")
        return
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    summary = f"✅ Model Trained!\nTest Accuracy: {acc:.4f}\n\nReport:\n{report}"
    result_text.set(summary)
    model_fitted = True


tk.Label(frame, text="3. Train Logistic Regression Model", font=("Arial", 12, "bold")).pack(pady=5)
tk.Button(frame, text="Train Model", command=train_model).pack(pady=5)
result_text = tk.StringVar()
tk.Label(frame, textvariable=result_text, justify="left", font=("Courier", 10)).pack(pady=5)


# -----------------------
# Step 4: Show Confusion Matrix
# -----------------------
def show_cm():
    if not model_fitted:
        messagebox.showwarning("Warning", "Train the model first!")
        return
    preds = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, preds)

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"], ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

    cm_win = tk.Toplevel(root)
    cm_win.title("Confusion Matrix")
    canvas = FigureCanvasTkAgg(fig, master=cm_win)
    canvas.draw()
    canvas.get_tk_widget().pack()


tk.Label(frame, text="4. Confusion Matrix", font=("Arial", 12, "bold")).pack(pady=5)
tk.Button(frame, text="Show Confusion Matrix", command=show_cm).pack(pady=5)


# -----------------------
# Step 5: Save Model
# -----------------------
def save_model():
    if not model_fitted:
        messagebox.showwarning("Warning", "Train the model first!")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".pkl",
                                             filetypes=[("Pickle Files", "*.pkl")])
    if file_path:
        joblib.dump(pipeline, file_path)
        messagebox.showinfo("Saved", f"✅ Model saved at:\n{file_path}")


tk.Label(frame, text="5. Save Model", font=("Arial", 12, "bold")).pack(pady=5)
tk.Button(frame, text="Save Trained Model", command=save_model).pack(pady=5)


# -----------------------
# Dataset Info at Bottom
# -----------------------
info_frame = tk.Frame(frame)
info_frame.pack(side="bottom", fill="x", pady=10)
tk.Label(info_frame, text=f"Dataset: {cleanedData.shape[0]} rows × {cleanedData.shape[1]} columns",
         font=("Arial", 10), fg="gray").pack()

root.mainloop()
