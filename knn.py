import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('cleaned_heart_disease_data.csv')

# Global variables
scaler = StandardScaler()
final_knn = None
X_train_scaled = X_test_scaled = y_train = y_test = None
target_column = 'target'

# Tkinter GUI setup
root = tk.Tk()
root.title("Heart Disease Prediction (KNN)")
root.geometry("800x600")

# Main frame for modeling
main_frame = tk.Frame(root)
main_frame.pack(expand=True, fill="both", padx=20, pady=20)

# Title
title_label = tk.Label(main_frame, text="Heart Disease KNN Model Training", font=("Arial", 16, "bold"))
title_label.pack(pady=(0, 20))

### Train/Test Split Function ###
def open_split_dialog():
    def update_test_ratio(*args):
        try:
            train_value = int(train_ratio_var.get())
            if 1 <= train_value <= 9:
                test_ratio_var.set(str(10 - train_value))
            else:
                test_ratio_var.set("?")
        except ValueError:
            test_ratio_var.set("?")

    def perform_split():
        try:
            train_part = int(train_ratio_var.get())
            test_part = 10 - train_part
            test_size = test_part / 10.0

            X = df.drop(target_column, axis=1)
            y = df[target_column]

            global X_train_scaled, X_test_scaled, y_train, y_test

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42
            )
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            result_text.set(
                f"✅ Train/Test Split Success!\n"
                f"Train Ratio: {train_part}/10\nTest Ratio: {test_part}/10\n"
                f"Train Rows: {X_train.shape[0]}\nTest Rows: {X_test.shape[0]}"
            )
            dialog.destroy()
        except Exception as e:
            result_text.set(f"❌ Error: {e}")

    dialog = tk.Toplevel()
    dialog.title("Train/Test Split")
    dialog.geometry("320x200")
    dialog.resizable(False, False)
    dialog.grab_set()

    input_frame = tk.Frame(dialog)
    input_frame.pack(pady=15)

    tk.Label(input_frame, text="Train Ratio (1-9):").grid(row=0, column=0, padx=5)
    train_ratio_var = tk.StringVar(value="8")
    train_entry = tk.Entry(input_frame, textvariable=train_ratio_var, width=5, justify='center')
    train_entry.grid(row=0, column=1, padx=5)

    tk.Label(input_frame, text="Test Ratio:").grid(row=0, column=2, padx=5)
    test_ratio_var = tk.StringVar()
    tk.Label(input_frame, textvariable=test_ratio_var, width=5, bg="lightgray").grid(row=0, column=3, padx=5)

    train_ratio_var.trace_add('write', update_test_ratio)
    update_test_ratio()

    btn_frame = tk.Frame(dialog)
    btn_frame.pack(pady=5)
    tk.Button(btn_frame, text="OK", command=perform_split, width=8).grid(row=0, column=0, padx=10)
    tk.Button(btn_frame, text="Cancel", command=dialog.destroy, width=8).grid(row=0, column=1, padx=10)

    result_text = tk.StringVar()
    result_label = tk.Label(dialog, textvariable=result_text, fg='green', justify='left')
    result_label.pack(pady=10)

# Step 1: Train/Test Split
tk.Label(main_frame, text="1. Train/Test Split", font=("Arial", 12, "bold")).pack(pady=(10, 0))
tk.Button(main_frame, text="Configure Train/Test Split", command=open_split_dialog, width=25).pack(pady=5)

### Find Best K Function ###
def find_best_k():
    if X_train_scaled is None:
        messagebox.showwarning("Warning", "Please configure train/test split first.")
        return
    k_range = range(1, 31)
    k_scores = []
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        k_scores.append(scores.mean())
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, k_scores, marker='o')
    plt.xlabel("k")
    plt.ylabel("CV Accuracy")
    plt.title("Optimal k Value")
    plt.grid(True)
    plt.show()

# Step 2: Find Best K
tk.Label(main_frame, text="2. Find Best K", font=("Arial", 12, "bold")).pack(pady=(20, 0))
tk.Button(main_frame, text="Find Best K", command=find_best_k, width=25).pack(pady=5)

### Initial KNN Model ###
def tune_and_train():
    global final_knn, grid

    if X_train_scaled is None:
        messagebox.showwarning("Warning", "Please configure train/test split first.")
        return

    # Grid search to find best model
    param_grid = {
        'n_neighbors': range(3, 21),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5,
                        scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    final_knn = grid.best_estimator_

    # Predictions and evaluation
    preds = final_knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Create result window
    result_win = tk.Toplevel()
    result_win.title("Initial KNN Model Results")
    result_win.geometry("600x500")

    # Show results
    summary = (
        f"✅ Best Parameters:\n"
        f"• n_neighbors: {grid.best_params_['n_neighbors']}\n"
        f"• weights: {grid.best_params_['weights']}\n"
        f"• metric: {grid.best_params_['metric']}\n\n"
        f"✅ Accuracy: {acc:.4f}\n\n"
        f"✅ Classification Report:\n"
        f"{report_df.round(2).to_string()}"
    )
    tk.Label(result_win, text=summary, justify="left", font=("Courier", 10), anchor="w").pack(padx=10, pady=10)

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=result_win)
    canvas.draw()
    canvas.get_tk_widget().pack(padx=10, pady=10)

# Step 3: Initial KNN Model
tk.Label(main_frame, text="3. Initial KNN Model", font=("Arial", 12, "bold")).pack(pady=(20, 0))
tk.Button(main_frame, text="Train Initial KNN Model", command=tune_and_train, width=25).pack(pady=5)

### Feature Correlation ###
def show_correlation():
    if target_column not in df.columns:
        messagebox.showerror("Error", f"Target column '{target_column}' not found in dataset")
        return
        
    corr_with_target = df.drop(target_column, axis=1).corrwith(df[target_column]).abs().sort_values(ascending=False)
    top_corr = corr_with_target.head()
    corr_matrix = df.drop(target_column, axis=1).corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()

    # Create result window
    dialog = tk.Toplevel()
    dialog.title("Feature Correlation")
    dialog.geometry("650x700")
    dialog.resizable(False, False)
    dialog.grab_set()

    tk.Label(dialog, text="Top Correlated Features with Target", font=("Arial", 12, "bold")).pack(pady=(10, 0))
    text = tk.Text(dialog, height=6, width=70, font=("Consolas", 10))
    text.insert(tk.END, top_corr.to_string())
    text.config(state='disabled')
    text.pack(pady=5)

    tk.Label(dialog, text="Feature Correlation Heatmap", font=("Arial", 12, "bold")).pack(pady=10)
    
    canvas = FigureCanvasTkAgg(fig, master=dialog)
    canvas.draw()
    canvas.get_tk_widget().pack()

    tk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)

# Step 4: Feature Correlation
tk.Label(main_frame, text="4. Feature Correlation", font=("Arial", 12, "bold")).pack(pady=(20, 0))
tk.Button(main_frame, text="Show Feature Correlation", command=show_correlation, width=25).pack(pady=5)

### Final Model with Feature Reduction ###
def final_model():
    global final_knn, scaler_r

    if X_train_scaled is None:
        messagebox.showwarning("Warning", "Please configure train/test split first.")
        return

    # Remove highly correlated features
    df_features_only = df.drop(columns=[target_column])
    corr_matrix = df_features_only.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
    df_reduced = df.drop(columns=to_drop)

    # Prepare data
    X = df_reduced.drop(columns=[target_column])
    y = df_reduced[target_column]
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler_r = StandardScaler()
    X_train_r_scaled = scaler_r.fit_transform(X_train_r)
    X_test_r_scaled = scaler_r.transform(X_test_r)

    # Train model
    param_grid = {
        'n_neighbors': range(3, 21),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_r_scaled, y_train_r)
    final_knn = grid.best_estimator_

    # Evaluate
    preds = final_knn.predict(X_test_r_scaled)
    acc = accuracy_score(y_test_r, preds)
    report_dict = classification_report(y_test_r, preds, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(2)

    # Create result window
    result_win = tk.Toplevel()
    result_win.title("Final KNN Model Results")
    result_win.geometry("600x500")

    # Summary
    summary = (
        f"✅ Final Reduced Model\n"
        f"• Accuracy: {acc:.4f}\n"
        f"• Dropped Features: {', '.join(to_drop) if to_drop else 'None'}\n\n"
        f"✅ Classification Report:\n{report_df.to_string()}"
    )
    tk.Label(result_win, text=summary, justify="left", font=("Courier", 10), anchor="w").pack(padx=10, pady=10)

    # Confusion Matrix
    cm = confusion_matrix(y_test_r, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=result_win)
    canvas.draw()
    canvas.get_tk_widget().pack(padx=10, pady=10)

# Step 5: Final Model
tk.Label(main_frame, text="5. Final Model with Feature Reduction", font=("Arial", 12, "bold")).pack(pady=(20, 0))
tk.Button(main_frame, text="Train Final Model", command=final_model, width=25).pack(pady=5)

# Dataset info at bottom
info_frame = tk.Frame(main_frame)
info_frame.pack(side="bottom", fill="x", pady=10)
tk.Label(info_frame, text=f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns", 
         font=("Arial", 10), fg="gray").pack()

root.mainloop()