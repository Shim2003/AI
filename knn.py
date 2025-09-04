import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
import joblib
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator

warnings.filterwarnings('ignore')

# -------------------------
# Load dataset
# -------------------------
try:
    df = pd.read_csv('cleaned_heart_disease_data.csv')
    print(f"Dataset loaded successfully: {df.shape}")
except FileNotFoundError:
    print("Error: 'cleaned_heart_disease_data.csv' not found. Please ensure the file is in the same directory.")
    exit()

# -------------------------
# Global / schema
# -------------------------
target_column = 'target'
best_k = None

# Based on your EDA:
categorical_cols = [
    'sex',
    'chest pain type',
    'fasting blood sugar',
    'resting ecg',
    'exercise angina',
    'ST slope'
]

numeric_cols = [
    'age',
    'resting bp s',
    'cholesterol',
    'max heart rate',
    'oldpeak'
]

# Which numeric columns were skewed in your plots
skewed_numeric_cols = ['cholesterol', 'oldpeak']
other_numeric_cols = [c for c in numeric_cols if c not in skewed_numeric_cols]

# Globals for train/test raw splits (pipeline will transform)
X_train_raw = X_test_raw = y_train = y_test = None
final_knn = None
grid = None

# -------------------------
# Tkinter GUI setup
# -------------------------
root = tk.Tk()
root.title("Heart Disease Prediction (Enhanced KNN)")
root.geometry("900x700")

main_frame = tk.Frame(root)
main_frame.pack(expand=True, fill="both", padx=20, pady=20)

title_label = tk.Label(
    main_frame,
    text="Heart Disease: Enhanced KNN Model Trainer",
    font=("Arial", 16, "bold")
)
title_label.pack(pady=(0, 20))

# -------------------------
# Helper: Build pipeline
# -------------------------
def build_knn_pipeline():
    """
    Preprocessing:
    - PowerTransform skewed numeric columns (yeo-johnson)
    - passthrough other numeric columns
    - StandardScale all numeric outputs
    - OneHotEncode categorical columns (drop='if_binary' to avoid trivial collinearity)

    Followed by:
    - SelectKBest (mutual_info_classif)
    - KNeighborsClassifier
    """
    # Apply PowerTransformer only to skewed columns and passthrough others
    numeric_power_ct = ColumnTransformer(
        transformers=[
            ('skewed', PowerTransformer(method='yeo-johnson', standardize=False), skewed_numeric_cols),
            ('other', 'passthrough', other_numeric_cols)
        ],
        remainder='drop'
    )

    # Numeric block: apply power transform then scale
    numeric_block = Pipeline(steps=[
        ('num_power', numeric_power_ct),
        ('num_scale', StandardScaler())
    ])

    # Categorical block
    categorical_block = OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False)

    # Full preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_block, numeric_cols),
            ('cat', categorical_block, categorical_cols)
        ],
        remainder='drop'
    )

    # Full pipeline
    pipeline = Pipeline(steps=[
        ('preproc', preprocessor),
        ('select', SelectKBest(score_func=mutual_info_classif, k=5)),  # Default k=5
        ('knn', KNeighborsClassifier())
    ])

    return pipeline

# -------------------------
# Train/Test Split dialog
# -------------------------
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

            global X_train_raw, X_test_raw, y_train, y_test
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42
            )

            result_text.set(
                f"✅ Train/Test Split Success!\n"
                f"Train Ratio: {train_part}/10\nTest Ratio: {test_part}/10\n"
                f"Train Rows: {X_train_raw.shape[0]}\nTest Rows: {X_test_raw.shape[0]}"
            )
            dialog.destroy()
        except Exception as e:
            result_text.set(f"❌ Error: {e}")

    dialog = tk.Toplevel()
    dialog.title("Train/Test Split")
    dialog.geometry("360x220")
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
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="OK", command=perform_split, width=10).grid(row=0, column=0, padx=10)
    tk.Button(btn_frame, text="Cancel", command=dialog.destroy, width=10).grid(row=0, column=1, padx=10)

    result_text = tk.StringVar()
    result_label = tk.Label(dialog, textvariable=result_text, fg='green', justify='left')
    result_label.pack(pady=5)

# UI: Step 1
tk.Label(main_frame, text="1. Train/Test Split", font=("Arial", 12, "bold")).pack(pady=(10, 0))
tk.Button(main_frame, text="Configure Train/Test Split", command=open_split_dialog, width=30).pack(pady=5)

# -------------------------
# Find Best K (uses pipeline & CV ROC-AUC)
# -------------------------
def find_best_k():
    if X_train_raw is None:
        messagebox.showwarning("Warning", "Please configure train/test split first.")
        return

    k_range = range(1, 31)
    auc_scores = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for k in k_range:
        pipe = build_knn_pipeline()
        pipe.set_params(
            select__k=5,
            knn__n_neighbors=k,
            knn__weights='distance',
            knn__metric='minkowski',
            knn__p=2
        )
        try:
            scores = cross_val_score(pipe, X_train_raw, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
            auc_scores.append(scores.mean())
        except Exception as e:
            print(f"Error with k={k}: {e}")
            auc_scores.append(np.nan)

    # Find best k
    best_idx = int(np.nanargmax(auc_scores))
    best_k_value = list(k_range)[best_idx]
    best_auc_value = auc_scores[best_idx]

    global best_k
    best_k = best_k_value

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(list(k_range), auc_scores, marker='o', label="CV ROC-AUC")
    plt.scatter(best_k_value, best_auc_value, color='red', s=100, label=f"Best k={best_k_value}")

    plt.xlabel("k")
    plt.ylabel("CV ROC-AUC")
    plt.title("K vs Cross-Validated ROC-AUC (with preprocessing)")
    plt.legend()
    plt.grid(True)
    plt.show()

# UI: Step 2
tk.Label(main_frame, text="2. Find Best K (CV ROC-AUC)", font=("Arial", 12, "bold")).pack(pady=(20, 0))
tk.Button(main_frame, text="Find Best K", command=find_best_k, width=30).pack(pady=5)

# -------------------------
# Enhanced KNN training & grid search
# -------------------------
def tune_and_train():
    global final_knn, grid
    if X_train_raw is None:
        messagebox.showwarning("Warning", "Please configure train/test split first.")
        return
    try:
        # Show progress
        progress_win = tk.Toplevel()
        progress_win.title("Training in Progress...")
        progress_win.geometry("300x100")
        progress_win.grab_set()
        tk.Label(progress_win, text="Training Enhanced KNN Model...\nThis may take a few minutes.", font=("Arial", 10)).pack(expand=True)
        progress_win.update()

        pipe = build_knn_pipeline()

        # Get total number of features after preprocessing
        pipe_temp = build_knn_pipeline()
        pipe_temp.fit(X_train_raw, y_train)
        n_features_after_preprocessing = pipe_temp.named_steps['preproc'].transform(X_train_raw).shape[1]

        # Set reasonable k values
        max_k = min(n_features_after_preprocessing, 10)
        k_values = list(range(3, max_k + 1)) + [n_features_after_preprocessing]

        param_grid = {
            'select__k': k_values,
            'knn__n_neighbors': [best_k],
            'knn__weights': ['uniform', 'distance'],
            'knn__metric': ['minkowski'],
            'knn__p': [1, 2]
        }

        print("Starting grid search...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
        grid.fit(X_train_raw, y_train)

        final_knn = grid.best_estimator_
        print("Grid search completed!")

        # Wrap with calibration
        calibrated_knn = CalibratedClassifierCV(FrozenEstimator(final_knn), method='sigmoid')
        calibrated_knn.fit(X_train_raw, y_train)
        joblib.dump(calibrated_knn, "final_knn_model.pkl")
        print("Model saved as final_knn_model.pkl")

        # Evaluate on test using calibrated model
        proba = calibrated_knn.predict_proba(X_test_raw)[:, 1]
        preds = (proba >= 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)
        report = classification_report(y_test, preds, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        # Result window
        result_win = tk.Toplevel()
        result_win.title("Enhanced KNN Model Results")
        result_win.geometry("900x800")

        # Create scrollable frame
        canvas_scroll = tk.Canvas(result_win)
        scrollbar = tk.Scrollbar(result_win, orient="vertical", command=canvas_scroll.yview)
        scrollable_frame = tk.Frame(canvas_scroll)
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        canvas_scroll.create_window((0, 0), window=scrollable_frame, anchor="nw")

        # Results summary
        summary = (
            f"✅ Best Parameters:\n{grid.best_params_}\n\n"
            f"✅ Best CV ROC-AUC: {grid.best_score_:.4f}\n"
            f"✅ Test Accuracy: {acc:.4f}\n"
            f"✅ Test ROC-AUC: {auc:.4f}\n\n"
            f"✅ Classification Report:\n{report_df.round(3).to_string()}"
        )
        tk.Label(scrollable_frame, text=summary, justify="left", font=("Courier", 9), anchor="w").pack(padx=10, pady=10)

        # Confusion matrix
        cm = confusion_matrix(y_test, preds)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
        ax_cm.set_title("Confusion Matrix")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_xlabel("Predicted")
        plt.tight_layout()
        canvas_cm = FigureCanvasTkAgg(fig_cm, master=scrollable_frame)
        canvas_cm.draw()
        canvas_cm.get_tk_widget().pack(padx=10, pady=10)

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, proba)
        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
        ax_roc.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        ax_roc.plot([0, 1], [0, 1], '--', linewidth=0.8, color='red', alpha=0.7)
        ax_roc.set_title("ROC Curve")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(loc="lower right")
        ax_roc.grid(True, alpha=0.3)
        plt.tight_layout()
        canvas_roc = FigureCanvasTkAgg(fig_roc, master=scrollable_frame)
        canvas_roc.draw()
        canvas_roc.get_tk_widget().pack(padx=10, pady=10)

        # Configure scrollable region
        scrollable_frame.update_idletasks()
        canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
        canvas_scroll.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    except Exception as e:
        if 'progress_win' in locals():
            progress_win.destroy()
        messagebox.showerror("Error", f"Training failed: {str(e)}\n\nTry reducing the parameter grid size or check your data.")
        print(f"Detailed error: {e}")

# UI: Step 3
tk.Label(main_frame, text="3. Train Enhanced KNN Model", font=("Arial", 12, "bold")).pack(pady=(20, 0))
tk.Button(main_frame, text="Train Enhanced KNN", command=tune_and_train, width=30).pack(pady=5)

# -------------------------
# Dataset info at bottom
# -------------------------
info_frame = tk.Frame(main_frame)
info_frame.pack(side="bottom", fill="x", pady=10)
tk.Label(info_frame, text=f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns", font=("Arial", 10), fg="gray").pack()

if __name__ == "__main__":
    root.mainloop()