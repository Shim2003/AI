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
from PIL import Image, ImageTk
import io
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')

# Global variables
scaler = StandardScaler()
final_knn = None
X_train_scaled = X_test_scaled = y_train = y_test = None
target_column = 'target'

# Tkinter GUI setup
root = tk.Tk()
root.title("Heart Disease Prediction (KNN)")
root.geometry("1000x700")

# Notebook widget
notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill="both")

# Tabs
tab1 = ttk.Frame(notebook)
tab2 = ttk.Frame(notebook)
tab3 = ttk.Frame(notebook)

notebook.add(tab1, text="Dataset Viewer")
notebook.add(tab2, text="Modeling")
notebook.add(tab3, text="Prediction")

### Dataset Viewer Tab ###
def clear_tab1():
    for widget in tab1.winfo_children():
        widget.destroy()

def show_summary():
    clear_tab1()
    summary_text = tk.Text(tab1, wrap="none", height=20)
    summary_text.pack(expand=True, fill="both")

    summary_text.insert(tk.END, "=== Dataset Summary ===\n")
    summary_text.insert(tk.END, f"Shape: {df.shape}\n\n")
    summary_text.insert(tk.END, "Column Data Types:\n")
    summary_text.insert(tk.END, df.dtypes.to_string())
    summary_text.insert(tk.END, "\n\nStatistical Summary:\n")
    summary_text.insert(tk.END, df.describe().to_string())

    btn_frame = tk.Frame(tab1)
    btn_frame.pack(pady=10)

    tk.Button(btn_frame, text="Show Full Dataset", command=show_data).grid(row=0, column=0, padx=10)
    tk.Button(btn_frame, text="Clean Data", command=clean_data).grid(row=0, column=1, padx=10)

def show_data():
    clear_tab1()
    frame = tk.Frame(tab1)
    frame.pack(expand=True, fill="both")

    text_area = tk.Text(frame, wrap="none")
    text_area.pack(side="left", expand=True, fill="both")

    y_scroll = tk.Scrollbar(frame, orient="vertical", command=text_area.yview)
    y_scroll.pack(side="right", fill="y")
    text_area.configure(yscrollcommand=y_scroll.set)

    x_scroll = tk.Scrollbar(tab1, orient="horizontal", command=text_area.xview)
    x_scroll.pack(fill="x")
    text_area.configure(xscrollcommand=x_scroll.set)

    text_area.insert(tk.END, df.to_string())

    tk.Button(tab1, text="Back to Summary", command=show_summary).pack(pady=10)

def clean_data():
    global df
    before = df.shape[0]
    df.drop_duplicates(inplace=True)
    after = df.shape[0]
    messagebox.showinfo("Data Cleaned",
                        f"Removed {before - after} duplicate rows.\nNew shape: {df.shape}")
    show_summary()

show_summary()

### Modeling Tab ###

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

            num_duplicates = df.duplicated().sum()
            num_missing = df.isnull().sum().sum()

            df_cleaned = df.drop_duplicates().dropna()

            X = df_cleaned.drop(target_column, axis=1)
            y = df_cleaned[target_column]

            global X_train_scaled, X_test_scaled, y_train, y_test

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42
            )
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            result_text.set(
                f"✅ Preprocess & Split Success!\n"
                f"Duplicates Removed: {num_duplicates}\n"
                f"Missing Values Removed: {num_missing}\n"
                f"Train Ratio: {train_part}/10\nTest Ratio: {test_part}/10\n"
                f"Train Rows: {X_train.shape[0]}\nTest Rows: {X_test.shape[0]}"
            )
        except Exception as e:
            result_text.set(f"❌ Error: {e}")

    dialog = tk.Toplevel()
    dialog.title("Preprocess & Split")
    dialog.geometry("320x260")
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

tk.Label(tab2, text="1. Preprocess & Split", font=("Arial", 11, "bold")).pack(pady=(10, 0))
tk.Button(tab2, text="Preprocess & Split", command=open_split_dialog).pack(pady=5)


def find_best_k():
    if X_train_scaled is None:
        messagebox.showwarning("Warning", "Please preprocess data first.")
        return
    k_range = range(1, 31)
    k_scores = []
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        k_scores.append(scores.mean())
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, k_scores, marker='o')
    plt.xlabel("k")
    plt.ylabel("CV Accuracy")
    plt.title("Optimal k Value")
    plt.grid(True)
    plt.show()

tk.Label(tab2, text="2. Find Best K", font=("Arial", 11, "bold")).pack(pady=(10, 0))
tk.Button(tab2, text="Find Best K", command=find_best_k).pack(pady=5)

def tune_and_train():
    global final_knn, grid

    if X_train_scaled is None:
        messagebox.showwarning("Warning", "Please preprocess data first.")
        return

    # Step 1: Grid search to find best model
    param_grid = {
        'n_neighbors': range(3, 21),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5,
                        scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    final_knn = grid.best_estimator_

    # Step 2: Predictions and classification report
    preds = final_knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Step 3: Create new Toplevel window
    result_win = tk.Toplevel()
    result_win.title("Initial KNN Model Results")

    # Step 4: Show accuracy and best parameters as label
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

    # Step 5: Confusion Matrix
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

tk.Label(tab2, text="3. Initial KNN Model", font=("Arial", 11, "bold")).pack(pady=(10, 0))
tk.Button(tab2, text="Show Initial KNN Model", command=tune_and_train).pack(pady=5)

def show_correlation():
    corr_with_target = df.drop(target_column, axis=1).corrwith(df[target_column]).abs().sort_values(ascending=False)
    top_corr = corr_with_target.head()
    corr_matrix = df.drop(target_column, axis=1).corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    img_tk = ImageTk.PhotoImage(img)
    buf.close()
    plt.close(fig)

    dialog = tk.Toplevel()
    dialog.title("Feature Correlation")
    dialog.geometry("650x700")
    dialog.resizable(False, False)
    dialog.grab_set()

    tk.Label(dialog, text="Top Correlated Features with Target", font=("Arial", 11, "bold")).pack(pady=(10, 0))
    text = tk.Text(dialog, height=6, width=70, font=("Consolas", 10))
    text.insert(tk.END, top_corr.to_string())
    text.config(state='disabled')
    text.pack(pady=5)

    tk.Label(dialog, text="Feature Correlation Heatmap", font=("Arial", 11, "bold")).pack(pady=10)
    img_label = tk.Label(dialog, image=img_tk)
    img_label.image = img_tk
    img_label.pack()

    tk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)

tk.Label(tab2, text="4. Feature Correlation", font=("Arial", 11, "bold")).pack(pady=(10, 0))
tk.Button(tab2, text="Show Feature Correlation", command=show_correlation).pack(pady=5)

def final_model():
    global final_knn, scaler_r

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

    # Summary string
    summary = (
        f"✅ Final Reduced Model\n"
        f"• Accuracy: {acc:.4f}\n"
        f"• Dropped Features: {', '.join(to_drop) if to_drop else 'None'}\n\n"
        f"✅ Classification Report:\n{report_df.to_string()}"
    )
    tk.Label(result_win, text=summary, justify="left", font=("Courier", 10), anchor="w").pack(padx=10, pady=10)

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test_r, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()

    # Embed matplotlib plot into Tkinter
    canvas = FigureCanvasTkAgg(fig, master=result_win)
    canvas.draw()
    canvas.get_tk_widget().pack(padx=10, pady=10)

tk.Label(tab2, text="5. Final Model", font=("Arial", 11, "bold")).pack(pady=(10, 0))
tk.Button(tab2, text="Run Final Model", command=final_model).pack(pady=5)

### Prediction Tab ###
entry_vars = []

def predict_from_input():
    try:
        inputs = [float(var.get()) for var in entry_vars]
        input_scaled = scaler_r.transform([inputs])  # Use reduced model scaler
        prediction = final_knn.predict(input_scaled)[0]
        prob = final_knn.predict_proba(input_scaled)[0]

        if prediction == 1:
            message = "🩺 You HAVE heart disease."
        else:
            message = "✅ You DO NOT have heart disease."

        result = f"{message}\n\nProbabilities:\n- No Disease: {prob[0]:.3f}\n- Disease: {prob[1]:.3f}"
        messagebox.showinfo("Prediction Result", result)

    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed:\n{e}")

def build_prediction_form():
    for widget in tab3.winfo_children():
        widget.destroy()
    entry_vars.clear()

    # Get the reduced features used in final model
    df_features_only = df.drop(columns=[target_column])
    corr_matrix = df_features_only.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
    reduced_features = df.drop(columns=[target_column] + to_drop).columns.tolist()

    for i, feat in enumerate(reduced_features):
        lbl = tk.Label(tab3, text=feat)
        lbl.grid(row=i, column=0, padx=5, pady=3, sticky='e')

        # Get min, max, unique values for dropdown
        values = df[feat].drop_duplicates().sort_values().tolist()
        var = tk.StringVar()
        dropdown = ttk.Combobox(tab3, textvariable=var, values=values, state="readonly")
        dropdown.grid(row=i, column=1, padx=5, pady=3, sticky='w')
        dropdown.current(0)

        entry_vars.append(var)

    tk.Button(tab3, text="Predict", command=predict_from_input).grid(
        row=len(reduced_features), column=0, columnspan=2, pady=10
    )

build_prediction_form()
root.mainloop()
