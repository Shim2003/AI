import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, Binarizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageTk
import io
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')

# Global variables
scaler = StandardScaler()
label_encoders = {}
final_nb = None
X_train_processed = X_test_processed = y_train = y_test = None
target_column = 'target'
processed_features = None

# Tkinter GUI setup
root = tk.Tk()
root.title("Heart Disease Prediction (Naive Bayes)")
root.geometry("1200x800")

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
    
    # Create main frame with scrollbar
    main_frame = tk.Frame(tab1)
    main_frame.pack(expand=True, fill="both")
    
    summary_text = tk.Text(main_frame, wrap="none", height=20)
    scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=summary_text.yview)
    summary_text.configure(yscrollcommand=scrollbar.set)
    
    summary_text.pack(side="left", expand=True, fill="both")
    scrollbar.pack(side="right", fill="y")

    summary_text.insert(tk.END, "=== Dataset Summary ===\n")
    summary_text.insert(tk.END, f"Shape: {df.shape}\n\n")
    summary_text.insert(tk.END, "Column Data Types:\n")
    summary_text.insert(tk.END, df.dtypes.to_string())
    summary_text.insert(tk.END, "\n\nMissing Values:\n")
    summary_text.insert(tk.END, df.isnull().sum().to_string())
    summary_text.insert(tk.END, "\n\nDuplicates: " + str(df.duplicated().sum()))
    summary_text.insert(tk.END, "\n\nTarget Distribution:\n")
    summary_text.insert(tk.END, df[target_column].value_counts().to_string())
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
    df.dropna(inplace=True)  # Remove missing values for Naive Bayes
    after = df.shape[0]
    messagebox.showinfo("Data Cleaned",
                        f"Removed {before - after} rows (duplicates & missing values).\nNew shape: {df.shape}")
    show_summary()

show_summary()

### Modeling Tab ###

def preprocess_for_nb():
    """Preprocessing specifically for Naive Bayes"""
    global df, processed_features, label_encoders
    
    # Clean data first
    df_clean = df.drop_duplicates().dropna()
    
    # Identify categorical and numerical columns
    categorical_cols = []
    numerical_cols = []
    
    for col in df_clean.columns:
        if col != target_column:
            unique_vals = df_clean[col].nunique()
            if unique_vals <= 10 or df_clean[col].dtype == 'object':
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
    
    processed_features = {
        'categorical': categorical_cols,
        'numerical': numerical_cols
    }
    
    return df_clean, categorical_cols, numerical_cols

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

            # Preprocess data
            df_clean, categorical_cols, numerical_cols = preprocess_for_nb()
            
            X = df_clean.drop(target_column, axis=1)
            y = df_clean[target_column]

            global X_train_processed, X_test_processed, y_train, y_test, scaler, label_encoders

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42
            )

            # Process numerical features (standardize for Gaussian NB)
            X_train_processed = X_train.copy()
            X_test_processed = X_test.copy()
            
            if numerical_cols:
                X_train_processed[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
                X_test_processed[numerical_cols] = scaler.transform(X_test[numerical_cols])

            # Process categorical features (label encoding)
            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                X_train_processed[col] = le.fit_transform(X_train[col].astype(str))
                X_test_processed[col] = le.transform(X_test[col].astype(str))
                label_encoders[col] = le

            result_text.set(
                f"✅ Preprocess & Split Success!\n"
                f"Cleaned Rows: {df_clean.shape[0]}\n"
                f"Categorical Features: {len(categorical_cols)}\n"
                f"Numerical Features: {len(numerical_cols)}\n"
                f"Train Ratio: {train_part}/10\nTest Ratio: {test_part}/10\n"
                f"Train Rows: {X_train.shape[0]}\nTest Rows: {X_test.shape[0]}"
            )
            
            dialog.destroy()
            
        except Exception as e:
            result_text.set(f"❌ Error: {e}")

    dialog = tk.Toplevel()
    dialog.title("Preprocess & Split for Naive Bayes")
    dialog.geometry("400x300")
    dialog.resizable(False, False)
    dialog.grab_set()

    tk.Label(dialog, text="Naive Bayes Preprocessing", font=("Arial", 12, "bold")).pack(pady=10)
    
    info_text = """
    Preprocessing for Naive Bayes:
    • Remove duplicates and missing values
    • Standardize numerical features
    • Encode categorical features
    • Split data maintaining class balance
    """
    tk.Label(dialog, text=info_text, justify="left").pack(pady=5)

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

def compare_nb_variants():
    """Compare different Naive Bayes variants"""
    if X_train_processed is None:
        messagebox.showwarning("Warning", "Please preprocess data first.")
        return
    
    results = {}
    
    # 1. Gaussian NB - works with standardized data
    try:
        gaussian_nb = GaussianNB()
        gaussian_nb.fit(X_train_processed, y_train)
        pred_gaussian = gaussian_nb.predict(X_test_processed)
        accuracy_gaussian = accuracy_score(y_test, pred_gaussian)
        cv_scores_gaussian = cross_val_score(gaussian_nb, X_train_processed, y_train, cv=5)
        
        results['Gaussian NB'] = {
            'accuracy': accuracy_gaussian,
            'cv_mean': cv_scores_gaussian.mean(),
            'cv_std': cv_scores_gaussian.std(),
            'model': gaussian_nb
        }
    except Exception as e:
        results['Gaussian NB'] = {'error': str(e)}
    
    # 2. Multinomial NB - needs non-negative data
    try:
        from sklearn.preprocessing import MinMaxScaler
        minmax_scaler = MinMaxScaler()
        X_train_minmax = minmax_scaler.fit_transform(X_train_processed)
        X_test_minmax = minmax_scaler.transform(X_test_processed)
        
        multinomial_nb = MultinomialNB(alpha=0.1)
        multinomial_nb.fit(X_train_minmax, y_train)
        pred_multinomial = multinomial_nb.predict(X_test_minmax)
        accuracy_multinomial = accuracy_score(y_test, pred_multinomial)
        cv_scores_multinomial = cross_val_score(multinomial_nb, X_train_minmax, y_train, cv=5)
        
        results['Multinomial NB'] = {
            'accuracy': accuracy_multinomial,
            'cv_mean': cv_scores_multinomial.mean(),
            'cv_std': cv_scores_multinomial.std(),
            'model': multinomial_nb,
            'scaler': minmax_scaler
        }
    except Exception as e:
        results['Multinomial NB'] = {'error': str(e)}
    
    # 3. Bernoulli NB - needs binary data
    try:
        from sklearn.preprocessing import Binarizer
        binarizer = Binarizer(threshold=0.0)  # Use 0 as threshold for standardized data
        X_train_binary = binarizer.fit_transform(X_train_processed)
        X_test_binary = binarizer.transform(X_test_processed)
        
        bernoulli_nb = BernoulliNB(alpha=0.1)
        bernoulli_nb.fit(X_train_binary, y_train)
        pred_bernoulli = bernoulli_nb.predict(X_test_binary)
        accuracy_bernoulli = accuracy_score(y_test, pred_bernoulli)
        cv_scores_bernoulli = cross_val_score(bernoulli_nb, X_train_binary, y_train, cv=5)
        
        results['Bernoulli NB'] = {
            'accuracy': accuracy_bernoulli,
            'cv_mean': cv_scores_bernoulli.mean(),
            'cv_std': cv_scores_bernoulli.std(),
            'model': bernoulli_nb,
            'binarizer': binarizer
        }
    except Exception as e:
        results['Bernoulli NB'] = {'error': str(e)}
    
    # Create results window
    result_win = tk.Toplevel()
    result_win.title("Naive Bayes Variants Comparison")
    result_win.geometry("700x500")
    
    tk.Label(result_win, text="Naive Bayes Variants Comparison", 
             font=("Arial", 14, "bold")).pack(pady=10)
    
    # Add explanation
    explanation = """
    • Gaussian NB: Uses standardized continuous features (best for mixed data)
    • Multinomial NB: Uses MinMax scaled features (good for count-like data)  
    • Bernoulli NB: Uses binarized features (good for binary classification)
    """
    tk.Label(result_win, text=explanation, justify="left", font=("Arial", 10)).pack(pady=5)
    
    # Results text
    results_text = tk.Text(result_win, height=12, width=80, font=("Courier", 10))
    scrollbar_results = tk.Scrollbar(result_win, orient="vertical", command=results_text.yview)
    results_text.configure(yscrollcommand=scrollbar_results.set)
    
    results_text.pack(side="left", padx=10, pady=10, expand=True, fill="both")
    scrollbar_results.pack(side="right", fill="y")
    
    results_text.insert(tk.END, f"{'Model':<15} {'Test Acc':<10} {'CV Mean':<10} {'CV Std':<10} {'Status':<15}\n")
    results_text.insert(tk.END, "-" * 75 + "\n")
    
    best_model = None
    best_accuracy = 0
    
    for name, result in results.items():
        if 'error' in result:
            results_text.insert(tk.END, f"{name:<15} {'ERROR':<10} {'N/A':<10} {'N/A':<10} {str(result['error'])[:15]}\n")
        else:
            status = "✅ Success"
            results_text.insert(tk.END, f"{name:<15} {result['accuracy']:<10.4f} {result['cv_mean']:<10.4f} {result['cv_std']:<10.4f} {status}\n")
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_model = name
    
    results_text.insert(tk.END, f"\n🏆 Best Model: {best_model} (Accuracy: {best_accuracy:.4f})\n")
    results_text.insert(tk.END, f"\nRecommendation: Use {best_model} for final model training.")
    results_text.config(state='disabled')

tk.Label(tab2, text="2. Compare NB Variants", font=("Arial", 11, "bold")).pack(pady=(10, 0))
tk.Button(tab2, text="Compare NB Variants", command=compare_nb_variants).pack(pady=5)

def tune_and_train():
    """Hyperparameter tuning for Gaussian Naive Bayes"""
    global final_nb

    if X_train_processed is None:
        messagebox.showwarning("Warning", "Please preprocess data first.")
        return

    # Parameter grid for Gaussian NB (limited parameters)
    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    }
    
    # Grid search
    grid = GridSearchCV(GaussianNB(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_processed, y_train)
    final_nb = grid.best_estimator_

    # Predictions and evaluation
    preds = final_nb.predict(X_test_processed)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Feature probabilities (if possible)
    feature_log_prob = None
    if hasattr(final_nb, 'theta_'):
        feature_log_prob = final_nb.theta_

    # Create result window
    result_win = tk.Toplevel()
    result_win.title("Optimized Gaussian Naive Bayes Results")
    result_win.geometry("800x600")

    # Summary
    summary = (
        f"✅ Best Parameters:\n"
        f"• var_smoothing: {grid.best_params_['var_smoothing']:.2e}\n\n"
        f"✅ Test Accuracy: {acc:.4f}\n"
        f"✅ Cross-validation Score: {grid.best_score_:.4f}\n\n"
        f"✅ Classification Report:\n"
        f"{report_df.round(4).to_string()}"
    )
    
    summary_label = tk.Label(result_win, text=summary, justify="left", 
                           font=("Courier", 10), anchor="w")
    summary_label.pack(padx=10, pady=10)

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion matrix heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    ax1.set_title("Confusion Matrix")
    ax1.set_ylabel("Actual")
    ax1.set_xlabel("Predicted")
    
    # Feature importance (mean values for each class)
    if hasattr(final_nb, 'theta_') and processed_features:
        feature_names = list(X_train_processed.columns)[:10]  # Top 10 features
        class_0_means = final_nb.theta_[0][:len(feature_names)]
        class_1_means = final_nb.theta_[1][:len(feature_names)]
        
        x_pos = np.arange(len(feature_names))
        ax2.bar(x_pos - 0.2, class_0_means, 0.4, label='No Disease', alpha=0.7)
        ax2.bar(x_pos + 0.2, class_1_means, 0.4, label='Disease', alpha=0.7)
        ax2.set_xlabel('Features')
        ax2.set_ylabel('Mean Values')
        ax2.set_title('Feature Means by Class')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(feature_names, rotation=45)
        ax2.legend()
    
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=result_win)
    canvas.draw()
    canvas.get_tk_widget().pack(padx=10, pady=10)

tk.Label(tab2, text="3. Tune Gaussian NB", font=("Arial", 11, "bold")).pack(pady=(10, 0))
tk.Button(tab2, text="Tune & Train Gaussian NB", command=tune_and_train).pack(pady=5)

def show_feature_analysis():
    """Analyze feature distributions and correlations"""
    if X_train_processed is None:
        messagebox.showwarning("Warning", "Please preprocess data first.")
        return
    
    # Combine processed data with target for analysis
    analysis_data = X_train_processed.copy()
    analysis_data['target'] = y_train
    
    # Create analysis window
    analysis_win = tk.Toplevel()
    analysis_win.title("Feature Analysis for Naive Bayes")
    analysis_win.geometry("1000x700")
    
    # Feature correlation with target
    correlations = analysis_data.drop('target', axis=1).corrwith(analysis_data['target']).abs().sort_values(ascending=False)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Top correlations
    top_10_corr = correlations.head(10)
    ax1.barh(range(len(top_10_corr)), top_10_corr.values)
    ax1.set_yticks(range(len(top_10_corr)))
    ax1.set_yticklabels(top_10_corr.index)
    ax1.set_xlabel('Absolute Correlation with Target')
    ax1.set_title('Top 10 Features - Correlation with Target')
    
    # Feature distribution for top feature
    top_feature = top_10_corr.index[0]
    for target_val in [0, 1]:
        data = analysis_data[analysis_data['target'] == target_val][top_feature]
        ax2.hist(data, alpha=0.7, label=f'Target = {target_val}', bins=20)
    ax2.set_xlabel(top_feature)
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Distribution of {top_feature} by Target')
    ax2.legend()
    
    # Correlation matrix heatmap (top features)
    top_features = top_10_corr.index[:8].tolist() + ['target']
    corr_matrix = analysis_data[top_features].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax3)
    ax3.set_title('Correlation Matrix - Top Features')
    
    # Class distribution
    class_counts = analysis_data['target'].value_counts()
    ax4.pie(class_counts.values, labels=['No Disease', 'Disease'], autopct='%1.1f%%')
    ax4.set_title('Target Class Distribution')
    
    plt.tight_layout()
    
    canvas = FigureCanvasTkAgg(fig, master=analysis_win)
    canvas.draw()
    canvas.get_tk_widget().pack(expand=True, fill="both")

tk.Label(tab2, text="4. Feature Analysis", font=("Arial", 11, "bold")).pack(pady=(10, 0))
tk.Button(tab2, text="Show Feature Analysis", command=show_feature_analysis).pack(pady=5)

### Prediction Tab ###
entry_vars = []

def predict_from_input():
    try:
        if final_nb is None:
            messagebox.showwarning("Warning", "Please train the model first.")
            return
            
        # Get input values
        inputs = {}
        feature_names = list(X_train_processed.columns)
        
        for i, var in enumerate(entry_vars):
            inputs[feature_names[i]] = float(var.get())
        
        # Create input dataframe
        input_df = pd.DataFrame([inputs])
        
        # Apply same preprocessing
        if processed_features:
            # Scale numerical features
            numerical_cols = processed_features['numerical']
            if numerical_cols:
                input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
            
            # Encode categorical features
            categorical_cols = processed_features['categorical']
            for col in categorical_cols:
                if col in label_encoders:
                    # Handle unseen categories
                    try:
                        input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
                    except ValueError:
                        # Use most frequent class for unseen categories
                        input_df[col] = 0
        
        # Make prediction
        prediction = final_nb.predict(input_df)[0]
        probabilities = final_nb.predict_proba(input_df)[0]

        if prediction == 1:
            message = "🩺 HIGH RISK: Heart disease predicted"
            color = "red"
        else:
            message = "✅ LOW RISK: No heart disease predicted"
            color = "green"

        result = f"{message}\n\nPrediction Probabilities:\n• No Disease: {probabilities[0]:.3f} ({probabilities[0]*100:.1f}%)\n• Disease: {probabilities[1]:.3f} ({probabilities[1]*100:.1f}%)"
        
        # Create custom message box with colors
        result_win = tk.Toplevel()
        result_win.title("Heart Disease Prediction Result")
        result_win.geometry("400x200")
        result_win.configure(bg="white")
        
        tk.Label(result_win, text=message, font=("Arial", 14, "bold"), 
                fg=color, bg="white").pack(pady=20)
        
        tk.Label(result_win, text=f"Confidence Scores:\n• No Disease: {probabilities[0]*100:.1f}%\n• Disease: {probabilities[1]*100:.1f}%", 
                font=("Arial", 12), bg="white").pack(pady=10)
        
        tk.Button(result_win, text="OK", command=result_win.destroy).pack(pady=10)

    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed:\n{e}")

def build_prediction_form():
    for widget in tab3.winfo_children():
        widget.destroy()
    entry_vars.clear()

    if X_train_processed is None:
        tk.Label(tab3, text="Please preprocess data first in the Modeling tab.", 
                font=("Arial", 12), fg="red").pack(pady=50)
        return

    tk.Label(tab3, text="Heart Disease Risk Assessment", 
            font=("Arial", 16, "bold")).pack(pady=10)
    
    tk.Label(tab3, text="Enter patient information:", 
            font=("Arial", 12)).pack(pady=5)

    # Create scrollable frame for inputs
    canvas = tk.Canvas(tab3)
    scrollbar = tk.Scrollbar(tab3, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Get feature names and create input fields
    feature_names = list(X_train_processed.columns)
    
    for i, feat in enumerate(feature_names):
        frame = tk.Frame(scrollable_frame)
        frame.pack(fill="x", padx=20, pady=3)
        
        lbl = tk.Label(frame, text=f"{feat}:", width=20, anchor="w")
        lbl.pack(side="left")

        # Get range of values for the feature
        min_val = df[feat].min()
        max_val = df[feat].max()
        mean_val = df[feat].mean()
        
        var = tk.StringVar(value=str(round(mean_val, 2)))
        entry = tk.Entry(frame, textvariable=var, width=15)
        entry.pack(side="left", padx=5)
        
        # Show range info
        info_text = f"Range: {min_val:.1f} - {max_val:.1f}"
        tk.Label(frame, text=info_text, font=("Arial", 8), fg="gray").pack(side="left", padx=5)

        entry_vars.append(var)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Prediction button
    tk.Button(tab3, text="🔍 Predict Heart Disease Risk", 
             command=predict_from_input, font=("Arial", 12, "bold"),
             bg="lightblue").pack(pady=20)

build_prediction_form()

# Start the application
root.mainloop()