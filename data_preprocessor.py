import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def clean_heart_disease_data(file_path):
    """
    Comprehensive data cleaning function for heart disease dataset
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    tuple: (pandas.DataFrame, dict) - Cleaned dataset and replacement statistics
    """
    
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Track replaced values
    replaced_values = {}
    
    # Display basic info about the dataset
    print("\n=== ORIGINAL DATASET INFO ===")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    
    # Check for missing values
    print("\n=== MISSING VALUES CHECK ===")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("Missing values found:")
        print(missing_values[missing_values > 0])
    else:
        print("No missing values found.")
    
    # Remove exact duplicates
    print("\n=== REMOVING DUPLICATES ===")
    initial_count = len(df)
    df_clean = df.drop_duplicates()
    duplicates_removed = initial_count - len(df_clean)
    print(f"Removed {duplicates_removed} duplicate rows")
    
    # Handle missing values by dropping rows with any missing values
    if df_clean.isnull().sum().sum() > 0:
        print("\n=== HANDLING MISSING VALUES ===")
        before_missing = len(df_clean)
        df_clean = df_clean.dropna()
        missing_removed = before_missing - len(df_clean)
        print(f"Removed {missing_removed} rows with missing values")
    
    # Clean invalid data by replacing with averages
    print("\n=== CLEANING INVALID DATA (REPLACING WITH AVERAGES) ===")
    
    # Assume common column names (adjust if different)
    # Map possible column names to standard names
    column_mapping = {}
    cols = df_clean.columns.str.lower()
    
    # Try to identify columns with more specific matching
    for col in df_clean.columns:
        col_lower = col.lower()
        if 'chol' in col_lower or 'cholesterol' in col_lower:
            column_mapping['cholesterol'] = col
        elif ('trestbps' in col_lower or 'restingbp' in col_lower or 
              ('resting' in col_lower and ('bp' in col_lower or 'blood' in col_lower or 'pressure' in col_lower)) or
              col_lower == 'restingbps' or col_lower == 'resting_bp'):
            column_mapping['resting_bp'] = col
        elif 'oldpeak' in col_lower:
            column_mapping['oldpeak'] = col
        elif 'slope' in col_lower or 'st_slope' in col_lower or col_lower == 'st_slope':
            column_mapping['st_slope'] = col
    
    # Print all columns to help with debugging
    print(f"All columns in dataset: {list(df_clean.columns)}")
    print(f"Column data types:")
    for col in df_clean.columns:
        col_min = df_clean[col].min() if pd.api.types.is_numeric_dtype(df_clean[col]) else "N/A"
        col_max = df_clean[col].max() if pd.api.types.is_numeric_dtype(df_clean[col]) else "N/A"
        unique_vals = len(df_clean[col].unique()) if len(df_clean[col].unique()) <= 10 else f"{len(df_clean[col].unique())} unique values"
        print(f"  {col}: {df_clean[col].dtype}, range: {col_min}-{col_max}, unique: {unique_vals}")
    
    print(f"Identified columns for cleaning: {column_mapping}")
    
    # If no resting BP column found, let's try to identify it manually
    if 'resting_bp' not in column_mapping:
        print("\nResting BP column not automatically identified. Checking for typical BP patterns...")
        for col in df_clean.columns:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                col_min, col_max = df_clean[col].min(), df_clean[col].max()
                # Blood pressure is typically between 70-200 mmHg
                if col_min >= 0 and col_max > 50 and col_max <= 250:
                    unique_count = len(df_clean[col].unique())
                    if unique_count > 10:  # BP should have many unique values
                        print(f"  Potential BP column: {col} (range: {col_min}-{col_max}, {unique_count} unique values)")
                        # Ask user or use heuristics to confirm
                        if col_max > 50 and unique_count > 20:  # Likely BP column
                            column_mapping['resting_bp'] = col
                            print(f"  -> Selected {col} as resting BP column")
                            break
    
    # Clean Cholesterol (replace invalid values with average)
    if 'cholesterol' in column_mapping:
        chol_col = column_mapping['cholesterol']
        print(f"\nCleaning {chol_col} (Cholesterol):")
        print(f"Original range: {df_clean[chol_col].min()} - {df_clean[chol_col].max()}")
        
        # Identify invalid cholesterol (0 or negative, or extremely high >600)
        invalid_chol = (df_clean[chol_col] <= 0) | (df_clean[chol_col] > 600)
        invalid_chol_count = invalid_chol.sum()
        
        if invalid_chol_count > 0:
            # Calculate average from valid values only
            valid_chol_values = df_clean[chol_col][(df_clean[chol_col] > 0) & (df_clean[chol_col] <= 600)]
            chol_mean = valid_chol_values.mean()
            
            print(f"Replacing {invalid_chol_count} invalid cholesterol values with average: {chol_mean:.2f}")
            df_clean.loc[invalid_chol, chol_col] = chol_mean
            replaced_values[chol_col] = invalid_chol_count
        else:
            print("No invalid cholesterol values found")
    
    # Clean Resting Blood Pressure (replace invalid values with average)
    if 'resting_bp' in column_mapping:
        bp_col = column_mapping['resting_bp']
        print(f"\nCleaning {bp_col} (Resting Blood Pressure):")
        print(f"Original range: {df_clean[bp_col].min()} - {df_clean[bp_col].max()}")
        
        # Identify invalid BP (0 or negative, or extremely high >250)
        invalid_bp = (df_clean[bp_col] <= 0) | (df_clean[bp_col] > 250)
        invalid_bp_count = invalid_bp.sum()
        
        if invalid_bp_count > 0:
            # Calculate average from valid values only
            valid_bp_values = df_clean[bp_col][(df_clean[bp_col] > 0) & (df_clean[bp_col] <= 250)]
            bp_mean = valid_bp_values.mean()
            
            print(f"Replacing {invalid_bp_count} invalid resting BP values with average: {bp_mean:.2f}")
            df_clean.loc[invalid_bp, bp_col] = bp_mean
            replaced_values[bp_col] = invalid_bp_count
        else:
            print("No invalid resting BP values found")
    
    # Clean Oldpeak (replace negative values with average)
    if 'oldpeak' in column_mapping:
        oldpeak_col = column_mapping['oldpeak']
        print(f"\nCleaning {oldpeak_col} (Oldpeak):")
        print(f"Original range: {df_clean[oldpeak_col].min()} - {df_clean[oldpeak_col].max()}")
        
        # Identify negative oldpeak values
        invalid_oldpeak = df_clean[oldpeak_col] < 0
        invalid_oldpeak_count = invalid_oldpeak.sum()
        
        if invalid_oldpeak_count > 0:
            # Calculate average from valid (non-negative) values only
            valid_oldpeak_values = df_clean[oldpeak_col][df_clean[oldpeak_col] >= 0]
            oldpeak_mean = valid_oldpeak_values.mean()
            
            print(f"Replacing {invalid_oldpeak_count} negative oldpeak values with average: {oldpeak_mean:.2f}")
            df_clean.loc[invalid_oldpeak, oldpeak_col] = oldpeak_mean
            replaced_values[oldpeak_col] = invalid_oldpeak_count
        else:
            print("No negative oldpeak values found")
    
    # Clean ST Slope (should be 0, 1, or 2, but you mentioned 0 is invalid) - STILL REMOVE ROWS
    if 'st_slope' in column_mapping:
        slope_col = column_mapping['st_slope']
        print(f"\nCleaning {slope_col} (ST Slope):")
        print(f"Unique values: {sorted(df_clean[slope_col].unique())}")
        
        # Remove rows with ST slope = 0 (as mentioned it's invalid)
        invalid_slope = df_clean[slope_col] == 0
        invalid_slope_count = invalid_slope.sum()
        
        if invalid_slope_count > 0:
            print(f"Removing {invalid_slope_count} rows with invalid ST slope (0)")
            df_clean = df_clean[~invalid_slope]
        else:
            print("No invalid ST slope values found")
    
    # Additional outlier detection using IQR method for continuous variables
    print("\n=== OUTLIER DETECTION ===")
    continuous_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in continuous_cols:
        if col not in ['target', 'heartdisease']:  # Don't treat target as continuous
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # Using 3*IQR instead of 1.5 to be less aggressive
            upper_bound = Q3 + 3 * IQR
            
            outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                print(f"{col}: {outlier_count} extreme outliers detected (beyond 3*IQR)")
                # Option to remove extreme outliers (commented out by default)
                # df_clean = df_clean[~outliers]
    
    # Final data validation
    print("\n=== FINAL VALIDATION ===")
    print(f"Final dataset shape: {df_clean.shape}")
    print(f"Rows removed: {len(df) - len(df_clean)}")
    print(f"Percentage of data retained: {len(df_clean)/len(df)*100:.2f}%")
    
    if replaced_values:
        print(f"\nValues replaced with averages:")
        for col, count in replaced_values.items():
            print(f"  {col}: {count} values")
    
    # Check for any remaining issues
    print("\nFinal data quality check:")
    print(f"Missing values: {df_clean.isnull().sum().sum()}")
    print(f"Duplicates: {df_clean.duplicated().sum()}")
    
    # Display summary statistics
    print("\n=== CLEANED DATA SUMMARY ===")
    print(df_clean.describe())
    
    return df_clean, replaced_values
    
    # Clean Cholesterol (replace invalid values with average)
    if 'cholesterol' in column_mapping:
        chol_col = column_mapping['cholesterol']
        print(f"\nCleaning {chol_col} (Cholesterol):")
        print(f"Original range: {df_clean[chol_col].min()} - {df_clean[chol_col].max()}")
        
        # Identify invalid cholesterol (0 or negative, or extremely high >600)
        invalid_chol = (df_clean[chol_col] <= 0) | (df_clean[chol_col] > 600)
        invalid_chol_count = invalid_chol.sum()
        
        if invalid_chol_count > 0:
            # Calculate average from valid values only
            valid_chol_values = df_clean[chol_col][(df_clean[chol_col] > 0) & (df_clean[chol_col] <= 600)]
            chol_mean = valid_chol_values.mean()
            
            print(f"Replacing {invalid_chol_count} invalid cholesterol values with average: {chol_mean:.2f}")
            df_clean.loc[invalid_chol, chol_col] = chol_mean
        else:
            print("No invalid cholesterol values found")
    
    # Clean Resting Blood Pressure (replace invalid values with average)
    if 'resting_bp' in column_mapping:
        bp_col = column_mapping['resting_bp']
        print(f"\nCleaning {bp_col} (Resting Blood Pressure):")
        print(f"Original range: {df_clean[bp_col].min()} - {df_clean[bp_col].max()}")
        
        # Identify invalid BP (0 or negative, or extremely high >250)
        invalid_bp = (df_clean[bp_col] <= 0) | (df_clean[bp_col] > 250)
        invalid_bp_count = invalid_bp.sum()
        
        if invalid_bp_count > 0:
            # Calculate average from valid values only
            valid_bp_values = df_clean[bp_col][(df_clean[bp_col] > 0) & (df_clean[bp_col] <= 250)]
            bp_mean = valid_bp_values.mean()
            
            print(f"Replacing {invalid_bp_count} invalid resting BP values with average: {bp_mean:.2f}")
            df_clean.loc[invalid_bp, bp_col] = bp_mean
        else:
            print("No invalid resting BP values found")
    
    # Clean Oldpeak (replace negative values with average)
    if 'oldpeak' in column_mapping:
        oldpeak_col = column_mapping['oldpeak']
        print(f"\nCleaning {oldpeak_col} (Oldpeak):")
        print(f"Original range: {df_clean[oldpeak_col].min()} - {df_clean[oldpeak_col].max()}")
        
        # Identify negative oldpeak values
        invalid_oldpeak = df_clean[oldpeak_col] < 0
        invalid_oldpeak_count = invalid_oldpeak.sum()
        
        if invalid_oldpeak_count > 0:
            # Calculate average from valid (non-negative) values only
            valid_oldpeak_values = df_clean[oldpeak_col][df_clean[oldpeak_col] >= 0]
            oldpeak_mean = valid_oldpeak_values.mean()
            
            print(f"Replacing {invalid_oldpeak_count} negative oldpeak values with average: {oldpeak_mean:.2f}")
            df_clean.loc[invalid_oldpeak, oldpeak_col] = oldpeak_mean
        else:
            print("No negative oldpeak values found")
    
    # Clean ST Slope (should be 0, 1, or 2, but you mentioned 0 is invalid)
    if 'st_slope' in column_mapping:
        slope_col = column_mapping['st_slope']
        print(f"\nCleaning {slope_col} (ST Slope):")
        print(f"Unique values: {sorted(df_clean[slope_col].unique())}")
        
        # Remove rows with ST slope = 0 (as mentioned it's invalid)
        invalid_slope = df_clean[slope_col] == 0
        invalid_slope_count = invalid_slope.sum()
        
        if invalid_slope_count > 0:
            print(f"Removing {invalid_slope_count} rows with invalid ST slope (0)")
            df_clean = df_clean[~invalid_slope]
        else:
            print("No invalid ST slope values found")
    
    # Additional outlier detection using IQR method for continuous variables
    print("\n=== OUTLIER DETECTION ===")
    continuous_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in continuous_cols:
        if col not in ['target', 'heartdisease']:  # Don't treat target as continuous
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # Using 3*IQR instead of 1.5 to be less aggressive
            upper_bound = Q3 + 3 * IQR
            
            outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                print(f"{col}: {outlier_count} extreme outliers detected (beyond 3*IQR)")
                # Option to remove extreme outliers (commented out by default)
                # df_clean = df_clean[~outliers]
    
    # Final data validation
    print("\n=== FINAL VALIDATION ===")
    print(f"Final dataset shape: {df_clean.shape}")
    print(f"Rows removed: {len(df) - len(df_clean)}")
    print(f"Percentage of data retained: {len(df_clean)/len(df)*100:.2f}%")
    
    # Check for any remaining issues
    print("\nFinal data quality check:")
    print(f"Missing values: {df_clean.isnull().sum().sum()}")
    print(f"Duplicates: {df_clean.duplicated().sum()}")
    
    # Display summary statistics
    print("\n=== CLEANED DATA SUMMARY ===")
    print(df_clean.describe())
    
    return df_clean

def save_cleaned_data(df_clean, output_path='cleaned_heart_disease_data.csv'):
    """Save the cleaned dataset"""
    df_clean.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path}")

def create_data_quality_report(df_original, df_clean, replaced_values=None):
    """Create a visual data quality report"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Data Quality Report', fontsize=16)
    
    # 1. Data shape comparison
    categories = ['Original', 'Cleaned']
    rows = [len(df_original), len(df_clean)]
    axes[0, 0].bar(categories, rows, color=['red', 'green'], alpha=0.7)
    axes[0, 0].set_title('Dataset Size Comparison')
    axes[0, 0].set_ylabel('Number of Rows')
    
    # Add value labels on bars
    for i, v in enumerate(rows):
        axes[0, 0].text(i, v + max(rows)*0.01, str(v), ha='center', va='bottom')
    
    # 2. Missing values comparison
    missing_orig = df_original.isnull().sum().sum()
    missing_clean = df_clean.isnull().sum().sum()
    axes[0, 1].bar(['Original', 'Cleaned'], [missing_orig, missing_clean], 
                   color=['red', 'green'], alpha=0.7)
    axes[0, 1].set_title('Missing Values Comparison')
    axes[0, 1].set_ylabel('Number of Missing Values')
    
    # 3. Invalid values replaced
    if replaced_values:
        columns = list(replaced_values.keys())
        counts = list(replaced_values.values())
        axes[1, 0].bar(columns, counts, color='orange', alpha=0.7)
        axes[1, 0].set_title('Invalid Values Replaced with Average')
        axes[1, 0].set_ylabel('Number of Values Replaced')
        axes[1, 0].tick_params(axis='x', rotation=45)
    else:
        axes[1, 0].text(0.5, 0.5, 'No invalid values\nwere replaced', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Invalid Values Replaced')
    
    # 4. Data retention percentage
    retention = len(df_clean) / len(df_original) * 100
    axes[1, 1].pie([retention, 100-retention], labels=[f'Retained\n{retention:.1f}%', 
                                                       f'Removed\n{100-retention:.1f}%'],
                   colors=['green', 'red'], alpha=0.7, autopct='')
    axes[1, 1].set_title('Data Retention')
    
    plt.tight_layout()
    plt.savefig('data_quality_report.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Usage example
    file_path = "heart_statlog_cleveland_hungary_final.csv"  # Update with your file path
    
    try:
        # Load original data for comparison
        df_original = pd.read_csv(file_path)
        
        # Clean the data
        df_cleaned, replacement_stats = clean_heart_disease_data(file_path)
        
        # Save cleaned data
        save_cleaned_data(df_cleaned, 'cleaned_heart_disease_data.csv')
        
        # Create data quality report
        create_data_quality_report(df_original, df_cleaned, replacement_stats)
        
        print("\n=== DATA CLEANING COMPLETED SUCCESSFULLY ===")
        print("The cleaned dataset is ready for model training!")
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        print("Please update the file_path variable with the correct path to your dataset.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")