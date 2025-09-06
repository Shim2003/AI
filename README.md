# Heart Disease Prediction Project

A comprehensive machine learning project for predicting heart disease using multiple algorithms including KNN, Logistic Regression, Random Forest, and Naive Bayes. Features data preprocessing, GUI training interfaces, and a Streamlit web application with personalized medical recommendations.

## Features

- Data Preprocessing: Automated data cleaning and validation
- Multiple ML Models: KNN, Logistic Regression, Random Forest, Naive Bayes
- GUI Training Interfaces: User-friendly Tkinter applications for model training
- Web Application: Streamlit-based prediction interface
- Comprehensive Analysis: EDA, cross-validation, hyperparameter tuning
- Model Persistence: Save and load trained models
- Medical Recommendations: Evidence-based lifestyle advice

## Project Structure

heart-disease-prediction/
├── data_preprocessor.py      # Data cleaning and preprocessing
├── eda.py                   # Exploratory Data Analysis
├── knn.py                   # KNN model with GUI trainer
├── logReg.py                # Logistic Regression (command line)
├── logReg_gui.py            # Logistic Regression with GUI
├── naive_bayes.py           # Naive Bayes with GUI trainer
├── random_forest_gui.py     # Random Forest with GUI trainer
├── random_forest_model.py   # Random Forest (command line)
├── ui.py                    # Streamlit web application
├── requirements.txt         # Python dependencies
└── README.md               # This file

## Quick Start

Get up and running in 5 minutes:

### Install Dependencies
pip install -r requirements.txt

### Prepare Data
python data_preprocessor.py

### Train a Model
python knn.py

### Launch Web App
streamlit run ui.py

### Open Browser
Navigate to http://localhost:8501

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Required Packages

Create a requirements.txt file with the following content:

pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.2.0
matplotlib>=3.5.0
seaborn>=0.11.0
streamlit>=1.28.0
joblib>=1.2.0
scipy>=1.9.0

### Installation Steps

1. Clone or download the project files

2. Install dependencies:
   pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib scipy
   
   Or using requirements file:
   pip install -r requirements.txt

3. Verify installation:
   python -c "import pandas, sklearn, streamlit; print('All packages installed successfully!')"

## Dataset Requirements

Make sure you have a CSV file named "heart_statlog_cleveland_hungary_final.csv" in your project directory. 

The dataset should contain the following columns:
- age: Age in years
- sex: Gender (0 = Female, 1 = Male)
- chest pain type: Type of chest pain (1-4)
- resting bp s: Resting blood pressure
- cholesterol: Serum cholesterol level
- fasting blood sugar: Fasting blood sugar > 120 mg/dl (0/1)
- resting ecg: Resting electrocardiogram results (0-2)
- max heart rate: Maximum heart rate achieved
- exercise angina: Exercise induced angina (0/1)
- oldpeak: ST depression induced by exercise
- ST slope: Slope of the peak exercise ST segment (1-3)
- target: Heart disease presence (0 = No, 1 = Yes)

## Usage Guide

### Step 1: Data Preprocessing

Clean and preprocess your raw dataset:

python data_preprocessor.py

What it does:
- Removes duplicates and handles missing values
- Replaces invalid values with averages
- Generates cleaned dataset: cleaned_heart_disease_data.csv
- Creates data quality report with visualizations

### Step 2: Exploratory Data Analysis (Optional)

python eda.py

Generates:
- Distribution plots for numerical features
- Correlation heatmap between variables

### Step 3: Train Machine Learning Models

#### Option A: GUI-Based Training (Recommended)

KNN Model:
python knn.py

Logistic Regression:
python logReg_gui.py

Random Forest:
python random_forest_gui.py

Naive Bayes:
python naive_bayes.py

Each GUI provides:
- Train/test split configuration
- Hyperparameter tuning with GridSearchCV
- Cross-validation results
- Performance metrics and visualizations
- Model saving functionality

#### Option B: Command Line Training

Logistic Regression:
python logReg.py

Random Forest:
python random_forest_model.py

### Step 4: Launch Web Application

After training at least one model (recommended: KNN):

streamlit run ui.py

Access the application:
- Open browser to http://localhost:8501
- Mobile-friendly responsive design

Features:
- Heart Disease Prediction: Input patient data for real-time predictions
- Lifestyle Recommendations: Personalized health advice based on risk assessment
- Risk Stratification: Critical, High, Medium, Low risk categories
- Visual Risk Assessment: Probability scores and risk indicators

## Application User Guides

### GUI Training Applications

Step-by-step workflow:

1. Launch the desired model trainer:
   python knn.py

2. Configure train/test split ratio (recommended: 80/20)

3. Train the model with automated hyperparameter tuning

4. View comprehensive results:
   - Accuracy and ROC-AUC scores
   - Confusion matrix heatmap
   - ROC curve visualization
   - Detailed classification report

5. Save the trained model for deployment

### Streamlit Web Application

For Healthcare Providers:

1. Navigate to the prediction page
2. Input patient information:
   - Demographics: Age, sex
   - Vital Signs: Blood pressure, heart rate, cholesterol
   - Medical History: Chest pain type, ECG results, exercise tolerance
3. Get instant risk assessment with probability score
4. Review evidence-based lifestyle recommendations
5. Follow prioritized action plan based on risk level

Risk Categories:
- Critical (>70%): Immediate medical attention required
- Moderate (50-70%): Schedule physician consultation
- Low (<50%): Continue preventive care

## Model Performance

Benchmark results on the heart disease dataset:

KNN: 85-90% Accuracy, 0.90-0.95 ROC-AUC - Distance-based classification
Logistic Regression: 85-88% Accuracy, 0.88-0.92 ROC-AUC - Interpretable, linear relationships
Random Forest: 88-92% Accuracy, 0.90-0.95 ROC-AUC - Feature importance, non-linear data
Naive Bayes: 80-85% Accuracy, 0.85-0.90 ROC-AUC - Baseline, probabilistic approach

### Recommended Model Selection

- Clinical Use: Logistic Regression (interpretable coefficients)
- Highest Accuracy: Random Forest or KNN
- Fastest Prediction: Naive Bayes
- Feature Analysis: Random Forest (feature importance)

## Troubleshooting

### Common Issues & Solutions

File Not Found Error:
FileNotFoundError: 'cleaned_heart_disease_data.csv' not found

Solution: Run data preprocessing first:
python data_preprocessor.py

Module Import Error:
ModuleNotFoundError: No module named 'streamlit'

Solution: Install missing packages:
pip install streamlit pandas scikit-learn matplotlib seaborn

Tkinter Not Available (Linux):
ModuleNotFoundError: No module named 'tkinter'

Solution: Install Tkinter:
sudo apt-get install python3-tk

Model File Missing in Web App:
FileNotFoundError: 'final_knn_model.pkl' not found

Solution: Train the KNN model first:
python knn.py

### Performance Tips

- Large Datasets: Use n_jobs=-1 for parallel processing
- Memory Issues: Reduce parameter grid size or use smaller splits
- GUI Responsiveness: Close unused visualization windows
- Web App Speed: Use lighter models (Logistic Regression) for faster predictions

## Advanced Features

### Automated Workflows

Batch Training Script:
# Train all models sequentially
python data_preprocessor.py && python logReg.py && python random_forest_model.py && echo "All models trained successfully!"

### Customization Options

Modify hyperparameter grids in model files:
# In knn.py - adjust parameter search space
param_grid = {
    'select__k': [3, 5, 7, 10],
    'knn__n_neighbors': [5, 7, 9],
    'knn__weights': ['uniform', 'distance'],
}

Customize web app recommendations:
# In ui.py - add new recommendation categories
recommendations.append({
    'category': 'Custom Health Tip',
    'icon': 'icon',
    'advice': 'Your personalized advice here',
    'priority': 'Medium'
})

## Future Enhancements

### Next Steps

1. Advanced Feature Engineering
   - Polynomial features for non-linear relationships
   - Feature interactions and domain-specific combinations

2. Ensemble Methods
   - Voting classifiers combining multiple models
   - Stacking algorithms for improved accuracy

3. Cloud Deployment
   - Docker containerization
   - AWS/Azure/GCP deployment options
   - Scalable API endpoints

4. Healthcare Integration
   - FHIR-compliant data format support
   - Electronic Health Record (EHR) integration
   - HIPAA-compliant security measures

5. Mobile Application
   - React Native or Flutter mobile app
   - Offline prediction capabilities
   - Push notifications for health reminders

## Contributing

We welcome contributions! Here's how to get started:

### Development Setup

# Fork and clone the repository
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch: git checkout -b feature-name
3. Make your changes with proper testing
4. Document new functionality
5. Test your changes thoroughly
6. Submit a pull request

### Bug Reports

Found a bug? Please create an issue with:
- Clear description of the problem
- Steps to reproduce
- System information (OS, Python version)
- Sample data (if applicable)

### Feature Requests

Have an idea? Open an issue with:
- Clear description of the proposed feature
- Use case and benefits for healthcare
- Implementation suggestions (optional)

## License & Legal

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Healthcare Compliance
- Educational Use: Designed for learning and research purposes
- Data Privacy: No patient data is stored or transmitted
- Regulatory: Not approved for clinical diagnosis
- Medical Advice: Consult healthcare providers for medical decisions

### Data Security
- Local Processing: All computations performed locally
- No Data Collection: Patient inputs are not saved or shared
- Privacy First: No external API calls for sensitive data

## Important Disclaimers

Medical Disclaimer: This tool is for educational and research purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

Research Use: While based on established medical datasets, this model should undergo clinical validation before any healthcare application.

Accuracy Limitations: Machine learning models can have errors and biases. Clinical judgment should always take precedence.

## Support & Contact

- Issues: Create a GitHub issue for bug reports
- Discussions: Use GitHub Discussions for questions
- Documentation: Check the wiki for detailed guides
- Star this repo if you find it helpful!

Built with love for Healthcare Innovation