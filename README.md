# Depression Severity Analysis Using Reported Health Data

This ZIP file contains the code and notebooks used to analyse behavioural, lifestyle, demographic, and clinical factors associated with short-term changes in PHQ-9 depression scores from the DiSCover study.

## Contents

### `01_data_exploration.ipynb`
Exploratory data analysis done on the raw, original dataset, including:
- PHQ-9 score distributions  
- missingness patterns  
- behavioural summaries  
- correlation analysis  
- visualisations  

### `preprocessing.py`
Script that prepares the dataset for modellinh:
- removes engineered Fitbit features  
- computes PHQ-9 change and derived variables (age, migraine indicator)  
- imputes missing values  

Outputs the cleaned dataset used in the modelling notebook.

### `02_modelling.ipynb`
Contains two parts:
1. Clustering: data standardisation, k-means clustering, silhouette scores, PCA visualisation, cluster profiling  
2. **Predictive modelling:** Logistic Regression and Random Forest classifiers, evaluation (precision, recall, F1, ROC–AUC), and model interpretation  

## How to Run
1. Open `01_data_exploration.ipynb` for exploratory analysis on original dataset.
2. Run `preprocessing.py` to generate the cleaned dataset for modelling
3. Open `02_modelling.ipynb` for clustering and supervised modelling.

## Requirements
The following core dependencies are used in the project:
pandas==2.3.2
numpy==2.3.2
matplotlib==3.10.6
seaborn==0.13.2
scikit-learn==1.7.1
pyarrow==22.0.0

Install them using:

```bash
pip install -r requirements.txt
