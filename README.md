# data-science-hotel-bookings-demand

## Project Overview

This project analyzes hotel booking demand using a modular data science pipeline. It covers data preprocessing, exploratory data analysis, feature engineering, model training, and model evaluation to predict booking cancellations.

## Folder Structure

```
project-root/
├── data/
│   └── hotel_bookings.csv     # Raw dataset
├── scripts/                   # Analysis scripts
│   ├── 01_data_preprocessing.py
│   ├── 02_exploratory_data_analysis.py
│   ├── 03_feature_engineering.py
│   ├── 04_model_training.py
│   └── 05_model_evaluation.py
└── outputs/                   # Generated outputs
    ├── cleaned_data.csv
    ├── eda/                  # EDA figures and summaries
    ├── features/             # Train/test feature sets
    ├── models/               # Saved model artifacts
    └── evaluation/           # Metrics and evaluation plots
```

## Usage

1. **Setup the Project:**
   Clone the repository.
   Ensure you have Python installed.
   Install required dependencies using the requirements.txt file.
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Data Preprocessing:**
   ```bash
   python scripts/01_data_preprocessing.py
   ```

3. **Perform Exploratory Data Analysis:**
   ```bash
   python scripts/02_exploratory_data_analysis.py
   ```

4. **Generate and Split Features:**
   ```bash
   python scripts/03_feature_engineering.py
   ```

5. **Train the Model:**
   ```bash
   python scripts/04_model_training.py
   ```

6. **Evaluate Model Performance:**
   ```bash
   python scripts/05_model_evaluation.py
   ```

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

## Acknowledgments

dataset name: Hotel bookings demand  
dataset author: Cookedwang  
dataset source: https://www.kaggle.com/datasets/qucwang/hotel-bookings-analysis-dataset