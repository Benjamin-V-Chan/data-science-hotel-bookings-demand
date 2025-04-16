import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def load_data():
    X = pd.read_csv('outputs/features/X_train.csv')
    y = pd.read_csv('outputs/features/y_train.csv').squeeze()
    return X, y

def train():
    X_train, y_train = load_data()
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    os.makedirs('outputs/models', exist_ok=True)
    joblib.dump(pipe, 'outputs/models/best_model.pkl')

def main():
    train()

if __name__ == '__main__':
    main()
