# scripts/05_model_evaluation.py
import pandas as pd
import joblib
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
import matplotlib.pyplot as plt
import os
import json

def load_all():
    model = joblib.load('outputs/models/best_model.pkl')
    X_test = pd.read_csv('outputs/features/X_test.csv')
    y_test = pd.read_csv('outputs/features/y_test.csv').squeeze()
    return model, X_test, y_test

def evaluate(model, X, y):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:,1]
    metrics = {
        'accuracy': accuracy_score(y, preds),
        'precision': precision_score(y, preds),
        'recall': recall_score(y, preds),
        'f1': f1_score(y, preds),
        'roc_auc': roc_auc_score(y, probs)
    }
    return preds, probs, metrics

def save_metrics(metrics):
    os.makedirs('outputs/evaluation', exist_ok=True)
    with open('outputs/evaluation/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

def plot_roc(y, probs):
    fpr, tpr, _ = roc_curve(y, probs)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0,1],[0,1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('outputs/evaluation/roc_curve.png')
    plt.close()

def plot_confmat(y, preds):
    cm = confusion_matrix(y, preds)
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i,j], ha='center', va='center')
    plt.savefig('outputs/evaluation/confusion_matrix.png')
    plt.close()

def main():
    model, X_test, y_test = load_all()
    preds, probs, metrics = evaluate(model, X_test, y_test)
    save_metrics(metrics)
    plot_roc(y_test, probs)
    plot_confmat(y_test, preds)

if __name__ == '__main__':
    main()
