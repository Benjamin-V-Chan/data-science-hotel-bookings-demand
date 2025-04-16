import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(path):
    return pd.read_csv(path)

def summary_stats(df):
    stats = df.describe(include='all').T
    stats.to_csv('outputs/eda/summary_stats.csv')

def plot_distributions(df):
    os.makedirs('outputs/eda', exist_ok=True)
    plt.figure()
    df['lead_time'].hist(bins=50)
    plt.title('Lead Time Distribution')
    plt.savefig('outputs/eda/lead_time_dist.png')
    plt.close()

    plt.figure()
    sns.countplot(x='is_canceled', data=df)
    plt.title('Cancellation Counts')
    plt.savefig('outputs/eda/cancel_counts.png')
    plt.close()

def plot_correlation(df):
    plt.figure(figsize=(12,10))
    corr = df.select_dtypes(include='number').corr()
    sns.heatmap(corr, cmap='coolwarm', center=0)
    plt.title('Numeric Feature Correlation')
    plt.savefig('outputs/eda/correlation_heatmap.png')
    plt.close()

def main():
    df = load_data('outputs/cleaned_data.csv')
    summary_stats(df)
    plot_distributions(df)
    plot_correlation(df)

if __name__ == '__main__':
    main()
