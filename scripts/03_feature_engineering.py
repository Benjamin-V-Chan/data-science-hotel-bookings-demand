import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def engineer(df):
    df['total_nights'] = df['stays_in_week_nights'] + df['stays_in_weekend_nights']
    df['booking_rate'] = df['lead_time'] / (df['total_nights'] + 1)
    df['is_family'] = ((df['adults'] + df['children'] + df['babies']) > 1).astype(int)
    return df

def split_save(df):
    X = df.drop(columns=['is_canceled'])
    y = df['is_canceled']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        stratify=y,
                                                        random_state=42)
    X_train.to_csv('outputs/features/X_train.csv', index=False)
    X_test.to_csv('outputs/features/X_test.csv', index=False)
    y_train.to_csv('outputs/features/y_train.csv', index=False)
    y_test.to_csv('outputs/features/y_test.csv', index=False)

def main():
    df = load_data('outputs/cleaned_data.csv')
    df_feat = engineer(df)
    split_save(df_feat)

if __name__ == '__main__':
    main()
