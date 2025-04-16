import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.drop(columns=['agent', 'company', 'reservation_status_date'], errors='ignore')
    df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' +
                                        df['arrival_date_month'] + '-' +
                                        df['arrival_date_day_of_month'].astype(str),
                                        format='%Y-%B-%d')
    df = df.drop(columns=['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month'])
    df = df.fillna({'children': 0, 'country': 'Unknown'})
    return df

def encode_basic(df):
    df['is_canceled'] = df['is_canceled'].astype(int)
    df['hotel'] = df['hotel'].map({'City Hotel':0, 'Resort Hotel':1})
    cat_cols = ['meal','market_segment','distribution_channel','reserved_room_type','deposit_type','customer_type']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

def main():
    df = load_data('data/hotel_bookings.csv')
    df_clean = clean_data(df)
    df_enc = encode_basic(df_clean)
    df_enc.to_csv('outputs/cleaned_data.csv', index=False)

if __name__ == '__main__':
    main()
