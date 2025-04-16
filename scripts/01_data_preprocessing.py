# 1. import pandas, numpy
# 2. define load_data(path) → DataFrame
# 3. define clean_data(df):
#      - handle missing values (e.g. drop or impute)
#      - drop irrelevant columns (ID, names)
#      - convert date columns to datetime
# 4. define encode_basic(df):
#      - label‑encode binary flags
#      - one‑hot encode small‑cardinality categoricals
# 5. in main:
#      - df = load_data('data/hotel_bookings.csv')
#      - df_clean = clean_data(df)
#      - df_enc = encode_basic(df_clean)
#      - save df_enc to 'outputs/cleaned_data.csv'
