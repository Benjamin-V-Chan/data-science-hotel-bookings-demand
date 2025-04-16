# 1. import pandas, numpy
# 2. load cleaned_data.csv
# 3. create new features:
#      - total_nights = stays_in_week_nights + stays_in_weekend_nights
#      - booking_rate = lead_time / total_nights
#      - is_family = (adults + children + babies) > 1
# 4. select X (drop target), y = is_canceled
# 5. split into train/test (e.g. 80/20) and save to outputs/features/
