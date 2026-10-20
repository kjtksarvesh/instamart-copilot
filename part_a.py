import os
import sqlite3
import pandas as pd
import numpy as np

os.makedirs('data/clean', exist_ok=True)

# --- 1. Load Data ---
customers = pd.read_csv('data/raw/customers.csv')
orders = pd.read_csv('data/raw/orders.csv')
products = pd.read_csv('data/raw/products.csv')

raw_orders_count = len(orders) # Checkpoint: 13,084

# --- 2. Clean Customers ---
customers['city'] = customers['city'].str.strip().str.title()
gender_map = {
    'male': 'male', 'Male': 'male', 'M': 'male', 'm': 'male', 'MALE': 'male',
    'female': 'female', 'Female': 'female', 'F': 'female', 'f': 'female', 'FEMALE': 'female'
}
customers['gender'] = customers['gender'].map(gender_map).fillna('unknown')

customers['age'] = pd.to_numeric(customers['age'], errors='coerce')
customers.loc[(customers['age'] < 10) | (customers['age'] > 100), 'age'] = np.nan
customers['age'] = customers.groupby('city')['age'].transform(lambda x: x.fillna(x.median()))
customers['signup_date'] = pd.to_datetime(customers['signup_date'])

# --- 3. Clean Orders ---
orders['order_ts'] = pd.to_datetime(orders['order_ts'])

# Remove exact duplicates
orders = orders.drop_duplicates()

# Remove partial duplicates (keep latest)
orders = orders.sort_values('order_ts').drop_duplicates('order_id', keep='last')

# Clean order_amount
orders['order_amount'] = pd.to_numeric(orders['order_amount'], errors='coerce')
orders = orders.merge(products[['product_id', 'price_inr']], on='product_id', how='left')
missing_amt = orders['order_amount'].isna()
orders.loc[missing_amt, 'order_amount'] = round(orders.loc[missing_amt, 'price_inr'] * 
                                                orders.loc[missing_amt, 'quantity'] * 
                                                (1 - orders.loc[missing_amt, 'discount_pct'] / 100), 2)
orders = orders.drop(columns=['price_inr'])

# Clean delivery_minutes
orders['delivery_minutes'] = pd.to_numeric(orders['delivery_minutes'], errors='coerce')
orders.loc[orders['delivery_minutes'] < 0, 'delivery_minutes'] = np.nan

orders = orders.merge(products[['product_id', 'category']], on='product_id', how='left')
completed_mask = orders['status'] == 'completed'
orders.loc[completed_mask, 'delivery_minutes'] = orders[completed_mask].groupby('category')['delivery_minutes'].transform(lambda x: x.fillna(x.median()))
orders = orders.drop(columns=['category'])

# --- 4. Export Clean Data ---
customers.to_csv('data/clean/customers.csv', index=False)
orders.to_csv('data/clean/orders.csv', index=False)
products.to_csv('data/clean/products.csv', index=False)

# --- 5. SQLite Storage ---
conn = sqlite3.connect('data/instamart.db')
customers.assign(signup_date=customers['signup_date'].dt.strftime('%Y-%m-%d')).to_sql('customers', conn, if_exists='replace', index=False)
orders.assign(order_ts=orders['order_ts'].dt.strftime('%Y-%m-%d %H:%M:%S')).to_sql('orders', conn, if_exists='replace', index=False)
products.to_sql('products', conn, if_exists='replace', index=False)
conn.close()

# --- 6. Validation Assertions ---
completed_orders = orders[orders['status'] == 'completed']
cancelled_orders = orders[orders['status'] == 'cancelled']

assert raw_orders_count == 13084
assert len(orders) == 12309
assert len(completed_orders) == 11110
assert len(cancelled_orders) == 1199
assert orders['order_amount'].isna().sum() == 0
assert abs(completed_orders['order_amount'].sum() - 2998668.80) < 0.01
assert completed_orders['delivery_minutes'].isna().sum() == 0
assert cancelled_orders['delivery_minutes'].isna().sum() == 1199

print("ALL CHECKPOINTS PASSED")