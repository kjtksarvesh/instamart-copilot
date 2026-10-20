import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

os.makedirs('outputs', exist_ok=True)

orders = pd.read_csv('data/clean/orders.csv', parse_dates=['order_ts'])
customers = pd.read_csv('data/clean/customers.csv', parse_dates=['signup_date'])
products = pd.read_csv('data/clean/products.csv')
orders = orders.merge(products[['product_id', 'category']], on='product_id', how='left')

# 1. RFM Clustering
snapshot = pd.to_datetime('2025-10-01')
completed = orders[orders['status'] == 'completed']
rfm = completed.groupby('customer_id').agg(
    recency=('order_ts', lambda x: (snapshot - x.max()).days),
    frequency=('order_id', 'nunique'),
    monetary=('order_amount', 'sum')
).reset_index()

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])

kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
rfm['segment_id'] = kmeans.fit_predict(rfm_scaled)

# Assign names and premium flag
segment_means = rfm.groupby('segment_id')['monetary'].mean()
premium_id = segment_means.idxmax()
rfm['is_premium'] = rfm['segment_id'] == premium_id
rfm['segment_name'] = rfm['segment_id'].map(lambda x: 'Premium' if x == premium_id else f'Segment_{x}')
rfm.to_csv('outputs/customer_segments.csv', index=False)

# 2. Lapse Prediction Data Prep
cutoff = pd.to_datetime('2025-09-01')
pre_cutoff = orders[orders['order_ts'] < cutoff]
cohort_cust = pre_cutoff[pre_cutoff['status'] == 'completed']['customer_id'].unique()

sept_orders = orders[(orders['order_ts'] >= cutoff) & (orders['order_ts'] < snapshot) & (orders['status'] == 'completed')]
lapsed_cust = set(cohort_cust) - set(sept_orders['customer_id'].unique())

features = []
for cid in cohort_cust:
    cust_orders = pre_cutoff[pre_cutoff['customer_id'] == cid]
    cust_comp = cust_orders[cust_orders['status'] == 'completed']
    cust_info = customers[customers['customer_id'] == cid].iloc[0]
    
    features.append({
        'customer_id': cid,
        'recency': (cutoff - cust_comp['order_ts'].max()).days,
        'frequency': cust_comp['order_id'].nunique(),
        'monetary': cust_comp['order_amount'].sum(),
        'avg_discount': cust_comp['discount_pct'].mean(),
        'cancel_rate': len(cust_orders[cust_orders['status'] == 'cancelled']) / len(cust_orders) if len(cust_orders) > 0 else 0,
        'avg_delivery': cust_comp['delivery_minutes'].mean(),
        'fav_category': cust_comp['category'].mode()[0] if not cust_comp.empty else 'Unknown',
        'city': cust_info['city'],
        'age': cust_info['age'],
        'gender': cust_info['gender'],
        'tenure': (cutoff - cust_info['signup_date']).days,
        'lapsed': 1 if cid in lapsed_cust else 0
    })

df_features = pd.DataFrame(features)
X = df_features.drop(['customer_id', 'lapsed'], axis=1)
y = df_features['lapsed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Pipeline without Leakage
numeric_features = ['recency', 'frequency', 'monetary', 'avg_discount', 'cancel_rate', 'avg_delivery', 'age', 'tenure']
categorical_features = ['fav_category', 'city', 'gender']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Param grid tuning
param_grid = {
    'classifier__max_depth': [3, 5, 8],
    'classifier__min_samples_leaf': [5, 10, 20]
}
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
joblib.dump(best_model, 'outputs/lapse_model.joblib')

# 4. Retention List
test_results = X_test.copy()
test_results['customer_id'] = df_features.loc[X_test.index, 'customer_id']
test_results['lapse_probability'] = best_model.predict_proba(X_test)[:, 1]
test_results['actually_lapsed'] = y_test

premium_cust = rfm[rfm['is_premium']]['customer_id'].tolist()
retention_list = test_results[test_results['customer_id'].isin(premium_cust)].sort_values('lapse_probability', ascending=False).head(10)
retention_list[['customer_id', 'lapse_probability', 'actually_lapsed']].to_csv('outputs/retention_list.csv', index=False)

print("Part B completed successfully.")