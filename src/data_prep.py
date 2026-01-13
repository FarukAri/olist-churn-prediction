import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def load_and_clean_data(data_path: str = 'data/raw/') -> Dict[str, pd.DataFrame]:
    """
    Loads raw Olist datasets and performs initial type conversions.
    """
    path = Path(data_path)
    print(f"Loading data from: {path.resolve()}")
    
    files = {
        'customers': 'olist_customers_dataset.csv',
        'orders': 'olist_orders_dataset.csv',
        'payments': 'olist_order_payments_dataset.csv',
        'reviews': 'olist_order_reviews_dataset.csv',
        'geolocation': 'olist_geolocation_dataset.csv' 
    }
    
    dfs = {}
    for key, filename in files.items():
        file_path = path / filename
        if file_path.exists():
            dfs[key] = pd.read_csv(file_path)
        else:
            print(f"Warning: {filename} not found.")

    # Standardize dates
    if 'orders' in dfs:
        date_cols = [
            'order_purchase_timestamp', 
            'order_approved_at', 
            'order_delivered_carrier_date', 
            'order_delivered_customer_date', 
            'order_estimated_delivery_date'
        ]
        for col in date_cols:
            dfs['orders'][col] = pd.to_datetime(dfs['orders'][col], errors='coerce')
            
        # Keep only delivered orders for consistent feature engineering
        dfs['orders'] = dfs['orders'][dfs['orders']['order_status'] == 'delivered'].copy()
    
    return dfs

def generate_churn_labels(df_orders: pd.DataFrame, df_customers: pd.DataFrame, threshold_days: int = 280) -> pd.DataFrame:
    """
    Generates the target variable based on inactivity.
    Churn (1): No purchase in the last 'threshold_days'.
    """
    print("Generating churn labels...")
    
    # Merge for unique customer tracking
    df = df_orders.merge(df_customers, on='customer_id', how='inner')
    
    snapshot_date = df['order_purchase_timestamp'].max()
    
    # Calculate recency per unique customer
    last_orders = df.groupby('customer_unique_id')['order_purchase_timestamp'].max().reset_index()
    last_orders.columns = ['customer_unique_id', 'last_order_date']
    
    last_orders['recency'] = (snapshot_date - last_orders['last_order_date']).dt.days
    last_orders['is_churn'] = (last_orders['recency'] > threshold_days).astype(int)
    
    churn_rate = last_orders['is_churn'].mean()
    print(f"Snapshot Date: {snapshot_date.date()} | Churn Rate: {churn_rate:.2%}")
    
    return last_orders

def build_features(dfs: Dict[str, pd.DataFrame], df_targets: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transactional data into a customer-centric feature set (RFM + Logistics + Satisfaction).
    """
    print("Building feature set...")
    
    orders = dfs['orders']
    customers = dfs['customers']
    payments = dfs['payments']
    reviews = dfs['reviews']
    
    # 1. Logistics Features
    orders_cust = orders.merge(customers, on='customer_id', how='inner')
    orders_cust['delivery_days'] = (orders_cust['order_delivered_customer_date'] - orders_cust['order_purchase_timestamp']).dt.days
    
    feat_logistics = orders_cust.groupby('customer_unique_id').agg({
        'order_id': 'count',
        'delivery_days': 'mean',
        'customer_state': 'first' # Required for state-based imputation
    }).reset_index().rename(columns={
        'order_id': 'frequency', 
        'delivery_days': 'avg_delivery_days'
    })
    
    # 2. Monetary Features
    # Chain merges to link payments to unique customers
    pay_merged = payments.merge(orders[['order_id', 'customer_id']], on='order_id') \
                         .merge(customers[['customer_id', 'customer_unique_id']], on='customer_id')
    
    feat_monetary = pay_merged.groupby('customer_unique_id').agg({
        'payment_value': 'sum',
        'payment_installments': 'mean'
    }).reset_index().rename(columns={
        'payment_value': 'monetary_value', 
        'payment_installments': 'avg_installments'
    })
    
    # 3. Satisfaction Features
    rev_merged = reviews.merge(orders[['order_id', 'customer_id']], on='order_id') \
                        .merge(customers[['customer_id', 'customer_unique_id']], on='customer_id')
    
    feat_reviews = rev_merged.groupby('customer_unique_id')['review_score'].mean().reset_index() \
                             .rename(columns={'review_score': 'avg_review_score'})
    
    # Merge all features
    df_final = feat_logistics.merge(feat_monetary, on='customer_unique_id', how='left') \
                             .merge(feat_reviews, on='customer_unique_id', how='left') \
                             .merge(df_targets[['customer_unique_id', 'is_churn', 'recency']], on='customer_unique_id', how='inner')
    
    # --- Context-Aware Imputation ---
    
    # Delivery days: Impute with State average (Geospatial context)
    state_means = df_final.groupby('customer_state')['avg_delivery_days'].transform('mean')
    df_final['avg_delivery_days'] = df_final['avg_delivery_days'].fillna(state_means)
    df_final['avg_delivery_days'] = df_final['avg_delivery_days'].fillna(df_final['avg_delivery_days'].mean())
    
    # Review scores: Impute with neutral score (3.0) and add binary flag
    df_final['has_review'] = df_final['avg_review_score'].notnull().astype(int)
    df_final['avg_review_score'] = df_final['avg_review_score'].fillna(3.0)
    
    # Monetary: Missing means zero spend in this context
    df_final[['monetary_value', 'avg_installments']] = df_final[['monetary_value', 'avg_installments']].fillna(0)
    
    # Cleanup
    df_final.drop(columns=['customer_state'], inplace=True, errors='ignore')
    
    return df_final

def split_and_scale(df: pd.DataFrame, target_col: str = 'is_churn', test_size: float = 0.2):
    """
    Prepares final arrays for modeling:
    1. Filters for repeat buyers (frequency >= 2).
    2. Splits into Train/Test.
    3. Scales features using RobustScaler (outlier-resistant).
    """
    print("Preprocessing for model input...")
    
    # Strategic Filter: Focus on customers with proven intent (2+ orders)
    df_elite = df[df['frequency'] >= 2].copy()
    print(f"Filtering: Retained {len(df_elite)} repeat buyers from {len(df)} total customers.")
    
    cols_to_drop = ['customer_unique_id', 'recency', target_col]
    X = df_elite.drop(columns=cols_to_drop, errors='ignore')
    y = df_elite[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, df_elite