import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any, Union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_baseline_models(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Trains a set of baseline models (Logistic Regression, RF, XGBoost) 
    with handling for class imbalance.
    
    Returns:
        metrics_df (pd.DataFrame): Performance metrics for comparison.
        trained_models (dict): Dictionary of fitted model objects.
    """
    print("Training baseline models with class weights...")
    
    # Calculate scale_pos_weight for XGBoost (Total Negative / Total Positive)
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    
    models = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced', 
            max_iter=1000, 
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            class_weight='balanced', 
            n_estimators=100, 
            max_depth=8, 
            random_state=42, 
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"Fitting {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models

def evaluate_models(models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Evaluates trained models on the test set and returns a comparison table.
    """
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        # specific check for predict_proba availability
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = [0] * len(y_test)
            
        results.append({
            'Model': name,
            'ROC-AUC': roc_auc_score(y_test, y_prob),
            'Recall': recall_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'Accuracy': accuracy_score(y_test, y_pred)
        })
        
    return pd.DataFrame(results).set_index('Model').sort_values('Recall', ascending=False)

def get_model_coefficients(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """
    Extracts coefficients from a linear model (e.g., Logistic Regression) 
    to interpret feature impact (Direction & Magnitude).
    """
    if not hasattr(model, 'coef_'):
        raise ValueError("Model must be linear and have 'coef_' attribute.")
        
    coefs = model.coef_[0]
    
    df_coef = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs
    })
    
    df_coef['Abs_Impact'] = df_coef['Coefficient'].abs()
    return df_coef.sort_values('Abs_Impact', ascending=False)

import matplotlib.pyplot as plt
import seaborn as sns

# Modelin Test seti için ürettiği olasılıkları alalım
y_probs = champion_model.predict_proba(X_test_scaled)[:, 1]

# İstatistiğine bakalım
print(f"Max Olasılık: {y_probs.max():.4f}")
print(f"Ortalama Olasılık: {y_probs.mean():.4f}")

# Görselleştirelim
plt.figure(figsize=(10,4))
sns.histplot(y_probs, bins=50, kde=True)
plt.axvline(x=0.75, color='r', linestyle='--', label='Senin Eşiğin (0.75)')
plt.title('Churn Olasılık Dağılımı')
plt.legend()
plt.show()

def generate_leads(model: Any, X_scaled: pd.DataFrame, df_raw: pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
    """
    Generates a list of high-risk customers by combining model predictions 
    with raw business data.
    
    Args:
        model: Trained classifier (supports predict_proba).
        X_scaled: Scaled features for prediction.
        df_raw: Original dataframe (unscaled) for reporting.
        threshold: Probability cutoff for 'High Risk' label.
        
    Returns:
        DataFrame of high-risk customers with raw metrics.
    """
    # 1. Predict probabilities
    probs = model.predict_proba(X_scaled)[:, 1]
    
    # 2. Align predictions with raw data via index
    leads = df_raw.loc[X_scaled.index].copy()
    leads['churn_probability'] = probs
    
    # 3. Filter by threshold
    high_risk = leads[leads['churn_probability'] > threshold].sort_values('churn_probability', ascending=False)
    
    # 4. Select readable columns if they exist
    display_cols = ['churn_probability', 'frequency', 'monetary_value', 
                   'avg_delivery_days', 'avg_review_score', 'tenure_days']
    
    final_cols = [c for c in display_cols if c in high_risk.columns]
    
    return high_risk[final_cols]