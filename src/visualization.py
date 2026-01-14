import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Optional, List
from sklearn.metrics import confusion_matrix, roc_curve, auc

def set_plot_style():
    """
    Sets a consistent plotting style for the project.
    """
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'font.family': 'sans-serif'
    })

def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, title: str = "Confusion Matrix", save_path: Optional[str] = None):
    """
    Plots a heatmap of the confusion matrix.
    Args:
        save_path: If provided, saves the figure to this path (e.g., 'reports/figures/cm.png').
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
        
    plt.show()

def plot_roc_curve(models: Dict, X_test: pd.DataFrame, y_test: pd.Series, save_path: Optional[str] = None):
    """
    Plots ROC curves for multiple models to compare performance.
    """
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
        
    plt.show()

def plot_coefficients(df_coef: pd.DataFrame, top_n: int = 15, save_path: Optional[str] = None):
    """
    Plots a diverging bar chart for linear model coefficients.
    Red bars indicate factors driving Churn (Positive coef).
    Blue bars indicate factors driving Retention (Negative coef).
    """
    data = df_coef.head(top_n).copy()
    
    # Define colors based on sign (Red for Churn/Positive, Blue for Retention/Negative)
    colors = ['#D62728' if x > 0 else '#1F77B4' for x in data['Coefficient']]
    
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Coefficient', y='Feature', data=data, palette=colors, hue='Feature', legend=False)
    
    # Add zero line for reference
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    plt.title(f'Top {top_n} Factors Influencing Churn (Logistic Regression)')
    plt.xlabel('Coefficient Impact (Left: Loyalty, Right: Churn Risk)')
    plt.ylabel('')
    
    # Custom Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#D62728', label='Increases Churn Risk'),
        Patch(facecolor='#1F77B4', label='Increases Loyalty')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to {save_path}")
        
    plt.show()