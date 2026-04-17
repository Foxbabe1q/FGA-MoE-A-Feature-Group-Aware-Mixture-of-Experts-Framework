"""
Evaluation metrics and utilities
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    brier_score_loss, roc_curve, precision_recall_curve
)
from scipy import stats

def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    """Compute all evaluation metrics"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Binary metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'auroc': roc_auc_score(y_true, y_pred_proba),
        'auprc': average_precision_score(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'brier_score': brier_score_loss(y_true, y_pred_proba)
    }
    
    return metrics

def bootstrap_confidence_interval(y_true, y_pred_proba, metric_fn, 
                                   n_bootstrap=1000, confidence=0.95):
    """Compute bootstrap confidence interval for a metric"""
    n_samples = len(y_true)
    
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred_proba[indices]
        
        try:
            score = metric_fn(y_true_boot, y_pred_boot)
            bootstrap_scores.append(score)
        except:
            continue
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    # Compute confidence interval
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_scores, alpha/2 * 100)
    upper = np.percentile(bootstrap_scores, (1 - alpha/2) * 100)
    mean = np.mean(bootstrap_scores)
    std = np.std(bootstrap_scores)
    
    return {
        'mean': mean,
        'std': std,
        'ci_lower': lower,
        'ci_upper': upper
    }

def compute_metrics_with_ci(y_true, y_pred_proba, n_bootstrap=1000, threshold=0.5):
    """Compute metrics with confidence intervals for all metrics"""
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    # Point estimates
    base_metrics = compute_metrics(y_true, y_pred_proba, threshold)
    
    # Bootstrap CIs for all metrics
    # AUROC and AUPRC use probabilities
    auroc_ci = bootstrap_confidence_interval(
        y_true, y_pred_proba, roc_auc_score, n_bootstrap
    )
    
    auprc_ci = bootstrap_confidence_interval(
        y_true, y_pred_proba, average_precision_score, n_bootstrap
    )
    
    # For other metrics, we need to threshold predictions
    def accuracy_fn(y_t, y_p):
        return accuracy_score(y_t, (y_p >= threshold).astype(int))
    
    def precision_fn(y_t, y_p):
        return precision_score(y_t, (y_p >= threshold).astype(int), zero_division=0)
    
    def recall_fn(y_t, y_p):
        return recall_score(y_t, (y_p >= threshold).astype(int), zero_division=0)
    
    def f1_fn(y_t, y_p):
        return f1_score(y_t, (y_p >= threshold).astype(int), zero_division=0)
    
    def specificity_fn(y_t, y_p):
        y_pred = (y_p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_t, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    accuracy_ci = bootstrap_confidence_interval(
        y_true, y_pred_proba, accuracy_fn, n_bootstrap
    )
    
    precision_ci = bootstrap_confidence_interval(
        y_true, y_pred_proba, precision_fn, n_bootstrap
    )
    
    recall_ci = bootstrap_confidence_interval(
        y_true, y_pred_proba, recall_fn, n_bootstrap
    )
    
    f1_ci = bootstrap_confidence_interval(
        y_true, y_pred_proba, f1_fn, n_bootstrap
    )
    
    specificity_ci = bootstrap_confidence_interval(
        y_true, y_pred_proba, specificity_fn, n_bootstrap
    )
    
    results = {
        'auroc': base_metrics['auroc'],
        'auroc_ci': f"[{auroc_ci['ci_lower']:.4f}, {auroc_ci['ci_upper']:.4f}]",
        'auroc_std': auroc_ci['std'],
        'auprc': base_metrics['auprc'],
        'auprc_ci': f"[{auprc_ci['ci_lower']:.4f}, {auprc_ci['ci_upper']:.4f}]",
        'auprc_std': auprc_ci['std'],
        'accuracy': base_metrics['accuracy'],
        'accuracy_ci': f"[{accuracy_ci['ci_lower']:.4f}, {accuracy_ci['ci_upper']:.4f}]",
        'accuracy_std': accuracy_ci['std'],
        'precision': base_metrics['precision'],
        'precision_ci': f"[{precision_ci['ci_lower']:.4f}, {precision_ci['ci_upper']:.4f}]",
        'precision_std': precision_ci['std'],
        'recall': base_metrics['recall'],
        'recall_ci': f"[{recall_ci['ci_lower']:.4f}, {recall_ci['ci_upper']:.4f}]",
        'recall_std': recall_ci['std'],
        'f1': base_metrics['f1'],
        'f1_ci': f"[{f1_ci['ci_lower']:.4f}, {f1_ci['ci_upper']:.4f}]",
        'f1_std': f1_ci['std'],
        'specificity': base_metrics['specificity'],
        'specificity_ci': f"[{specificity_ci['ci_lower']:.4f}, {specificity_ci['ci_upper']:.4f}]",
        'specificity_std': specificity_ci['std'],
        'brier_score': base_metrics['brier_score']
    }
    
    return results
