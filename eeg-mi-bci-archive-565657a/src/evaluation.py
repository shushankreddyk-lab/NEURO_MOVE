"""
Evaluation & LOSO (Leave-One-Subject-Out) Cross-Validation.
"""
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from typing import Dict, List, Tuple, Any, Optional
import json
from pathlib import Path


def loso_cv_scores(
    pipeline_class,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    pipeline_kwargs: Dict = None,
    fit_kwargs: Dict = None,
    n_jobs: int = 1
) -> Dict[str, Any]:
    """
    Leave-One-Subject-Out Cross-Validation.
    
    Args:
        pipeline_class: Pipeline class (e.g., MiniRocketPipeline)
        X: (n_trials, n_channels, n_samples)
        y: (n_trials,) labels
        groups: (n_trials,) subject IDs
        pipeline_kwargs: kwargs for pipeline __init__
        fit_kwargs: kwargs for pipeline.fit()
        n_jobs: parallel jobs (1 for GPU pipelines)
    
    Returns:
        Dict with per-subject scores and aggregate metrics
    """
    if pipeline_kwargs is None:
        pipeline_kwargs = {}
    if fit_kwargs is None:
        fit_kwargs = {}
    
    logo = LeaveOneGroupOut()
    subject_scores = []
    subject_details = []
    
    unique_subjects = np.unique(groups)
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_subject = groups[test_idx][0]
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Create fresh pipeline for each fold
        pipeline = pipeline_class(**pipeline_kwargs)
        pipeline.fit(X_train, y_train, **fit_kwargs)
        
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)
        
        subject_scores.append(acc)
        subject_details.append({
            'fold': fold,
            'subject': int(test_subject),
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'accuracy': float(acc),
            'f1_macro': float(f1),
            'confusion_matrix': cm.tolist(),
        })
        
        print(f"LOSO Fold {fold+1}/{len(unique_subjects)}: Subject {test_subject} - Acc: {acc:.4f}, F1: {f1:.4f}")
    
    return {
        'subject_scores': subject_scores,
        'subject_details': subject_details,
        'mean_accuracy': float(np.mean(subject_scores)),
        'std_accuracy': float(np.std(subject_scores)),
        'median_accuracy': float(np.median(subject_scores)),
        'min_accuracy': float(np.min(subject_scores)),
        'max_accuracy': float(np.max(subject_scores)),
        'bci_illiteracy_rate': float(np.mean(np.array(subject_scores) < 0.7)),  # <70% threshold
        'n_subjects': len(unique_subjects),
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = None
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_per_class = f1_score(y_true, y_pred, average=None)
    cm = confusion_matrix(y_true, y_pred)
    
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    
    return {
        'accuracy': float(acc),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
    }


def evaluate_model_on_subjects(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    labels: List[str] = None
) -> Dict[str, Any]:
    """Evaluate a fitted pipeline per subject."""
    unique_subjects = np.unique(subject_ids)
    results = []
    
    for subj in unique_subjects:
        mask = subject_ids == subj
        X_subj = X[mask]
        y_subj = y[mask]
        
        if len(X_subj) == 0:
            continue
        
        y_pred = pipeline.predict(X_subj)
        metrics = compute_metrics(y_subj, y_pred, labels)
        metrics['subject'] = int(subj)
        metrics['n_trials'] = len(X_subj)
        results.append(metrics)
    
    return {
        'per_subject': results,
        'mean_accuracy': float(np.mean([r['accuracy'] for r in results])),
        'std_accuracy': float(np.std([r['accuracy'] for r in results])),
    }


def save_evaluation_results(results: Dict[str, Any], output_path: str):
    """Save evaluation results to JSON."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation results to {output_path}")


def load_evaluation_results(input_path: str) -> Dict[str, Any]:
    """Load evaluation results from JSON."""
    with open(input_path, 'r') as f:
        return json.load(f)


def print_loso_summary(results: Dict[str, Any]):
    """Print formatted LOSO summary."""
    print("\n" + "="*60)
    print("LOSO CROSS-VALIDATION SUMMARY")
    print("="*60)
    print(f"Subjects evaluated: {results['n_subjects']}")
    print(f"Mean Accuracy:      {results['mean_accuracy']*100:.2f}%")
    print(f"Std Accuracy:       {results['std_accuracy']*100:.2f}%")
    print(f"Median Accuracy:    {results['median_accuracy']*100:.2f}%")
    print(f"Min Accuracy:       {results['min_accuracy']*100:.2f}%")
    print(f"Max Accuracy:       {results['max_accuracy']*100:.2f}%")
    print(f"BCI Illiteracy:     {results['bci_illiteracy_rate']*100:.1f}% (<70%)")
    print("-"*60)
    print("Per-Subject:")
    for detail in results['subject_details']:
        status = "✓" if detail['accuracy'] >= 0.7 else "✗"
        print(f"  Subject {detail['subject']:3d}: {detail['accuracy']*100:5.1f}% {status}")
    print("="*60)


if __name__ == "__main__":
    # Demo with synthetic data
    from scripts.generate_synthetic import generate_synthetic_eeg
    
    print("Generating synthetic data for LOSO demo...")
    X, y, ch_names = generate_synthetic_eeg(
        n_trials=400, n_channels=20, n_samples=656, n_classes=4
    )
    
    # Create subject groups (10 subjects, 40 trials each)
    n_subjects = 10
    groups = np.repeat(np.arange(n_subjects), 40)
    
    # Shuffle together
    idx = np.random.permutation(len(y))
    X, y, groups = X[idx], y[idx], groups[idx]
    
    print(f"Data: X={X.shape}, y={y.shape}, groups={len(np.unique(groups))} subjects")
    
    # Note: Actual pipeline would need torch/cuda
    print("\nSkipping actual LOSO (requires GPU + trained pipeline)")
    print("Use: python -m src.evaluation loso ...")