"""
ICU 28-Day Mortality Prediction using Feature-Group-Aware MoE Model
Ablation Study: Comprehensive evaluation of model components
"""
import os
import sys

# Block transformers import
import builtins
_original_import = builtins.__import__

def custom_import(name, *args, **kwargs):
    if name == 'transformers' or name.startswith('transformers.'):
        raise ImportError(f"Import of {name} is blocked to avoid version conflicts")
    return _original_import(name, *args, **kwargs)

builtins.__import__ = custom_import

os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK'] = '0'

import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH, LABEL_COL, FEATURE_GROUPS, CATEGORICAL_FEATURES,
    BINARY_FEATURES, LOG_TRANSFORM_FEATURES, TEST_SIZE,
    BASELINE_MOE_CONFIG, ABLATION_CONFIGS
)
from preprocessing import DataPreprocessor
from groupwise_moe_model import FeatureGroupAwareMoEModel
from train_moe import ICUDataset, MoETrainer, collate_fn
from evaluation import compute_metrics_with_ci

SEED = 39

def set_seed(seed):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_and_evaluate(config, X_train, y_train, X_val, y_val, X_test, y_test, group_dims):
    """Train and evaluate a single config"""
    # Create model
    model = FeatureGroupAwareMoEModel(group_dims, config)
    
    # Create datasets
    train_dataset = ICUDataset(X_train, y_train)
    val_dataset = ICUDataset(X_val, y_val)
    test_dataset = ICUDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['batch_size'],
        shuffle=False, collate_fn=collate_fn
    )
    
    # Train
    trainer = MoETrainer(model, config)
    trainer.fit(train_loader, val_loader, verbose=False)
    
    # Test
    y_pred_proba = trainer.predict_proba(test_loader)
    metrics = compute_metrics_with_ci(y_test, y_pred_proba, n_bootstrap=1000)
    
    # Print results
    print(f"  AUROC: {metrics['auroc']:.4f} {metrics['auroc_ci']}")
    print(f"  AUPRC: {metrics['auprc']:.4f} {metrics['auprc_ci']}")
    print(f"  F1: {metrics['f1']:.4f} {metrics['f1_ci']}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    # Create safe filename from config name
    safe_name = config['name'].replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
    model_path = f'models/{safe_name}_seed{SEED}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': {
            'auroc': metrics['auroc'],
            'auprc': metrics['auprc'],
            'f1': metrics['f1']
        },
        'seed': SEED
    }, model_path)
    print(f"  Model saved: {model_path}")
    
    return {
        'config': config['name'],
        'auroc': metrics['auroc'],
        'auprc': metrics['auprc'],
        'f1': metrics['f1'],
        'seed': SEED
    }

def main():
    print("="*80)
    print("ICU 28-Day Mortality Prediction")
    print("Feature-Group-Aware Mixture of Experts - ABLATION STUDY")
    print("="*80)
    print(f"Seed: {SEED}")
    print(f"Total configurations: {1 + len(ABLATION_CONFIGS)}")
    print()
    
    # Set seed for reproducibility
    set_seed(SEED)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    if LABEL_COL in df.columns and 'Label' not in df.columns:
        df['Label'] = (df[LABEL_COL] == 0).astype(int)
    
    print(f"Total samples: {len(df)}")
    print(f"Positive rate: {df[LABEL_COL].mean():.2%}")
    
    # Split data
    print(f"\nSplitting data (seed={SEED})...")
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED,
        stratify=df['Label']
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.15, random_state=SEED,
        stratify=train_df['Label']
    )
    
    print(f"Train size: {len(train_df)}")
    print(f"Val size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
    # Preprocess
    print("\nPreprocessing...")
    preprocessor = DataPreprocessor(
        FEATURE_GROUPS, CATEGORICAL_FEATURES,
        BINARY_FEATURES, LOG_TRANSFORM_FEATURES
    )
    X_train = preprocessor.fit_transform(train_df)
    X_val = preprocessor.transform(val_df)
    X_test = preprocessor.transform(test_df)
    
    y_train = train_df[LABEL_COL].values
    y_val = val_df[LABEL_COL].values
    y_test = test_df[LABEL_COL].values
    
    group_dims = {group: X_train[group].shape[1] for group in FEATURE_GROUPS.keys()}
    
    print("\nFeature groups:")
    for name, dim in group_dims.items():
        print(f"  {name}: {dim} features")
    
    # Test all configs
    print("\n" + "="*80)
    print("STARTING ABLATION STUDY")
    print("="*80)
    
    config_results = []
    
    # Baseline
    print(f"\n[1/14] Testing Baseline...")
    baseline_config = BASELINE_MOE_CONFIG.copy()
    baseline_config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    result = train_and_evaluate(baseline_config, X_train, y_train, X_val, y_val, X_test, y_test, group_dims)
    config_results.append(result)
    baseline_auroc = result['auroc']
    
    # Ablations
    for idx, (name, ablation_config) in enumerate(ABLATION_CONFIGS.items(), 2):
        print(f"\n[{idx}/14] Testing {name}...")
        config = ablation_config.copy()
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        result = train_and_evaluate(config, X_train, y_train, X_val, y_val, X_test, y_test, group_dims)
        config_results.append(result)
    
    # Sort by AUROC
    config_df = pd.DataFrame(config_results)
    config_df = config_df.sort_values('auroc', ascending=False).reset_index(drop=True)
    
    # Check Baseline rank
    baseline_rank = config_df[config_df['config'] == 'Full Model (Baseline)'].index[0] + 1
    
    # Save results
    os.makedirs('results', exist_ok=True)
    config_df.to_csv('results/ablation_results.csv', index=False)
    
    # Summary
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print("="*80)
    print(f"\nBaseline Performance:")
    print(f"  AUROC: {baseline_auroc:.4f}")
    print(f"  Rank: {baseline_rank}/14")
    
    print(f"\nTop 5 Configurations (by AUROC):")
    for idx, row in config_df.head(5).iterrows():
        print(f"  {idx+1}. {row['config']}: AUROC {row['auroc']:.4f}")
    
    print(f"\nResults saved to:")
    print(f"  - results/ablation_results.csv")
    print(f"  - models/ (14 model weight files)")
    print("="*80)

if __name__ == '__main__':
    main()
