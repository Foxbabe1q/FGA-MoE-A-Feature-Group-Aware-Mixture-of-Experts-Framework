"""
Evaluate all saved ablation models: load checkpoints, run inference on the
held-out test set, compute metrics with 95% bootstrap CIs, and compare
against the original ablation_results.csv.
"""
import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import builtins
_original_import = builtins.__import__
def custom_import(name, *args, **kwargs):
    if name == 'transformers' or name.startswith('transformers.'):
        raise ImportError(f"Import of {name} is blocked")
    return _original_import(name, *args, **kwargs)
builtins.__import__ = custom_import

import glob
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH, LABEL_COL, FEATURE_GROUPS,
    CATEGORICAL_FEATURES, BINARY_FEATURES, LOG_TRANSFORM_FEATURES,
    TEST_SIZE,
)
from preprocessing import DataPreprocessor
from groupwise_moe_model import FeatureGroupAwareMoEModel
from train_moe import ICUDataset, collate_fn
from evaluation import compute_metrics_with_ci

SEED = 39
MODELS_DIR = 'models'
RESULTS_DIR = 'results'
ORIGINAL_CSV = os.path.join(RESULTS_DIR, 'ablation_results.csv')


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepare_data():
    """Reproduce the exact train/val/test split and preprocessing."""
    df = pd.read_csv(DATA_PATH)
    if LABEL_COL in df.columns and 'Label' not in df.columns:
        df['Label'] = (df[LABEL_COL] == 0).astype(int)

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df['Label']
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.15, random_state=SEED, stratify=train_df['Label']
    )

    preprocessor = DataPreprocessor(
        FEATURE_GROUPS, CATEGORICAL_FEATURES,
        BINARY_FEATURES, LOG_TRANSFORM_FEATURES,
    )
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)

    y_test = test_df[LABEL_COL].values
    group_dims = {g: X_train[g].shape[1] for g in FEATURE_GROUPS}

    return X_test, y_test, group_dims


def load_and_predict(ckpt_path, group_dims, X_test, device):
    """Load a checkpoint, build the model, and return predicted probabilities."""
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint['config']
    config['device'] = str(device)

    model = FeatureGroupAwareMoEModel(group_dims, config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    dummy_y = np.zeros(len(list(X_test.values())[0]))
    dataset = ICUDataset(X_test, dummy_y)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=False, collate_fn=collate_fn
    )

    all_preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = {k: v.to(device) for k, v in X_batch.items()}
            logits = model(X_batch)
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())

    return np.array(all_preds), config['name']


def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Preparing data (same split as training) ...")
    X_test, y_test, group_dims = prepare_data()
    print(f"Test set size: {len(y_test)}")

    ckpt_files = sorted(glob.glob(os.path.join(MODELS_DIR, '*_seed39.pt')))
    print(f"Found {len(ckpt_files)} checkpoint(s)\n")

    rows = []
    for path in ckpt_files:
        fname = os.path.basename(path)
        print(f"  Loading {fname} ...")
        y_pred, config_name = load_and_predict(path, group_dims, X_test, device)
        metrics = compute_metrics_with_ci(y_test, y_pred, n_bootstrap=1000, threshold=0.5)

        rows.append({
            'config': config_name,
            'auroc': metrics['auroc'],
            'auroc_ci': metrics['auroc_ci'],
            'auprc': metrics['auprc'],
            'auprc_ci': metrics['auprc_ci'],
            'f1': metrics['f1'],
            'f1_ci': metrics['f1_ci'],
        })
        print(f"    AUROC={metrics['auroc']:.4f} {metrics['auroc_ci']}  "
              f"AUPRC={metrics['auprc']:.4f} {metrics['auprc_ci']}  "
              f"F1={metrics['f1']:.4f} {metrics['f1_ci']}")

    result_df = pd.DataFrame(rows).sort_values('auroc', ascending=False).reset_index(drop=True)

    # ---- Compare with original CSV ----
    print("\n" + "=" * 80)
    print("COMPARISON WITH ORIGINAL ablation_results.csv")
    print("=" * 80)

    if os.path.exists(ORIGINAL_CSV):
        orig = pd.read_csv(ORIGINAL_CSV)
        merged = result_df.merge(orig, on='config', suffixes=('_new', '_orig'))
        all_match = True
        for _, r in merged.iterrows():
            diff = abs(r['auroc_new'] - r['auroc_orig'])
            status = "OK" if diff < 1e-6 else f"DIFF={diff:.6f}"
            if diff >= 1e-6:
                all_match = False
            print(f"  {r['config']:35s}  orig={r['auroc_orig']:.6f}  new={r['auroc_new']:.6f}  {status}")
        if all_match:
            print("\nAll point estimates match the original results exactly.")
        else:
            print("\nSome point estimates differ — check above.")
    else:
        print(f"  Original CSV not found at {ORIGINAL_CSV}")

    # ---- Save ----
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'ablation_results_with_ci.csv')
    result_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
