"""
Configuration for Feature-Group-Aware MoE Model
Based on Model & Experiment Design.md
"""
import random
import numpy as np
import torch

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Data configuration
DATA_PATH = 'data/eicu_mimiciv_sepsis_cad_final.csv'
LABEL_COL = 'has_survived_28days'
TEST_SIZE = 0.2

# Feature Groups (7 groups total, 46 features)
FEATURE_GROUPS = {
    'demographics': ['age', 'gender', 'race', 'bmi'],
    'comorbidity': [
        'cci', 'has_cardiovascular', 'has_chronic_pulmonary', 'has_liver',
        'has_renal', 'has_diabetes', 'has_marlignancy', 'has_dementia',
        'has_hiv', 'has_hemodialysis'
    ],
    'severity': ['msofa', 'avg_GCS'],
    'inflammation': ['nlr', 'mlr', 'plr'],
    'labs': [
        'avg_sodium', 'avg_potassium', 'avg_calcium', 'avg_wbc',
        'avg_hemoglobin', 'avg_platelets', 'avg_lactate', 'avg_alt',
        'avg_ast', 'avg_bicarbonate', 'avg_aniongap', 'avg_pt',
        'avg_ptt', 'avg_creatinine', 'avg_urineoutput', 'avg_crp',
        'avg_fibrin'
    ],
    'hemodynamics': ['avg_sbp', 'avg_dbp', 'avg_map'],
    'respiratory': [
        'avg_pH', 'avg_paco2', 'avg_pao2', 'avg_peep',
        'avg_tidal_volume_set', 'has_mechanical_ventilation',
        'avg_duration_of_mechanical_ventilation'
    ]
}

# Preprocessing configuration
CATEGORICAL_FEATURES = ['gender', 'race']
BINARY_FEATURES = [
    'has_cardiovascular', 'has_chronic_pulmonary', 'has_liver',
    'has_renal', 'has_diabetes', 'has_marlignancy', 'has_dementia',
    'has_hiv', 'has_mechanical_ventilation', 'has_hemodialysis'
]
LOG_TRANSFORM_FEATURES = ['avg_lactate', 'avg_crp']

# Full FGA-MoE configuration used as the baseline in the ablation study
BASELINE_MOE_CONFIG = {
    'name': 'Full Model (Baseline)',
    'num_experts_per_group': 3,
    'group_encoder_hidden_dim': 16,
    'group_encoder_layers': 1,
    'expert_hidden_dim': 8,
    'top_k': 1,
    'fusion_hidden_dim': 128,
    'dropout': 0.2,
    'use_attention_fusion': False,
    'load_balance_weight': 0.01,
    
    # Training
    'batch_size': 64,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'max_epochs': 100,
    'patience': 15,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Ablation Study Configurations
ABLATION_CONFIGS = {
    # 1. Remove Group Encoder
    'w/o Group Encoder': {
        **BASELINE_MOE_CONFIG,
        'name': 'w/o Group Encoder',
        'group_encoder_layers': 0,  # No group encoder
    },
    
    # 2. Remove MoE (Single Expert per group)
    'w/o MoE (Single Expert)': {
        **BASELINE_MOE_CONFIG,
        'name': 'w/o MoE (Single Expert)',
        'num_experts_per_group': 1,
        'top_k': 1,
    },
    
    # 3. Remove Load Balance Loss
    'w/o Load Balance': {
        **BASELINE_MOE_CONFIG,
        'name': 'w/o Load Balance',
        'load_balance_weight': 0.0,
    },
    
    # 4. Use Attention Fusion
    'w/ Attention Fusion': {
        **BASELINE_MOE_CONFIG,
        'name': 'w/ Attention Fusion',
        'use_attention_fusion': True,
    },
    
    # 5. Different Top-K
    'Top-K=2': {
        **BASELINE_MOE_CONFIG,
        'name': 'Top-K=2',
        'top_k': 2,
    },
    
    # 6. Different Number of Experts
    'Num Experts=2': {
        **BASELINE_MOE_CONFIG,
        'name': 'Num Experts=2',
        'num_experts_per_group': 2,
    },
    
    'Num Experts=5': {
        **BASELINE_MOE_CONFIG,
        'name': 'Num Experts=5',
        'num_experts_per_group': 5,
    },
    
    # 7. Group Encoder Dimension Ablation
    'Larger Group Encoder (32)': {
        **BASELINE_MOE_CONFIG,
        'name': 'Larger Group Encoder (32)',
        'group_encoder_hidden_dim': 32,  # 16 → 32
    },
    
    'Smaller Group Encoder (8)': {
        **BASELINE_MOE_CONFIG,
        'name': 'Smaller Group Encoder (8)',
        'group_encoder_hidden_dim': 8,  # 16 → 8
    },
    
    # 8. Expert Dimension Ablation
    'Larger Expert (16)': {
        **BASELINE_MOE_CONFIG,
        'name': 'Larger Expert (16)',
        'expert_hidden_dim': 16,  # 8 → 16
    },
    
    'Smaller Expert (4)': {
        **BASELINE_MOE_CONFIG,
        'name': 'Smaller Expert (4)',
        'expert_hidden_dim': 4,  # 8 → 4
    },
    
    # 9. Fusion Dimension Ablation
    'Larger Fusion (256)': {
        **BASELINE_MOE_CONFIG,
        'name': 'Larger Fusion (256)',
        'fusion_hidden_dim': 256,  # 128 → 256
    },
    
    'Smaller Fusion (64)': {
        **BASELINE_MOE_CONFIG,
        'name': 'Smaller Fusion (64)',
        'fusion_hidden_dim': 64,  # 128 → 64
    },
}

# Evaluation configuration
EVAL_METRICS = [
    'auroc', 'auprc', 'accuracy', 'precision', 'recall',
    'f1', 'specificity', 'brier_score'
]
BOOTSTRAP_SAMPLES = 1000
