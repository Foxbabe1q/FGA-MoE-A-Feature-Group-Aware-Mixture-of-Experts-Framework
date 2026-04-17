"""
Training utilities for Group-wise MoE model
"""
import os
# Disable torch compile to avoid library conflicts
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ICUDataset(Dataset):
    """Dataset for grouped features"""
    def __init__(self, X_grouped, y):
        self.X_grouped = X_grouped
        self.y = torch.FloatTensor(y)
        self.group_names = list(X_grouped.keys())
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        X_dict = {name: torch.FloatTensor(self.X_grouped[name][idx]) 
                  for name in self.group_names}
        return X_dict, self.y[idx]

def collate_fn(batch):
    """Custom collate function for batching grouped features"""
    X_batch = {}
    y_batch = []
    
    for X_dict, y in batch:
        for name, features in X_dict.items():
            if name not in X_batch:
                X_batch[name] = []
            X_batch[name].append(features)
        y_batch.append(y)
    
    # Stack features for each group
    X_batch = {name: torch.stack(tensors) for name, tensors in X_batch.items()}
    y_batch = torch.stack(y_batch)
    
    return X_batch, y_batch


class MoETrainer:
    """Trainer for MoE model"""
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config['device']
        
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Loss
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_bce_loss = 0
        total_load_loss = 0
        
        for X_batch, y_batch in train_loader:
            # Move to device
            X_batch = {k: v.to(self.device) for k, v in X_batch.items()}
            y_batch = y_batch.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            
            # BCE loss
            bce_loss = self.criterion(logits, y_batch)
            
            # Load balance loss
            load_loss = self.model.compute_load_balance_loss()
            
            # Total loss
            loss = bce_loss + self.config['load_balance_weight'] * load_loss
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_bce_loss += bce_loss.item()
            total_load_loss += load_loss.item()
        
        n_batches = len(train_loader)
        return {
            'total_loss': total_loss / n_batches,
            'bce_loss': total_bce_loss / n_batches,
            'load_loss': total_load_loss / n_batches
        }
    
    def evaluate(self, val_loader):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = {k: v.to(self.device) for k, v in X_batch.items()}
                y_batch = y_batch.to(self.device)
                
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                
                total_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        # Compute AUROC
        from sklearn.metrics import roc_auc_score
        try:
            auroc = roc_auc_score(all_labels, all_preds)
        except:
            auroc = 0.5
        
        return {
            'loss': total_loss / len(val_loader),
            'auroc': auroc
        }
    
    def fit(self, train_loader, val_loader, verbose=True):
        """Train the model"""
        if verbose:
            if self.config.get('use_class_weight', False):
                print(f"  Using weighted BCE with pos_weight={self.config.get('pos_weight', 3.0)}")
        
        for epoch in range(self.config['max_epochs']):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            self.scheduler.step()
            
            # Early stopping
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                self.patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config['max_epochs']} - "
                      f"Train Loss: {train_metrics['total_loss']:.4f} "
                      f"(BCE: {train_metrics['bce_loss']:.4f}, Load: {train_metrics['load_loss']:.4f}) - "
                      f"Val Loss: {val_metrics['loss']:.4f} - "
                      f"Val AUC: {val_metrics['auroc']:.4f}")
            
            if self.patience_counter >= self.config['patience']:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        self.model.load_state_dict(self.best_state)
        
        if verbose:
            print(f"\nTraining complete! Best Val AUC: {val_metrics['auroc']:.4f}")
    
    def predict_proba(self, test_loader):
        """Predict probabilities"""
        self.model.eval()
        all_preds = []
        
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = {k: v.to(self.device) for k, v in X_batch.items()}
                logits = self.model(X_batch)
                probs = torch.sigmoid(logits)
                all_preds.extend(probs.cpu().numpy())
        
        return np.array(all_preds)
