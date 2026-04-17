"""
Feature-Group-Aware Mixture of Experts Model
Each feature group has its own encoder and MoE module
"""
import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupEncoder(nn.Module):
    """Encoder for a single feature group"""
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        
        if num_layers == 0:
            # No encoding (for ablation study: w/o Group Encoder)
            # Just project to hidden_dim for compatibility
            self.encoder = nn.Linear(input_dim, hidden_dim)
        else:
            layers = []
            
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            
            self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)

class ExpertNetwork(nn.Module):
    """Single expert network"""
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class GroupWiseRouter(nn.Module):
    """Router for selecting experts within a group"""
    def __init__(self, input_dim, num_experts, top_k=1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        # x: [batch, input_dim]
        gate_logits = self.gate(x)  # [batch, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Select top-k experts
        topk_probs, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        
        # Normalize selected probabilities
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Create sparse gate mask
        gate_mask = torch.zeros_like(gate_probs)
        gate_mask.scatter_(1, topk_indices, topk_probs)
        
        return gate_probs, gate_mask

class GroupWiseMoE(nn.Module):
    """MoE for a single feature group"""
    def __init__(self, input_dim, hidden_dim, num_experts, top_k=1, dropout=0.2):
        super().__init__()
        self.num_experts = num_experts
        
        # Router
        self.router = GroupWiseRouter(input_dim, num_experts, top_k)
        
        # Expert pool
        self.experts = nn.ModuleList([
            ExpertNetwork(input_dim, hidden_dim, dropout)
            for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # x: [batch, input_dim]
        batch_size = x.size(0)
        
        # Routing
        gate_probs, gate_mask = self.router(x)
        
        # Apply experts
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, hidden_dim]
        
        # Weighted combination
        gate_mask_expanded = gate_mask.unsqueeze(-1)  # [batch, num_experts, 1]
        output = (expert_outputs * gate_mask_expanded).sum(dim=1)  # [batch, hidden_dim]
        
        return output, gate_probs, gate_mask

class FeatureGroupAwareMoEModel(nn.Module):
    """
    Main model: Each feature group has its own encoder and MoE
    All group outputs are fused for final prediction
    """
    def __init__(self, group_input_dims, config):
        super().__init__()
        self.group_names = list(group_input_dims.keys())
        
        # Extract config
        hidden_dim = config['group_encoder_hidden_dim']
        expert_dim = config['expert_hidden_dim']
        num_experts = config['num_experts_per_group']
        top_k = config['top_k']
        dropout = config['dropout']
        fusion_dim = config['fusion_hidden_dim']
        num_layers = config['group_encoder_layers']
        use_attention = config['use_attention_fusion']
        
        # Group encoders
        self.group_encoders = nn.ModuleDict({
            name: GroupEncoder(group_input_dims[name], hidden_dim, num_layers, dropout)
            for name in self.group_names
        })
        
        # Group-wise MoE
        self.group_moes = nn.ModuleDict({
            name: GroupWiseMoE(hidden_dim, expert_dim, num_experts, top_k, dropout)
            for name in self.group_names
        })
        
        # Fusion module
        total_dim = expert_dim * len(self.group_names)
        
        if use_attention:
            # Attention-based fusion
            self.fusion = nn.Sequential(
                nn.Linear(expert_dim, expert_dim),
                nn.Tanh()
            )
            self.attention = nn.Linear(expert_dim, 1)
            self.classifier = nn.Linear(expert_dim, 1)
        else:
            # MLP-based fusion (default)
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, fusion_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.classifier = nn.Linear(fusion_dim // 2, 1)
        
        self.use_attention_fusion = use_attention
        self.gate_probs_dict = {}
        self.gate_masks_dict = {}
    
    def forward(self, X_dict):
        """
        X_dict: dict of {group_name: tensor[batch, group_features]}
        """
        group_outputs = []
        
        # Process each group
        for group_name in self.group_names:
            # Encode group
            h_group = self.group_encoders[group_name](X_dict[group_name])
            
            # Apply MoE
            z_group, gate_probs, gate_mask = self.group_moes[group_name](h_group)
            
            # Store routing info for analysis
            self.gate_probs_dict[group_name] = gate_probs.detach()
            self.gate_masks_dict[group_name] = gate_mask.detach()
            
            group_outputs.append(z_group)
        
        # Fusion
        if self.use_attention_fusion:
            # Attention-based fusion
            group_outputs_stacked = torch.stack(group_outputs, dim=1)  # [batch, num_groups, expert_dim]
            attended = self.fusion(group_outputs_stacked)  # [batch, num_groups, expert_dim]
            attn_weights = F.softmax(self.attention(attended), dim=1)  # [batch, num_groups, 1]
            fused = (group_outputs_stacked * attn_weights).sum(dim=1)  # [batch, expert_dim]
        else:
            # Concatenation-based fusion
            fused = torch.cat(group_outputs, dim=-1)  # [batch, total_dim]
            fused = self.fusion(fused)
        
        # Classification
        logits = self.classifier(fused).squeeze(-1)
        
        return logits
    
    def get_routing_info(self):
        """Return routing information for all groups"""
        return {
            'gate_probs': self.gate_probs_dict,
            'gate_masks': self.gate_masks_dict
        }
    
    def compute_load_balance_loss(self):
        """Compute load balancing loss across all groups"""
        total_loss = 0
        num_groups = len(self.group_names)
        
        for group_name in self.group_names:
            gate_probs = self.gate_probs_dict[group_name]
            # Encourage uniform distribution
            avg_probs = gate_probs.mean(dim=0)
            num_experts = avg_probs.size(0)
            target = torch.ones_like(avg_probs) / num_experts
            total_loss += F.mse_loss(avg_probs, target)
        
        return total_loss / num_groups
    
    def get_load_balance_stats(self):
        """Get detailed statistics for analysis"""
        stats = {}
        
        for group_name in self.group_names:
            gate_probs = self.gate_probs_dict[group_name]
            gate_mask = self.gate_masks_dict[group_name]
            
            # Expert usage statistics
            expert_usage = gate_mask.sum(dim=0)  # How many samples each expert handled
            expert_avg_prob = gate_probs.mean(dim=0)  # Average probability
            
            stats[group_name] = {
                'expert_usage': expert_usage.cpu().numpy(),
                'expert_avg_prob': expert_avg_prob.cpu().numpy(),
                'max_usage': expert_usage.max().item(),
                'min_usage': expert_usage.min().item(),
                'usage_std': expert_usage.std().item()
            }
        
        return stats
