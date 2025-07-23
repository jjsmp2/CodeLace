

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from config import CodeLACEConfig

class SparseAttention(nn.Module):
    
    def __init__(self, config: CodeLACEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.sparsity_ratio = config.sparsity_ratio
        
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout_prob)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length, hidden_size = hidden_states.shape
        
        # Compute Q, K, V
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply sparse attention mask - FIXED VERSION
        sparse_mask = self._create_sparse_mask(seq_length, attention_scores.device, batch_size)
        attention_scores = attention_scores * sparse_mask
        
        # Apply padding mask if provided - FIXED VERSION
        if attention_mask is not None:
            # Ensure attention_mask has correct dimensions
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            
            # Expand to match attention_scores dimensions
            attention_mask = attention_mask.expand(batch_size, self.num_heads, seq_length, seq_length)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, value)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)
        
        return context
    
    def _create_sparse_mask(self, seq_length: int, device: torch.device, batch_size: int) -> torch.Tensor:
        """Create sparse attention mask."""
        mask = torch.zeros(seq_length, seq_length, device=device)
        
        # Always attend to self
        mask.fill_diagonal_(1)
        
        # Add local attention (attend to neighbors) - SAFE VERSION
        for i in range(seq_length):
            start = max(0, i - 2)
            end = min(seq_length, i + 3)
            mask[i, start:end] = 1
        
        # Add random sparse connections - SAFE VERSION
        num_random = max(1, int(seq_length * self.sparsity_ratio))
        for i in range(seq_length):
            if seq_length > 1:
                indices = torch.randperm(seq_length, device=device)[:num_random]
                mask[i, indices] = 1
        
        # Expand for batch and head dimensions
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        mask = mask.expand(batch_size, self.num_heads, seq_length, seq_length)
        
        return mask

class TokenPooling(nn.Module):
    """hierarchical token pooling for sequence reduction."""
    
    def __init__(self, config: CodeLACEConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.pooling_ratio = 0.5  # Reduce sequence length by half
        
        self.pooling_layer = nn.Linear(config.hidden_size * 2, config.hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, hidden_size = hidden_states.shape
        
        # Handle edge case for very short sequences
        if seq_length < 2:
            return hidden_states
        
        # Simple pooling: combine adjacent tokens
        if seq_length % 2 == 1:
            # Pad if odd length
            padding = torch.zeros(batch_size, 1, hidden_size, device=hidden_states.device)
            hidden_states = torch.cat([hidden_states, padding], dim=1)
            seq_length += 1
        
        # Reshape and pool
        pooled_length = seq_length // 2
        reshaped = hidden_states.view(batch_size, pooled_length, 2, hidden_size)
        concatenated = reshaped.view(batch_size, pooled_length, 2 * hidden_size)
        
        # Apply linear transformation
        pooled = self.pooling_layer(concatenated)
        
        return pooled

class MixtureOfExperts(nn.Module):
    """mixture of experts for specialized processing."""
    
    def __init__(self, config: CodeLACEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_size = config.hidden_size
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                nn.ReLU(),
                nn.Linear(config.intermediate_size, config.hidden_size)
            ) for _ in range(config.num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(config.hidden_size, config.num_experts)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, hidden_size = hidden_states.shape
        
        # Compute gating weights
        gate_logits = self.gate(hidden_states)  # [batch, seq, num_experts]
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        # Apply experts
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(hidden_states)  # [batch, seq, hidden]
            expert_outputs.append(expert_output)
        
        # Stack expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # [batch, seq, hidden, num_experts]
        
        # Weighted combination
        gate_weights = gate_weights.unsqueeze(2)  # [batch, seq, 1, num_experts]
        output = torch.sum(expert_outputs * gate_weights, dim=-1)  # [batch, seq, hidden]
        
        return output

class CodeLACELayer(nn.Module):
    """single CodeLACE transformer layer."""
    
    def __init__(self, config: CodeLACEConfig):
        super().__init__()
        self.attention = SparseAttention(config)
        self.moe = MixtureOfExperts(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_prob)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.layer_norm1(hidden_states + self.dropout(attention_output))
        
        # MoE with residual connection
        moe_output = self.moe(hidden_states)
        hidden_states = self.layer_norm2(hidden_states + self.dropout(moe_output))
        
        return hidden_states

class CodeLACE(nn.Module):
    """main CodeLACE model for semantic source code analysis."""
    
    def __init__(self, config: CodeLACEConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_prob)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            CodeLACELayer(config) for _ in range(config.num_layers)
        ])
        
        # Token pooling (optional)
        self.token_pooling = TokenPooling(config)
        
        # Classification heads
        self.syntactic_classifier = nn.Linear(config.hidden_size, config.num_syntactic_classes)
        self.semantic_classifier = nn.Linear(config.hidden_size, config.num_semantic_classes)
        self.pragmatic_classifier = nn.Linear(config.hidden_size, config.num_pragmatic_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_length = input_ids.shape
        
        # Ensure sequence length doesn't exceed max_position_embeddings
        if seq_length > self.config.max_position_embeddings:
            input_ids = input_ids[:, :self.config.max_position_embeddings]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.config.max_position_embeddings]
            seq_length = self.config.max_position_embeddings
        
        # Create position IDs
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = token_embeds + position_embeds
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Global pooling for classification
        if attention_mask is not None:
            # Mask out padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = hidden_states.mean(dim=1)
        
        # Classification
        syntactic_logits = self.syntactic_classifier(pooled_output)
        semantic_logits = self.semantic_classifier(pooled_output)
        pragmatic_logits = self.pragmatic_classifier(pooled_output)
        
        return syntactic_logits, semantic_logits, pragmatic_logits
    
    def analyze(self, code: str, language: str = 'python') -> dict:
        """High-level interface for code analysis."""
        from tokenizer import CodeTokenizer
        
        tokenizer = CodeTokenizer()
        
        # Tokenize and encode
        input_ids = tokenizer.encode(code, language).unsqueeze(0)  # Add batch dimension
        attention_mask = tokenizer.create_attention_mask(input_ids)
        
        # Forward pass
        self.eval()
        with torch.no_grad():
            syntactic_logits, semantic_logits, pragmatic_logits = self.forward(input_ids, attention_mask)
            
            # Get predictions
            syntactic_pred = torch.argmax(syntactic_logits, dim=-1).item()
            semantic_pred = torch.argmax(semantic_logits, dim=-1).item()
            pragmatic_pred = torch.argmax(pragmatic_logits, dim=-1).item()
        
        return {
            'syntactic': syntactic_pred,
            'semantic': semantic_pred,
            'pragmatic': pragmatic_pred,
            'syntactic_confidence': torch.softmax(syntactic_logits, dim=-1).max().item(),
            'semantic_confidence': torch.softmax(semantic_logits, dim=-1).max().item(),
            'pragmatic_confidence': torch.softmax(pragmatic_logits, dim=-1).max().item()
        }

# Test the fixed model
if __name__ == "__main__":
    from config import create_lightweight_config
    
    # Create model with lightweight config for testing
    config = create_lightweight_config()
    model = CodeLACE(config)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size, seq_length = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    try:
        syntactic_logits, semantic_logits, pragmatic_logits = model(input_ids, attention_mask)
        
        print(f"✅ Forward pass successful!")
        print(f"Syntactic logits shape: {syntactic_logits.shape}")
        print(f"Semantic logits shape: {semantic_logits.shape}")
        print(f"Pragmatic logits shape: {pragmatic_logits.shape}")
        
        # Test analysis interface
        code = "def hello(): print('Hello, World!')"
        results = model.analyze(code)
        print(f"✅ Analysis results: {results}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

