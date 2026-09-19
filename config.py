"""
Configuration settings for CodeLACE model.
"""

class CodeLACEConfig:
    def __init__(
        self,
        vocab_size=1000,
        hidden_size=768,
        num_layers=6,
        num_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        num_experts=8,
        sparsity_ratio=0.1,
        num_syntactic_classes=10,
        num_semantic_classes=8,
        num_pragmatic_classes=6,
        dropout_prob=0.1,
        layer_norm_eps=1e-12,
        # Ablation flags — all True = full CodeLACE
        use_sparse_attention=True,
        use_token_pooling=True,
        use_moe=True
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_experts = num_experts
        self.sparsity_ratio = sparsity_ratio
        self.num_syntactic_classes = num_syntactic_classes
        self.num_semantic_classes = num_semantic_classes
        self.num_pragmatic_classes = num_pragmatic_classes
        self.dropout_prob = dropout_prob
        self.layer_norm_eps = layer_norm_eps

# Ablation flags
        self.use_sparse_attention = use_sparse_attention
        self.use_token_pooling = use_token_pooling
        self.use_moe = use_moe

def create_codelace_config():
    """Create default CodeLACE configuration."""
    return CodeLACEConfig()

def create_lightweight_config():
    """Create lightweight configuration for testing."""
    return CodeLACEConfig(
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        intermediate_size=1024,
        max_position_embeddings=256,
        num_experts=4
    )

def create_ablation_configs():
    base = dict(
        vocab_size=1000,
        hidden_size=768,
        num_layers=6,
        num_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        num_experts=8,
        sparsity_ratio=0.1,
        num_syntactic_classes=10,
        num_semantic_classes=8,
        num_pragmatic_classes=6,
        dropout_prob=0.1,
        layer_norm_eps=1e-12
    )
    return {
        'full_codelace': CodeLACEConfig(
            **base,
            use_sparse_attention=True,
            use_token_pooling=True,
            use_moe=True
        ),
        'no_moe': CodeLACEConfig(
            **base,
            use_sparse_attention=True,
            use_token_pooling=True,
            use_moe=False
        ),
        'no_token_pooling': CodeLACEConfig(
            **base,
            use_sparse_attention=True,
            use_token_pooling=False,
            use_moe=True
        ),
        'no_sparse_attention': CodeLACEConfig(
            **base,
            use_sparse_attention=False,
            use_token_pooling=True,
            use_moe=True
        ),
        'baseline': CodeLACEConfig(
            **base,
            use_sparse_attention=False,
            use_token_pooling=False,
            use_moe=False
        )
    }
