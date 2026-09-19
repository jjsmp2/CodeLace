import os
import json
import time
import torch
import numpy as np

# Import from your existing files
from config import CodeLACEConfig
from model import CodeLACE
from trainer import CodeLACETrainer      # use your actual class name
from evaluation import evaluate_model    # use your actual function name

# -----------------------------------------------------------
# Define the five configurations
# Each one turns components on or off via config flags
# -----------------------------------------------------------

def get_ablation_configs():
    
    # Base settings shared across all five configurations
    # These must match exactly what you used for your
    # original full CodeLACE training run
    base = dict(
        vocab_size                = 1000,
        hidden_size               = 768,
        num_layers                = 6,
        num_heads                 = 12,
        intermediate_size         = 3072,
        max_position_embeddings   = 512,
        num_experts               = 8,
        sparsity_ratio            = 0.1,
        num_syntactic_classes     = 10,
        num_semantic_classes      = 8,
        num_pragmatic_classes     = 6,
        dropout_prob              = 0.1,
        layer_norm_eps            = 1e-12
    )
    
    return {
        
        # Config 1: everything on
        'full_codelace': CodeLACEConfig(
            **base,
            use_sparse_attention = True,
            use_token_pooling    = True,
            use_moe              = True
        ),
        
        # Config 2: remove MoE only
        'no_moe': CodeLACEConfig(
            **base,
            use_sparse_attention = True,
            use_token_pooling    = True,
            use_moe              = False
        ),
        
        # Config 3: remove token pooling only
        'no_token_pooling': CodeLACEConfig(
            **base,
            use_sparse_attention = True,
            use_token_pooling    = False,
            use_moe              = True
        ),
        
        # Config 4: remove sparse attention only
        'no_sparse_attention': CodeLACEConfig(
            **base,
            use_sparse_attention = False,
            use_token_pooling    = True,
            use_moe              = True
        ),
        
        # Config 5: nothing on — this is your baseline
        'baseline': CodeLACEConfig(
            **base,
            use_sparse_attention = False,
            use_token_pooling    = False,
            use_moe              = False
        )
    }


# -----------------------------------------------------------
# Main ablation runner
# -----------------------------------------------------------

def run_ablation():
    
    configs = get_ablation_configs()
    results = {}
    os.makedirs('checkpoints', exist_ok=True)
    
    run_order = [
        'full_codelace',
        'no_moe',
        'no_token_pooling',
        'no_sparse_attention',
        'baseline'
    ]
    
    for name in run_order:
        config = configs[name]
        
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        print(f"  sparse_attention : {config.use_sparse_attention}")
        print(f"  token_pooling    : {config.use_token_pooling}")
        print(f"  moe              : {config.use_moe}")
        print(f"{'='*50}")
        
        # Fix random seed for every configuration
        # This ensures fair comparison across all runs
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Build model from this configuration
        model  = CodeLACE(config)
        params = sum(
            p.numel() for p in model.parameters() 
            if p.requires_grad
        )
        print(f"  Parameters: {params:,}")
        
        # Wire your existing trainer
        # Replace CodeLACETrainer with whatever
        # your trainer class is actually called
        trainer = CodeLACETrainer(model, config)
        
        # Time the training
        t_start = time.time()
        trainer.train()   # pass whatever args your train() needs
        t_mins  = round((time.time() - t_start) / 60, 1)
        
        # Measure inference time over 100 forward passes
        model.eval()
        dummy_input = torch.randint(
            0, config.vocab_size, (1, 128)
        )
        if torch.cuda.is_available():
            model       = model.cuda()
            dummy_input = dummy_input.cuda()
        
        times = []
        with torch.no_grad():
            for _ in range(100):
                t0 = time.time()
                model(dummy_input)
                times.append((time.time() - t0) * 1000)
        
        avg_ms = round(np.mean(times), 2)
        
        # Evaluate using your existing evaluation function
        # Replace evaluate_model with whatever your
        # evaluation function is actually called
        metrics = evaluate_model(model, config)
        
        # Store results
        results[name] = {
            'overall_accuracy'   : metrics['overall_accuracy'],
            'syntactic_accuracy' : metrics['syntactic_accuracy'],
            'semantic_accuracy'  : metrics['semantic_accuracy'],
            'pragmatic_accuracy' : metrics['pragmatic_accuracy'],
            'macro_f1'           : metrics['macro_f1'],
            'parameters'         : params,
            'inference_ms'       : avg_ms,
            'train_minutes'      : t_mins
        }
        
        # Save checkpoint for this configuration
        torch.save(
            model.state_dict(),
            f"checkpoints/ablation_{name}.pt"
        )
        
        print(f"  Accuracy : {metrics['overall_accuracy']:.1f}%")
        print(f"  F1       : {metrics['macro_f1']:.1f}%")
        print(f"  Time     : {t_mins} mins")
    
    # Save all results to JSON
    with open('ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print final comparison table
    print_table(results, run_order)
    return results


def print_table(results, order):
    
    header = (
        f"\n{'Config':<22} "
        f"{'Overall':>8} "
        f"{'Syntactic':>10} "
        f"{'Semantic':>9} "
        f"{'Pragmatic':>10} "
        f"{'F1':>7} "
        f"{'Params':>10} "
        f"{'ms':>7}"
    )
    
    print("\n" + "="*85)
    print(header)
    print("="*85)
    
    for name in order:
        r = results[name]
        print(
            f"{name:<22} "
            f"{r['overall_accuracy']:>7.1f}% "
            f"{r['syntactic_accuracy']:>9.1f}% "
            f"{r['semantic_accuracy']:>8.1f}% "
            f"{r['pragmatic_accuracy']:>9.1f}% "
            f"{r['macro_f1']:>6.1f}% "
            f"{r['parameters']:>10,} "
            f"{r['inference_ms']:>6.1f}"
        )
    
    print("="*85)


if __name__ == '__main__':
    run_ablation()
