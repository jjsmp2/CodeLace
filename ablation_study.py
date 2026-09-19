"""
Ablation study for CodeLACE.
Tests five configurations by removing one component at a time.
"""

import os
import json
import time
import torch
import numpy as np

from config import CodeLACEConfig
from model import CodeLACE
from trainer import CodeLACETrainer, HierarchicalLoss
from evaluation import ModelEvaluator
from data.sample_data import create_data_loaders
from utils import set_seed


# -----------------------------------------------------------
# Step 1 — Define five configurations
# -----------------------------------------------------------

def get_ablation_configs():

    base = dict(
        vocab_size              = 1000,
        hidden_size             = 256,
        num_layers              = 4,
        num_heads               = 8,
        intermediate_size       = 1024,
        max_position_embeddings = 256,
        num_experts             = 4,
        sparsity_ratio          = 0.1,
        num_syntactic_classes   = 10,
        num_semantic_classes    = 8,
        num_pragmatic_classes   = 6,
        dropout_prob            = 0.1,
        layer_norm_eps          = 1e-12
    )

    return {
        'full_codelace': CodeLACEConfig(
            **base,
            use_sparse_attention = True,
            use_token_pooling    = True,
            use_moe              = True
        ),
        'no_moe': CodeLACEConfig(
            **base,
            use_sparse_attention = True,
            use_token_pooling    = True,
            use_moe              = False
        ),
        'no_token_pooling': CodeLACEConfig(
            **base,
            use_sparse_attention = True,
            use_token_pooling    = False,
            use_moe              = True
        ),
        'no_sparse_attention': CodeLACEConfig(
            **base,
            use_sparse_attention = False,
            use_token_pooling    = True,
            use_moe              = True
        ),
        'baseline': CodeLACEConfig(
            **base,
            use_sparse_attention = False,
            use_token_pooling    = False,
            use_moe              = False
        )
    }


# -----------------------------------------------------------
# Step 2 — Run all five configurations
# -----------------------------------------------------------

def run_ablation():

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    configs   = get_ablation_configs()
    evaluator = ModelEvaluator(results_dir='results')
    results   = {}

    run_order = [
        'full_codelace',
        'no_moe',
        'no_token_pooling',
        'no_sparse_attention',
        'baseline'
    ]

    # Create data loaders once
    # Same loaders used for every configuration
    # This is critical for fair comparison
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        train_size = 800,
        val_size   = 200,
        batch_size = 16
    )

    for name in run_order:

        config = configs[name]

        print(f"\n{'='*55}")
        print(f"Configuration: {name}")
        print(f"  sparse_attention : {config.use_sparse_attention}")
        print(f"  token_pooling    : {config.use_token_pooling}")
        print(f"  moe              : {config.use_moe}")
        print(f"{'='*55}")

        # Fix seed before every run
        set_seed(42)

        # Build model
        model  = CodeLACE(config)
        params = sum(
            p.numel() for p in model.parameters()
            if p.requires_grad
        )
        print(f"  Parameters: {params:,}")

        # Wire your existing trainer directly
        trainer = CodeLACETrainer(
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            learning_rate = 2e-5,
            num_epochs   = 5
        )

        # Override checkpoint path so each config
        # saves its own checkpoint
        trainer.best_model_path = (
            f'checkpoints/ablation_{name}_best.pt'
        )

        # Train
        t_start = time.time()
        trainer.train()
        t_mins  = round((time.time() - t_start) / 60, 1)

        # Evaluate using your existing evaluator
        metrics = evaluator.evaluate_model(
            model       = model,
            data_loader = val_loader,
            model_name  = name
        )

        # Measure inference time separately
        # Average over 100 forward passes
        model.eval()
        sample_batch = next(iter(val_loader))
        infer_times  = []

        with torch.no_grad():
            for _ in range(100):
                t0 = time.time()
                model(
                    sample_batch['input_ids'],
                    sample_batch['attention_mask']
                )
                infer_times.append(
                    (time.time() - t0) * 1000
                )

        avg_ms = round(np.mean(infer_times), 2)

        # Store everything
        results[name] = {
            'overall_accuracy'   : round(
                metrics['overall_accuracy'] * 100, 1
            ),
            'syntactic_accuracy' : round(
                metrics['syntactic_accuracy'] * 100, 1
            ),
            'semantic_accuracy'  : round(
                metrics['semantic_accuracy'] * 100, 1
            ),
            'pragmatic_accuracy' : round(
                metrics['pragmatic_accuracy'] * 100, 1
            ),
            'macro_f1'           : round(
                metrics['overall_f1'] * 100, 1
            ),
            'parameters'         : params,
            'inference_ms'       : avg_ms,
            'train_minutes'      : t_mins
        }

        # Save checkpoint for this configuration
        torch.save(
            model.state_dict(),
            f'checkpoints/ablation_{name}_final.pt'
        )

        print(f"  Overall Accuracy : {results[name]['overall_accuracy']}%")
        print(f"  Macro F1         : {results[name]['macro_f1']}%")
        print(f"  Training Time    : {t_mins} mins")
        print(f"  Inference Time   : {avg_ms} ms")

    # Save results to JSON
    with open('results/ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print final comparison table
    print_table(results, run_order)

    return results


# -----------------------------------------------------------
# Step 3 — Print the comparison table
# -----------------------------------------------------------

def print_table(results, order):

    print("\n" + "="*90)
    print(
        f"{'Config':<22} "
        f"{'Overall':>8} "
        f"{'Syntactic':>10} "
        f"{'Semantic':>9} "
        f"{'Pragmatic':>10} "
        f"{'F1':>7} "
        f"{'Params':>10} "
        f"{'ms':>7}"
    )
    print("="*90)

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

    print("="*90)
    print("\nFull results saved to: results/ablation_results.json")


# -----------------------------------------------------------
# Entry point
# -----------------------------------------------------------

if __name__ == '__main__':
    run_ablation()
