import json
import time
import torch
import numpy as np
from config import create_ablation_configs
from model import CodeLACE

def run_ablation(trainer_class, train_loader, val_loader, test_loader):

    configs  = create_ablation_configs()
    results  = {}

    order = [
        'full_codelace',
        'no_moe',
        'no_token_pooling',
        'no_sparse_attention',
        'baseline'
    ]

    for name in order:
        config = configs[name]
        print(f"\nRunning: {name}")
        print(f"  sparse_attention : {config.use_sparse_attention}")
        print(f"  token_pooling    : {config.use_token_pooling}")
        print(f"  moe              : {config.use_moe}")

        # Fix seed for every run
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        model   = CodeLACE(config)
        params  = sum(p.numel() for p in model.parameters()
                      if p.requires_grad)

        trainer = trainer_class(model, config)
        t_start = time.time()
        trainer.train(train_loader, val_loader, epochs=5)
        t_end   = (time.time() - t_start) / 60

        # Inference time over 100 batches
        model.eval()
        sample  = next(iter(test_loader))
        times   = []
        with torch.no_grad():
            for _ in range(100):
                t0 = time.time()
                model(
                    sample['input_ids'],
                    sample.get('attention_mask', None)
                )
                times.append((time.time() - t0) * 1000)

        metrics = trainer.evaluate(test_loader)

        results[name] = {
            'overall_accuracy'   : metrics['overall_accuracy'],
            'syntactic_accuracy' : metrics['syntactic_accuracy'],
            'semantic_accuracy'  : metrics['semantic_accuracy'],
            'pragmatic_accuracy' : metrics['pragmatic_accuracy'],
            'macro_f1'           : metrics['macro_f1'],
            'parameters'         : params,
            'inference_ms'       : round(np.mean(times), 2),
            'train_minutes'      : round(t_end, 1)
        }

        torch.save(
            model.state_dict(),
            f"checkpoints/ablation_{name}.pt"
        )
        print(f"  Done — accuracy: {metrics['overall_accuracy']:.1f}%")

    with open('ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print_table(results, order)
    return results


def print_table(results, order):
    h = (f"{'Config':<22} {'Overall':>8} {'Syntactic':>10} "
         f"{'Semantic':>9} {'Pragmatic':>10} "
         f"{'F1':>7} {'Params':>9} {'ms':>7}")
    print("\n" + "="*85)
    print(h)
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
            f"{r['parameters']:>9,} "
            f"{r['inference_ms']:>6.1f}"
        )
    print("="*85)
