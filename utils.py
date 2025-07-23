"""
Utility functions for CodeLACE implementation.
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
from typing import List, Dict, Any, Optional


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def create_attention_mask(seq_length, sparsity_ratio=0.1):
    """Create sparse attention mask."""
    mask = torch.zeros(seq_length, seq_length)

    # Always attend to self
    mask.fill_diagonal_(1)

    # Add random sparse connections
    num_connections = int(seq_length * seq_length * sparsity_ratio)
    indices = torch.randperm(seq_length * seq_length)[:num_connections]
    rows = indices // seq_length
    cols = indices % seq_length
    mask[rows, cols] = 1

    return mask


def calculate_metrics(predictions, labels):
    """Calculate evaluation metrics."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def save_results(results: Dict[str, Any], filepath: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {filepath}")


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


class Logger:
    """Simple logger for training progress."""

    def __init__(self, log_file: str):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log(self, message: str):
        """Log message to file and console."""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"{message}\n")