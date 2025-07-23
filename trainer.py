"""
Training pipeline for CodeLACE model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm
from typing import Dict, List, Tuple

from model import CodeLACE
from config import create_codelace_config, create_lightweight_config
from data.sample_data import create_data_loaders
from utils import set_seed, save_checkpoint, calculate_metrics, Logger


class HierarchicalLoss(nn.Module):
    """Multi-level loss function for hierarchical classification."""

    def __init__(self, syntactic_weight: float = 1.0, semantic_weight: float = 1.0, pragmatic_weight: float = 1.0):
        super().__init__()
        self.syntactic_weight = syntactic_weight
        self.semantic_weight = semantic_weight
        self.pragmatic_weight = pragmatic_weight
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, syntactic_logits, semantic_logits, pragmatic_logits,
                syntactic_labels, semantic_labels, pragmatic_labels):
        """Compute weighted combination of classification losses."""
        syntactic_loss = self.criterion(syntactic_logits, syntactic_labels)
        semantic_loss = self.criterion(semantic_logits, semantic_labels)
        pragmatic_loss = self.criterion(pragmatic_logits, pragmatic_labels)

        total_loss = (self.syntactic_weight * syntactic_loss +
                      self.semantic_weight * semantic_loss +
                      self.pragmatic_weight * pragmatic_loss)

        return total_loss, syntactic_loss, semantic_loss, pragmatic_loss


class CodeLACETrainer:
    """Trainer for CodeLACE model."""

    def __init__(self, model: CodeLACE, train_loader: DataLoader, val_loader: DataLoader,
                 learning_rate: float = 2e-5, num_epochs: int = 5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs

        # Loss function and optimizer
        self.criterion = HierarchicalLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

        # Scheduler
        total_steps = len(train_loader) * num_epochs
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1,
                                                     total_iters=total_steps)

        # Logging
        self.logger = Logger('logs/training.log')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_path = 'checkpoints/best_model.pt'

        # Create directories
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('logs', exist_ok=True)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_syntactic_preds, all_syntactic_labels = [], []
        all_semantic_preds, all_semantic_labels = [], []
        all_pragmatic_preds, all_pragmatic_labels = [], []

        progress_bar = tqdm(self.train_loader, desc="Training")

        for batch in progress_bar:
            # Move to device (CPU in this case)
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            syntactic_labels = batch['syntactic_label']
            semantic_labels = batch['semantic_label']
            pragmatic_labels = batch['pragmatic_label']

            # Forward pass
            syntactic_logits, semantic_logits, pragmatic_logits = self.model(input_ids, attention_mask)

            # Compute loss
            loss, syn_loss, sem_loss, prag_loss = self.criterion(
                syntactic_logits, semantic_logits, pragmatic_logits,
                syntactic_labels, semantic_labels, pragmatic_labels
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            # Track metrics
            total_loss += loss.item()

            # Predictions
            syntactic_preds = torch.argmax(syntactic_logits, dim=-1)
            semantic_preds = torch.argmax(semantic_logits, dim=-1)
            pragmatic_preds = torch.argmax(pragmatic_logits, dim=-1)

            all_syntactic_preds.extend(syntactic_preds.cpu().numpy())
            all_syntactic_labels.extend(syntactic_labels.cpu().numpy())
            all_semantic_preds.extend(semantic_preds.cpu().numpy())
            all_semantic_labels.extend(semantic_labels.cpu().numpy())
            all_pragmatic_preds.extend(pragmatic_preds.cpu().numpy())
            all_pragmatic_labels.extend(pragmatic_labels.cpu().numpy())

            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        syntactic_metrics = calculate_metrics(all_syntactic_preds, all_syntactic_labels)
        semantic_metrics = calculate_metrics(all_semantic_preds, all_semantic_labels)
        pragmatic_metrics = calculate_metrics(all_pragmatic_preds, all_pragmatic_labels)

        # Overall accuracy (average of all levels)
        overall_accuracy = (syntactic_metrics['accuracy'] + semantic_metrics['accuracy'] + pragmatic_metrics[
            'accuracy']) / 3

        return {
            'loss': avg_loss,
            'accuracy': overall_accuracy,
            'syntactic_accuracy': syntactic_metrics['accuracy'],
            'semantic_accuracy': semantic_metrics['accuracy'],
            'pragmatic_accuracy': pragmatic_metrics['accuracy']
        }

    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_syntactic_preds, all_syntactic_labels = [], []
        all_semantic_preds, all_semantic_labels = [], []
        all_pragmatic_preds, all_pragmatic_labels = [], []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                syntactic_labels = batch['syntactic_label']
                semantic_labels = batch['semantic_label']
                pragmatic_labels = batch['pragmatic_label']

                # Forward pass
                syntactic_logits, semantic_logits, pragmatic_logits = self.model(input_ids, attention_mask)

                # Compute loss
                loss, _, _, _ = self.criterion(
                    syntactic_logits, semantic_logits, pragmatic_logits,
                    syntactic_labels, semantic_labels, pragmatic_labels
                )

                total_loss += loss.item()

                # Predictions
                syntactic_preds = torch.argmax(syntactic_logits, dim=-1)
                semantic_preds = torch.argmax(semantic_logits, dim=-1)
                pragmatic_preds = torch.argmax(pragmatic_logits, dim=-1)

                all_syntactic_preds.extend(syntactic_preds.cpu().numpy())
                all_syntactic_labels.extend(syntactic_labels.cpu().numpy())
                all_semantic_preds.extend(semantic_preds.cpu().numpy())
                all_semantic_labels.extend(semantic_labels.cpu().numpy())
                all_pragmatic_preds.extend(pragmatic_preds.cpu().numpy())
                all_pragmatic_labels.extend(pragmatic_labels.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        syntactic_metrics = calculate_metrics(all_syntactic_preds, all_syntactic_labels)
        semantic_metrics = calculate_metrics(all_semantic_preds, all_semantic_labels)
        pragmatic_metrics = calculate_metrics(all_pragmatic_preds, all_pragmatic_labels)

        # Overall accuracy
        overall_accuracy = (syntactic_metrics['accuracy'] + semantic_metrics['accuracy'] + pragmatic_metrics[
            'accuracy']) / 3

        return {
            'loss': avg_loss,
            'accuracy': overall_accuracy,
            'syntactic_accuracy': syntactic_metrics['accuracy'],
            'semantic_accuracy': semantic_metrics['accuracy'],
            'pragmatic_accuracy': pragmatic_metrics['accuracy']
        }

    def train(self):
        """Full training loop."""
        self.logger.log("Starting CodeLACE training...")
        self.logger.log(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")

        for epoch in range(self.num_epochs):
            self.logger.log(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Log metrics
            self.logger.log(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            self.logger.log(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            self.logger.log(
                f"Syntactic - Train: {train_metrics['syntactic_accuracy']:.4f}, Val: {val_metrics['syntactic_accuracy']:.4f}")
            self.logger.log(
                f"Semantic - Train: {train_metrics['semantic_accuracy']:.4f}, Val: {val_metrics['semantic_accuracy']:.4f}")
            self.logger.log(
                f"Pragmatic - Train: {train_metrics['pragmatic_accuracy']:.4f}, Val: {val_metrics['pragmatic_accuracy']:.4f}")

            # Save training history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_accuracy'].append(train_metrics['accuracy'])
            self.training_history['val_accuracy'].append(val_metrics['accuracy'])

            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                save_checkpoint(self.model, self.optimizer, epoch, val_metrics['loss'], self.best_model_path)
                self.logger.log(f"New best model saved with validation loss: {val_metrics['loss']:.4f}")

            # Save checkpoint every epoch
            checkpoint_path = f'checkpoints/checkpoint_epoch_{epoch + 1}.pt'
            save_checkpoint(self.model, self.optimizer, epoch, val_metrics['loss'], checkpoint_path)

        # Save training history
        with open('checkpoints/training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)

        self.logger.log("\nTraining completed!")
        self.logger.log(f"Best validation loss: {self.best_val_loss:.4f}")


def main():
    """Main training function."""
    # Set seed for reproducibility
    set_seed(42)

    # Create model configuration
    # Use lightweight config for faster training on CPU
    config = create_lightweight_config()

    # Create model
    model = CodeLACE(config)
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(train_size=800, val_size=200, batch_size=16)

    # Create trainer
    trainer = CodeLACETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=2e-5,
        num_epochs=5
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()