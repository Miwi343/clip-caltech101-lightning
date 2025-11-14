"""
Main training script for CLIP-based Caltech-101 classifier
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import wandb
from src.model import CLIPClassifier
from src.data import get_data_loaders


def main():
    # Hyperparameters
    NUM_CLASSES = 102  # 101 categories + 1 background class
    LEARNING_RATE = 1e-3
    MAX_EPOCHS = 20
    
    wandb.init(
        project="clip-caltech101",
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": "CLIP-ViT-B/32 + Linear",
            "dataset": "Caltech-101",
            "epochs": MAX_EPOCHS,
        }
    )
    
    # Setup WandB 
    wandb_logger = WandbLogger(
        project="clip-caltech101",
        log_model=True  # Log model checkpoints to WandB
    )
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, dataset = get_data_loaders()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Initialize model
    print("\nInitializing model...")
    model = CLIPClassifier(
        num_classes=NUM_CLASSES,
        learning_rate=LEARNING_RATE
    )
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="clip-caltech101-{epoch:02d}-{val_acc:.2f}",
        monitor="val_acc",
        mode="max",
        save_top_k=3,
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        verbose=True
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",  # Automatically use GPU if available
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
    )
    
    # Train
    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Log best metrics
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best validation accuracy: {checkpoint_callback.best_model_score:.4f}")
    print("="*50)
    
    wandb.finish()


if __name__ == "__main__":
    main()