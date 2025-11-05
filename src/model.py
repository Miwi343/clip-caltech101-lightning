"""
Trains model
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import CLIPModel


class CLIPClassifier(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3):
        super().__init__()

        self.save_hyperparameters()

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False

        embedding_dim = self.clip_model.config.projection_dim

        self.classifier = nn.Linear(embedding_dim, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, images):
        """
        images: tensor of shape [B, 3, 224, 224]
        returns: logits of shape [B, num_classes]
        """
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(pixel_values=images)
        logits = self.classifier(image_features)
        return logits

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)

        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)

        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        # Only train the classifier head
        optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.hparams.learning_rate,
        )
        return optimizer