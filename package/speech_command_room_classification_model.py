from typing import Union, List, Optional, Tuple, Dict, Any

import numpy as np
from tqdm import tqdm

import torch
import transformers
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision import models
from torch import nn, Tensor
from torch.optim import Optimizer
from pytorch_lightning import metrics
from pytorch_lightning.metrics import MetricCollection

from .speech_command_classification_config import SpeechCommandClassificationConfig


class SpeechCommandClassificationModel(pl.LightningModule):

    def __init__(self,
                 cfg: SpeechCommandClassificationConfig,
                 num_warmup_steps: int = 10,
                 num_training_steps: int = 100,
                 class_weights: Optional[List] = None):
        """
        Resnet based pytorch lightning model for multiclass classification.

        Parameters
        ----------
        cfg : (SpeechCommandClassificationConfig) Config, which holds full configuration
                for multiclass classification pipeline.
        num_warmup_steps : (int) Number steps for warmup.
        num_training_steps :  (int) Number steps for training.
        class_weights : (float or None)  pos_weight for disbalanced dataset.
        """

        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.labels = cfg.labels
        self.lr = cfg.lr
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.class_weights = class_weights

        self.train_classification_metrics, self.valid_classification_metrics = self.configure_metrics()
        self.criterion = self.configure_criterion()

        # Build model
        self.model = models.__dict__[cfg.model_name](pretrained=cfg.pretrained)
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=len(cfg.labels), bias=True)

    def configure_metrics(self) -> Tuple[MetricCollection, MetricCollection]:
        metric_classification = metrics.MetricCollection(
            [
                metrics.Precision(num_classes=len(self.labels), average=self.cfg.metric_average),
                metrics.Recall(num_classes=len(self.labels), average=self.cfg.metric_average),
                metrics.F1(num_classes=len(self.labels), average=self.cfg.metric_average),
                metrics.Accuracy(num_classes=len(self.labels), average=self.cfg.metric_average),
            ]
        )

        train_metrics = metric_classification.clone(prefix='train_')
        valid_metrics = metric_classification.clone(prefix='valid_')

        return train_metrics, valid_metrics

    def calculate_metrics(self, y_pred: Tensor, y_true: Tensor, mode: str = 'train'):
        on_step = True if mode == 'train' else False
        prob = torch.sigmoid(y_pred)

        if mode == 'train':
            metric_scores = self.train_classification_metrics(prob, y_true)
        else:
            metric_scores = self.valid_classification_metrics(prob, y_true)

        self.log_dict(metric_scores, on_step=on_step, on_epoch=True)

    def forward(self, x_in: Tensor) -> Tensor:
        return self.model(x_in)

    def training_step(self,batch: Dict[str, Tensor], batch_idx: Tensor) -> Tensor:
        x, y = batch

        output = self(x)
        loss = self.criterion(output, y)

        self.log('train_loss', loss)
        self.calculate_metrics(output, y, mode='train')

        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: Tensor) -> Tensor:
        x, y = batch
        output = self(x)

        loss = self.criterion(output, y)

        self.log('valid_loss', loss)
        self.calculate_metrics(output, y, mode='valid')
        return loss

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Dict[str, Any]]]:

        optimizer = transformers.AdamW(
            self.parameters(),
            lr=self.lr,
        )

        lr_scheduler = {
            'name': 'lr_scheduler',
            'interval': 'step',
            'frequency': 1
        }

        if self.cfg.lr_scheduler_name == 'linear':
            lr_scheduler['scheduler'] = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                                     num_warmup_steps=self.num_warmup_steps,
                                                                                     num_training_steps=self.num_training_steps
                                                                                     )
        elif self.cfg.lr_scheduler_name == 'cosine':
            lr_scheduler['scheduler'] = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                                                   num_warmup_steps=self.num_warmup_steps,
                                                                                   num_training_steps=self.num_training_steps
                                                                                   )

        return [optimizer], [lr_scheduler]

    def configure_criterion(self) -> nn.Module:
        if self.class_weights is None:
            return nn.CrossEntropyLoss()
        else:
            return nn.CrossEntropyLoss(weight=torch.FloatTensor(self.class_weights))

    def predict_proba(self,
                      images: Union[np.ndarray, List[np.ndarray]],
                      batch_size=16,
                      show_progress_bar=False,
                      mc_prediction=False,
                      transform=None
                      ) -> Dict[str, np.array]:

        self.eval()

        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.cfg.img_size),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        if mc_prediction:
            def apply_dropout(m):
                if type(m) == nn.Dropout:
                    m.train()

            self.apply(apply_dropout)

        all_scores = np.zeros((len(images), len(self.labels)))
        with torch.no_grad():
            for start_index in tqdm(
                    range(0, len(images), batch_size),
                    desc="Predict probability",
                    disable=not show_progress_bar,
            ):
                batch = images[start_index: start_index + batch_size]
                batch_len = len(batch)
                batch = [transform(img) for img in batch]
                batch = torch.stack(batch)
                batch = batch.to(self.device)

                score = self.forward(batch)
                score = torch.softmax(score, dim=1)
                score = score.detach().cpu()
                all_scores[start_index: start_index + batch_len] = score

        all_scores = {col: all_scores[:, i] for i, col in enumerate(self.labels)}

        return all_scores
