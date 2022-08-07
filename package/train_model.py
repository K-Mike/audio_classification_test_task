
import sys
import json
import gzip
import shutil
import random
from pathlib import Path

import cv2
import wandb
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader  # noqa
from pytorch_lightning.loggers import WandbLogger
from datasets import load_dataset
import torchvision.transforms as transforms
from sklearn.utils.class_weight import compute_class_weight

import warnings
warnings.filterwarnings("ignore")

from .speech_command_classification_config import SpeechCommandClassificationConfig
from .speech_command_room_classification_model import SpeechCommandClassificationModel
from .dataset import SpeechCommandDataset
from .logger import AdvancedLoggerManager


def train_model(
        cfg: SpeechCommandClassificationConfig,
        predict_valid=True):
    """
    Train classification model with pytorch lightning.

    Parameters
    ----------
    cfg : SpeechCommandClassificationConfig, which holds full configuration for multiclass classification pipeline.
    predict_valid : bool,  Whether or not to predict validation data on the best checkpoint.

    Returns
    -------
    None
    """

    print('Start train Room Classification Model')

    # Init Wandb
    wandb.login()
    wandb.init(
        project=cfg.project_name,
        entity=cfg.wandb_entity,
        reinit=True,
        name=cfg.run_name,
        config=cfg.to_dict(),
    )

    output_dir = Path(cfg.output_dir)
    # Directory guard
    if output_dir.exists():
        decision = input(f'Output dir {cfg.output_dir} already exists. Remove it? (y/n)')
        if decision != 'y':
            sys.exit(1)

        shutil.rmtree(cfg.output_dir)
    output_dir.mkdir()

    # Init logger
    logger = AdvancedLoggerManager(output_dir / 'training.log', logger_name='training')
    logger.info(f'Project: {cfg.project_name} run: {cfg.run_name}')
    cfg.save(output_dir / 'train_config.json')

    # Data
    dataset = load_dataset('speech_commands', 'v0.02', cache_dir='/mnt/raid/huggingface_cache_dir')

    if cfg.debug:
        for col in ['train', 'validation']:
            dataset[col] = dataset[col].select(random.sample(range(len(dataset[col])), 1000))

    logger.info(f'Train n samples {len(dataset["train"])}')
    logger.info(f'Valid n samples {len(dataset["validation"])}')

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(cfg.img_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    train_dataset = SpeechCommandDataset(
        data=dataset['train'],
        img_dir=cfg.input_img_dir,
        transforms=transform,
    )

    valid_dataset = SpeechCommandDataset(
        data=dataset['test'],
        img_dir=cfg.input_img_dir,
        transforms=transform,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_loader_workers,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_loader_workers,
    )

    # Model
    num_training_steps = int(cfg.epoch * len(train_loader))
    num_warmup_steps = int(num_training_steps * cfg.warmup_steps_p)

    # Calculate class weights for criterion
    y = [row['label'] for row in dataset['test']]
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weights = 1.0 / class_weights

    model = SpeechCommandClassificationModel(
        cfg,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        class_weights=class_weights if cfg.use_class_weights else None
    )

    # Train
    wandb_logger = WandbLogger(project=cfg.project_name, name=cfg.run_name)
    lr_monitor_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')
    model_checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        str(output_dir),
        monitor='valid_F1',
        save_top_k=1,
        mode='max',
    )

    trainer = pl.Trainer(
        gpus=[cfg.device] if isinstance(cfg.device, int) else cfg.device,
        max_epochs=cfg.epoch,
        default_root_dir=str(output_dir),
        logger=wandb_logger,
        log_every_n_steps=1,
        num_sanity_val_steps=1,
        callbacks=[
            lr_monitor_callback,
            model_checkpoint_callback,
        ],
        gradient_clip_val=cfg.clip_value,
    )

    trainer.fit(
        model=model,
        train_dataloader=train_loader,
        val_dataloaders=valid_loader,

    )

    logger.info(f'Best model path: {model_checkpoint_callback.best_model_path}')

    # Predict valid dataset and save result
    if predict_valid:
        model = SpeechCommandClassificationModel.load_from_checkpoint(model_checkpoint_callback.best_model_path)
        device_idx = cfg.device if isinstance(cfg.device, int) else cfg.device[0]  # noqa
        device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        def load_img(sample):
            img_path = cfg.input_img_dir / f'{sample["file"][:-4]}.jpg'
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

            return image

        valid_images = [load_img(sample) for sample in dataset['test']]

        predict_all = model.predict_proba(valid_images)
        prob_all = np.stack([predict_all[col] for col in cfg.labels], axis=1)

        y_true = np.zeros_like(prob_all)
        for i, d in enumerate(dataset['test']):
            label_idx = d['label']
            y_true[i, label_idx] = 1

        y_pred = (prob_all > 0.5).astype(np.int_)

        precision_valid = precision_score(y_true, y_pred, average=cfg.metric_average)
        recall_valid = recall_score(y_true, y_pred, average=cfg.metric_average)
        f1_valid = f1_score(y_true, y_pred, average=cfg.metric_average)
        auc_valid = roc_auc_score(y_true, prob_all, average=cfg.metric_average)
        accuracy_valid = accuracy_score(y_true, y_pred)

        logger.info(f'Precision: {precision_valid}')
        logger.info(f'Recall: {recall_valid}')
        logger.info(f'F1: {f1_valid}')
        logger.info(f'accuracy: {accuracy_valid}')
        logger.info(f'Auc: {auc_valid}')

        status_metrics = {
            'precision': precision_valid,
            'recall': recall_valid,
            'f1': f1_valid,
            'accuracy': accuracy_valid,
            'auc': auc_valid
        }
        wandb.log(status_metrics)

        # Predict for each label
        for i, col in enumerate(cfg.labels):
            precision_valid = precision_score(y_true[:, i], y_pred[:, i], average='binary')
            recall_valid = recall_score(y_true[:, i], y_pred[:, i], average='binary')
            f1_valid = f1_score(y_true[:, i], y_pred[:, i], average='binary')
            auc_valid = roc_auc_score(y_true[:, i], prob_all[:, i])

            logger.info(f'Precision_{col}: {precision_valid}')
            logger.info(f'Recall_{col}: {recall_valid}')
            logger.info(f'F1_{col}: {f1_valid}')
            logger.info(f'Auc_{col}: {auc_valid}')
            status_metrics = {
                f'precision_{col}': precision_valid,
                f'recall_{col}': recall_valid,
                f'f1_{col}': f1_valid,
                f'auc_{col}': auc_valid
            }
            wandb.log(status_metrics)

        valid_data_pred = list()
        for i, d in enumerate(dataset['test']):
            for label, scores in predict_all.items():
                d['audio']['array'] = list(d['audio']['array'])
                d[label] = scores[i]

            valid_data_pred.append(d)

        with gzip.open(output_dir / 'valid_pred.json.gz', 'wt') as handler:
            json.dump(valid_data_pred, handler)
