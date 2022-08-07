import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'  # noqa

import resource
from pathlib import Path

from package.speech_command_classification_config import SpeechCommandClassificationConfig
from package.train_model import train_model
from package import constants


def set_file_descriptor_limit(val: int):
    r_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (val, r_limit[1]))


if __name__ == '__main__':
    set_file_descriptor_limit(50_000)

    os.environ['WANDB_API_KEY'] = constants.WANDB_API_KEY

    train_cfg = SpeechCommandClassificationConfig(
        wandb_entity=constants.WANDB_ENTITY,
        device=0,
    )

    train_model(
        cfg=train_cfg,
        predict_valid=True,
    )
