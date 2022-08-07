import dataclasses
from pathlib import Path
from typing import Union, List, Sequence, Tuple

from .model_config import ModelConfig


@dataclasses.dataclass
class SpeechCommandClassificationConfig(ModelConfig):

    # Common
    debug: bool = False
    project_name: str = 'speech_command'
    run_name: str = 'resnet34_18_v1'
    wandb_entity: str = 'kmike'
    device: Union[int, List[int]] = 0
    seed: int = 42

    # Mel
    duration: int = 16000
    n_mels: int = 128
    clip_silence: bool = False

    # Model parameters:
    lr_scheduler_name: str = 'linear'
    model_name: str = 'resnet34'
    pretrained: bool = True
    freeze_encoder: bool = False

    # Dataset parameters:
    img_size: Union[int, Tuple[int]] = (224, 224)
    labels: Union[Sequence[str], str] = ('yes',
                                         'no',
                                         'up',
                                         'down',
                                         'left',
                                         'right',
                                         'on',
                                         'off',
                                         'stop',
                                         'go',
                                         'zero',
                                         'one',
                                         'two',
                                         'three',
                                         'four',
                                         'five',
                                         'six',
                                         'seven',
                                         'eight',
                                         'nine',
                                         'bed',
                                         'bird',
                                         'cat',
                                         'dog',
                                         'happy',
                                         'house',
                                         'marvin',
                                         'sheila',
                                         'tree',
                                         'wow',
                                         'backward',
                                         'forward',
                                         'follow',
                                         'learn',
                                         'visual',
                                         '_silence_'
                                         )

    # Train parameters:
    input_img_dir: Path = Path('/mnt/data/zp/languageconfidenceai_test_task/data/images_v1')
    output_dir: Path = Path('/mnt/data/zp/languageconfidenceai_test_task/experiments/resnet34_18_v2')
    epoch: int = 18
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 0
    clip_value: float = 1.0
    warmup_steps_p: float = 0.1
    num_loader_workers: int = 40
    use_class_weights: bool = True
    metric_average: str = 'micro'
