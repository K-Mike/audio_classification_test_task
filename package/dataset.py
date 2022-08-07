
from pathlib import Path
from typing import Tuple, Optional, Callable

import cv2
import torch
from torch import Tensor
from torch.utils.data import Dataset
from datasets import DatasetDict

from . import utils
from .speech_command_classification_config import SpeechCommandClassificationConfig


class SpeechCommandDataset(Dataset):

    def __init__(self, data: DatasetDict, img_dir: Path,
                 transforms: Optional[Callable] = None, duration=1600):
        super().__init__()

        self.data = data
        self.img_dir = img_dir
        self.transforms = transforms
        self.duration = duration

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        sample = self.data[idx]

        img_path = self.img_dir / f'{sample["file"][:-4]}.jpg'
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

        # Applying transforms on image
        if self.transforms is not None:
            image = self.transforms(image)

        label = self.data[idx]['label']
        label = torch.tensor(label, dtype=torch.long)

        return image, label

    def __len__(self) -> int:
        return len(self.data)


def generate_melspec_image_dataset(cfg: SpeechCommandClassificationConfig, dataset: DatasetDict):
    output_dir = Path(cfg.input_img_dir)

    for row in utils.iterate_over_all_samples(dataset):
        p_out = output_dir / f'{row["file"][:-4]}.jpg'

        if not p_out.parent.exists():
            p_out.parent.mkdir(parents=True)

        y, sr = row['audio']['array'], row['audio']['sampling_rate']
        y = utils.pad_audio(y, duration=cfg.duration, clip_silence=cfg.clip_silence)
        melspec = utils.create_melspec(y, sr, n_mels=cfg.n_mels)
        melspec_color = utils.mono_to_color(melspec)

        cv2.imwrite(str(p_out), melspec_color)
