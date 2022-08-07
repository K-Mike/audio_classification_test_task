import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Union, Sequence, Type, Optional, Tuple

import dacite

from .json_encoders import ConfigEncoder


def _remove_none_values(d: Dict[str, Any]) -> Dict[str, Any]:
    res = dict()
    for key, val in d.items():
        if val is None:
            continue
        elif isinstance(val, dict):
            res[key] = _remove_none_values(val)
        else:
            res[key] = val
    return res


@dataclasses.dataclass
class ModelConfig:

    def to_dict(self) -> Dict[str, Any]:
        """
        Recursively convert fields to dict, skipping None valued fields
        """
        fields = dataclasses.asdict(self)
        fields = _remove_none_values(fields)
        return fields

    def save(self,
             save_path: Union[str, Path],
             exclude_fields: Optional[Sequence[str]] = None,
             encoder: Optional[Type[json.JSONEncoder]] = ConfigEncoder,
             ensure_ascii=True):

        exclude_fields = exclude_fields or []

        to_save = {key: val for key, val in self.to_dict().items()
                   if key not in exclude_fields}

        with Path(save_path).open('wt') as fp:
            json.dump(
                obj=to_save,
                fp=fp,
                cls=encoder,
                ensure_ascii=ensure_ascii,
                sort_keys=True,
                indent=4,
            )

    @classmethod
    def load(cls,
             load_path: Union[str, Path],
             decoder: Optional[Type[json.JSONDecoder]] = None) -> 'ModelConfig':

        with Path(load_path).open('rt') as fp:
            data = json.load(fp=fp, cls=decoder)

        return dacite.from_dict(
            data_class=cls,
            data=data,
            config=dacite.Config(
                cast=[Tuple, Path],
            ),
        )
