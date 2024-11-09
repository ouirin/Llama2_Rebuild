import json
import sys

import torch

from pathlib import Path
from tqdm.auto import tqdm
from dataclasses import dataclass, asdict, field
from safetensors.torch import safe_open
from model_code.tokenizer import Llama2Tokenizer
from model_code.model import Llama2Model, Llama2Config


@dataclass
class Llama2LoadConfig():

    model_type: Llama2Model = "Llama2Model"
    model_config: Llama2Config = field(default_factory=Llama2Config)
    quant_type: str = "none"
    weight_files: list = field(default_factory=list)
    tokenizer_file: str = "sentencepiece.model"
    torch_dtype: str = "float32"

    def __post_init__(self):
        if not isinstance(self.model_config, Llama2Config):
            self.model_config = Llama2Config(**self.model_config)

    def get_torch_dtype(self):
        return getattr(torch, self.torch_dtype)

    @staticmethod
    def from_json(json_str):
        return Llama2LoadConfig(**json.loads(json_str))

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


@torch.no_grad()
def load_model_and_tokenizer(model_path):

    model_path = Path(model_path)
    config_path = model_path / "config.json"
    config = Llama2LoadConfig.from_json(config_path.read_bytes())

    model = Llama2Model(config.model_config, config.get_torch_dtype())
    state_dict = dict(**model.state_dict())

    for file in tqdm(config.weight_files):

        with safe_open(model_path / file, framework='pt') as f:

            for k in f.keys():

                state_dict[k].copy_(f.get_tensor(k))
                state_dict.pop(k)

    tokenizer = Llama2Tokenizer(model_path / config.tokenizer_file)

    return config, model, tokenizer
