import shutil
import torch
from pathlib import Path

from collections import OrderedDict
from safetensors.torch import save_file
from model_code.loader import Llama2LoadConfig

src_path = Path("../model_file/llama2")
dst_path = Path("../Llama2_Rebuild/model_file")

name_mapping = {
    'tok_embeddings.weight': 'word_embedding.weight',
    'norm.weight': 'final_ln.weight',
    'output.weight': 'lm_head.weight'
}

for i in range(32):
    name_mapping.update({
        f'layers.{i}.attention.wq.weight': f'layers.{i}.attn.q_proj.weight',
        f'layers.{i}.attention.wk.weight': f'layers.{i}.attn.k_proj.weight',
        f'layers.{i}.attention.wv.weight': f'layers.{i}.attn.v_proj.weight',
        f'layers.{i}.attention.wo.weight': f'layers.{i}.attn.o_proj.weight',
        f'layers.{i}.feed_forward.w1.weight': f'layers.{i}.ffn.gate_proj.weight',
        f'layers.{i}.feed_forward.w3.weight': f'layers.{i}.ffn.up_proj.weight',
        f'layers.{i}.feed_forward.w2.weight': f'layers.{i}.ffn.down_proj.weight',
        f'layers.{i}.attention_norm.weight': f'layers.{i}.attn_ln.weight',
        f'layers.{i}.ffn_norm.weight': f'layers.{i}.ffn_ln.weight'
    })

state_dict = torch.load(src_path / "consolidated.00.pth", map_location=torch.device("cpu"))

new_state_dict = OrderedDict()

for k, v in state_dict.items():
    print(k)
    if k not in name_mapping:
        continue
    new_state_dict[name_mapping[k]] = v

save_file(new_state_dict, dst_path / "llama2-7b.safetensors")

config = Llama2LoadConfig(
    weight_files=["llama2-7b.safetensors"],
    torch_dtype="float16",
)

shutil.copy(src_path / "tokenizer.model", dst_path / config.tokenizer_file)

config_path = dst_path / "config.json"
config_path.write_text(config.to_json())
