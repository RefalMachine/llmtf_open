from pathlib import Path
import json
import shutil

from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration


FILES = [
    "chat_template.jinja",
    "cpt_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "training_args.bin",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
]


def remap_key(key: str) -> str:
    replacements = [
        ("model.language_model.language_model.language_model.", "model.language_model."),
        ("model.language_model.visual.", "model.visual.")
    ]

    for old, new in replacements:
        if key.startswith(old):
            return new + key[len(old):]

    return key


def load_safetensors_state_dict(model_dir: str):
    model_dir = Path(model_dir)
    state_dict = {}

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        files = sorted(set(index["weight_map"].values()))
    else:
        files = sorted(p.name for p in model_dir.glob("*.safetensors"))

    for filename in files:
        path = model_dir / filename
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                new_key = remap_key(key)
                if new_key in state_dict:
                    raise ValueError(f"Duplicate remapped key: {new_key}")
                state_dict[new_key] = f.get_tensor(key)

    return state_dict


def copy_files(input_dir: str, output_dir: str):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        src = input_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


input_dir = "models/RuadaptQwen3.5-2B-Base-b64-minlen4-reldist_mse_cosine_smart_k50_1e_1e3_bs144"
output_dir = "models/RuadaptQwen3.5-2B-Base-b64-minlen4-reldist_mse_cosine_smart_k50_1e_1e3_bs144-fixed"

config = AutoConfig.from_pretrained(input_dir, trust_remote_code=True)
model = Qwen3_5ForConditionalGeneration(config)

state_dict = load_safetensors_state_dict(input_dir)

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("missing:", missing)
print("unexpected:", unexpected)

model.tie_weights()

model.save_pretrained(output_dir)
copy_files(input_dir, output_dir)