"""Repair the duplicated Qwen3.5 module prefixes in a local checkpoint.

This is a maintainer utility for historical checkpoints, not an LLMTF runtime
entry point. It deliberately imports the heavyweight model dependencies only
while executing the conversion so its pure key mapping remains testable.
"""

import argparse
import json
import shutil
from pathlib import Path


COPY_FILES = (
    "chat_template.jinja",
    "cpt_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "training_args.bin",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)


def remap_key(key: str) -> str:
    replacements = (
        (
            "model.language_model.language_model.language_model.",
            "model.language_model.",
        ),
        ("model.language_model.visual.", "model.visual."),
    )
    for old, new in replacements:
        if key.startswith(old):
            return new + key[len(old):]
    return key


def load_safetensors_state_dict(model_dir: Path):
    from safetensors import safe_open

    state_dict = {}
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as source:
            index = json.load(source)
        filenames = sorted(set(index["weight_map"].values()))
    else:
        filenames = sorted(path.name for path in model_dir.glob("*.safetensors"))
    if not filenames:
        raise FileNotFoundError(f"No safetensors checkpoint found in {model_dir}")

    for filename in filenames:
        path = model_dir / filename
        with safe_open(path, framework="pt", device="cpu") as source:
            for key in source.keys():
                new_key = remap_key(key)
                if new_key in state_dict:
                    raise ValueError(f"Duplicate remapped key: {new_key}")
                state_dict[new_key] = source.get_tensor(key)
    return state_dict


def copy_tokenizer_files(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in COPY_FILES:
        source = input_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)


def remap_checkpoint(input_dir: Path, output_dir: Path, *, trust_remote_code=False):
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
    )

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if input_dir == output_dir:
        raise ValueError("input and output directories must be different")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input checkpoint directory does not exist: {input_dir}")

    config = AutoConfig.from_pretrained(
        input_dir, trust_remote_code=trust_remote_code
    )
    model = Qwen3_5ForConditionalGeneration(config)
    state_dict = load_safetensors_state_dict(input_dir)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("missing:", missing)
    print("unexpected:", unexpected)

    model.tie_weights()
    model.save_pretrained(output_dir)
    copy_tokenizer_files(input_dir, output_dir)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Repair duplicated module prefixes in a Qwen3.5 checkpoint"
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    remap_checkpoint(
        args.input_dir,
        args.output_dir,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
