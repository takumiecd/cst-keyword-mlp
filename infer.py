"""Recognize the Speech Commands label of a 16 kHz mono PCM16 WAV clip."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torchcst.hub import from_pretrained, load_safetensors

from data import Features, read_wave
from model import build_model


HUB = "takumiecd/cst-keyword-mlp"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wave", type=Path)
    parser.add_argument("--variant", choices=("baseline", "cst-512"), default="cst-512")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local", type=Path, help="Directory with config.json and Safetensors")
    args = parser.parse_args()
    config_path = args.local / "config.json" if args.local else Path(
        hf_hub_download(HUB, "config.json", revision=args.revision, token=False)
    )
    config = json.loads(config_path.read_text())
    factory = partial(build_model, **config["variants"][args.variant])
    name = f"{args.variant}.safetensors"
    if args.local:
        model = load_safetensors(args.local / name, model_factory=factory, device=args.device)
    else:
        model = from_pretrained(
            HUB, filename=name, model_factory=factory, device=args.device,
            revision=args.revision, token=False,
        )
    with torch.inference_mode():
        sound = torch.from_numpy(read_wave(args.wave)).unsqueeze(0)
        x = Features()(sound).unsqueeze(1).to(args.device)
        probabilities, indices = model(x).softmax(-1)[0].topk(5)
    print(json.dumps([
        {"label": config["labels"][int(index)], "probability": float(probability)}
        for probability, index in zip(probabilities, indices, strict=True)
    ], indent=2))


if __name__ == "__main__":
    main()
