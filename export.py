"""Export the validation-selected baseline and 512-Atom release checkpoints."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from safetensors.torch import save_file
from torchcst.hub import load_safetensors

from model import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = {
        variant: json.loads((args.run / variant / "metrics.json").read_text())
        for variant in ("baseline", "cst")
    }
    baseline, cst = records.values()
    for key in ("dataset", "seed", "epochs", "batch_size", "shared_initial_sha256"):
        if baseline[key] != cst[key]:
            raise ValueError(f"Release training conditions differ in {key}")
    if cst["atoms_per_projection"] != 512:
        raise ValueError("Expected the 512-Atom release run")
    args.output.mkdir(parents=True, exist_ok=False)
    config = {"labels": baseline["dataset"]["classes"], "variants": {}, "release_evaluation": {}}
    x = torch.randn(2, 1, 40, 98, generator=torch.Generator().manual_seed(17))
    for variant, record in records.items():
        name = "cst-512" if variant == "cst" else variant
        options = dict(variant=variant, atoms=512, backend="materialized")
        factory = partial(build_model, **options)
        original = factory().eval()
        state = torch.load(args.run / variant / "best.pt", map_location="cpu", weights_only=True)
        original.load_state_dict(state)
        path = args.output / f"{name}.safetensors"
        save_file({
            key: value.detach().contiguous().clone()
            for key, value in state.items() if torch.is_tensor(value)
        }, path, metadata={"format": "pt"})
        portable = load_safetensors(path, model_factory=factory)
        with torch.inference_mode():
            torch.testing.assert_close(portable(x), original(x), rtol=0, atol=0)
        config["variants"][name] = options
        test = record["training"]
        config["release_evaluation"][name] = {
            "seed": record["seed"], "epoch": test["best_epoch"],
            "test_accuracy": test["test"]["accuracy"],
            "last_test_accuracy": test["last_test"]["accuracy"],
        }
        print(name, "Safetensors logits match the checkpoint", flush=True)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
