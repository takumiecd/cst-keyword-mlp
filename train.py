"""Train KW-MLP from scratch or time a short paired training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F
from torchcst import CSTAdamW, CSTLinear

from data import CachedSpeech, spec_augment
from model import build_model, shared_initial_sha256


UPSTREAM_REVISION = "a87c8ac9c4d3ba01745d68cc5dcf9cd21dd67b72"
TORCHCST_REVISION = "1750c5ea6cbefe3d4a7d8f6ba02fd9272df3cf98"
NUM_CLASSES = 35
# The reference assigns 0.1 mass to the 34 wrong classes only.
LABEL_SMOOTHING = 0.1 * NUM_CLASSES / (NUM_CLASSES - 1)


def make_optimizer(model, args):
    options = dict(lr=args.lr, weight_decay=0.1, foreach=True)
    if args.variant == "baseline":
        return torch.optim.AdamW(model.parameters(), **options)
    atoms = [module.atoms.p for module in model.modules() if isinstance(module, CSTLinear)]
    excluded = {id(parameter) for parameter in atoms}
    groups = [
        {"params": [p for p in model.parameters() if id(p) not in excluded]},
        {"params": atoms, "lr": args.atom_lr, "weight_decay": 0.0},
    ]
    return CSTAdamW(groups, model=model, amplitude_weight_decay=0.1, **options)


def train_step(model, optimizer, features, targets):
    optimizer.zero_grad(set_to_none=True)
    scores = model(features)
    loss = F.cross_entropy(scores, targets, label_smoothing=LABEL_SMOOTHING)
    loss.backward()
    optimizer.step()
    return loss.detach()


def batches(indices, size):
    for start in range(0, len(indices), size):
        yield indices[start : start + size]


def train_batch(data, indices, device, augmentation):
    features, targets = data.batch(indices, device)
    return spec_augment(features, augmentation), targets


@torch.inference_mode()
def evaluate(model, data, indices, batch_size, device):
    model.eval()
    loss = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    for batch in batches(indices, batch_size):
        features, targets = data.batch(batch, device)
        predictions = model(features)
        loss += F.cross_entropy(predictions, targets, reduction="sum")
        correct += (predictions.argmax(dim=1) == targets).sum()
    result = {"loss": loss.item() / len(indices), "accuracy": correct.item() / len(indices)}
    if not math.isfinite(result["loss"]):
        raise RuntimeError("Non-finite evaluation loss")
    return result


def pilot(model, optimizer, data, args, device, shuffle, augmentation):
    total = args.pilot_warmup + args.pilot_updates
    if total * args.batch_size > len(data.train):
        raise ValueError("Pilot needs enough full batches from the training split")
    indices = shuffle.permutation(data.train)[: total * args.batch_size]
    initial_atoms = [
        (module.atoms.p, module.atoms.p.detach().cpu().clone())
        for module in model.modules()
        if isinstance(module, CSTLinear)
    ]
    times = []
    losses = torch.empty(total, device=device)
    model.train()
    for index, batch in enumerate(batches(indices, args.batch_size)):
        x, y = train_batch(data, batch, device, augmentation)
        torch.cuda.synchronize(device)
        if index == args.pilot_warmup:
            torch.cuda.reset_peak_memory_stats(device)
        start = perf_counter()
        loss = train_step(model, optimizer, x, y)
        torch.cuda.synchronize(device)
        elapsed = (perf_counter() - start) * 1000
        losses[index] = loss
        if index >= args.pilot_warmup:
            times.append(elapsed)
    peak = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    if not torch.isfinite(losses).all():
        raise RuntimeError("Non-finite pilot training loss")
    atom_updates = [
        (parameter.detach().cpu() - initial).abs().max().item()
        for parameter, initial in initial_atoms
    ]
    if any(not math.isfinite(value) or value == 0 for value in atom_updates):
        raise RuntimeError("Invalid or missing CST Atom update")
    return {
        "warmup_updates": args.pilot_warmup,
        "timed_updates": len(times),
        "batch_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "median_step_ms": float(np.median(times)),
        "mean_step_ms": float(np.mean(times)),
        "p90_step_ms": float(np.quantile(times, 0.9)),
        "step_ms": times,
        "peak_allocated_bytes": peak,
        "peak_reserved_bytes": reserved,
        "last_loss": losses[-1].item(),
        "atom_update_max_per_projection": atom_updates,
    }


def train(model, optimizer, data, args, device, shuffle, augmentation):
    steps = math.ceil(len(data.train) / args.batch_size)
    warmup_steps = min(args.lr_warmup_epochs, args.epochs - 1) * steps
    total_steps = args.epochs * steps

    def schedule(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps - 1), 1)
        return 1e-5 + (1 - 1e-5) * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)
    history = []
    best_accuracy = -1.0
    best_epoch = None
    checkpoint = args.output / "best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = torch.zeros((), device=device)
        for indices in batches(shuffle.permutation(data.train), args.batch_size):
            features, targets = train_batch(data, indices, device, augmentation)
            total_loss += train_step(model, optimizer, features, targets) * len(indices)
            scheduler.step()
        training_loss = total_loss.item() / len(data.train)
        if not math.isfinite(training_loss):
            raise RuntimeError("Non-finite training loss")
        validation = evaluate(model, data, data.validation, args.batch_size, device)
        record = {"epoch": epoch, "train_loss": training_loss, "validation": validation}
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = validation["accuracy"], epoch
            temporary = checkpoint.with_suffix(".tmp")
            torch.save(model.state_dict(), temporary)
            temporary.replace(checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return {
        "steps_per_epoch": steps,
        "lr_warmup_steps": warmup_steps,
        "history": history,
        "best_epoch": best_epoch,
        "test": evaluate(model, data, data.test, args.batch_size, device),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("baseline", "cst"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--lr-warmup-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--atom-lr", type=float, default=0.0048)
    parser.add_argument("--atoms", type=int, default=128, help="Atoms per temporal projection")
    parser.add_argument("--backend", choices=("materialized", "factored"), default="materialized")
    parser.add_argument("--pilot-updates", type=int, default=0)
    parser.add_argument("--pilot-warmup", type=int, default=15)
    args = parser.parse_args()
    if min(args.batch_size, args.atoms, args.epochs) <= 0:
        parser.error("Batch size, epochs and atoms must be positive")
    if min(args.lr_warmup_epochs, args.pilot_warmup, args.pilot_updates) < 0:
        parser.error("Warmup and pilot update counts must be nonnegative")
    if not all(math.isfinite(rate) and rate > 0 for rate in (args.lr, args.atom_lr)):
        parser.error("Learning rates must be finite and positive")
    if not torch.cuda.is_available():
        parser.error("A CUDA-enabled GPU is required")
    device = torch.device("cuda")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    data = CachedSpeech(args.cache)
    torch.manual_seed(args.seed)
    model = build_model(args.variant, args.atoms, args.backend)
    initial = shared_initial_sha256(model)
    model.to(device)
    optimizer = make_optimizer(model, args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    shuffle = np.random.default_rng(args.seed)
    augmentation = torch.Generator(device=device).manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    gpu = torch.cuda.get_device_properties(device)
    result = {
        "variant": args.variant,
        "backend": args.backend if args.variant == "cst" else "conv1d",
        "epochs": args.epochs if not args.pilot_updates else None,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "atoms_per_projection": args.atoms if args.variant == "cst" else None,
        "learning_rates": [group["lr"] for group in optimizer.param_groups],
        "parameters": sum(p.numel() for p in model.parameters()),
        "shared_initial_sha256": initial,
        "dataset": data.manifest,
        "upstream_revision": UPSTREAM_REVISION,
        "torchcst_revision": TORCHCST_REVISION,
        "device": gpu.name,
        "device_memory_bytes": gpu.total_memory,
        "device_capability": [gpu.major, gpu.minor],
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "precision": "FP32 with TF32 enabled",
    }
    start = perf_counter()
    action = pilot if args.pilot_updates else train
    result["pilot" if args.pilot_updates else "training"] = action(
        model, optimizer, data, args, device, shuffle, augmentation
    )
    result["wall_seconds"] = perf_counter() - start
    path = args.output / "metrics.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
