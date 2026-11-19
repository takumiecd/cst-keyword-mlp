# CST Keyword-MLP

English | [日本語](README.ja.md)

Train the [official Keyword-MLP](https://github.com/AI-Research-BD/Keyword-MLP/tree/a87c8ac9c4d3ba01745d68cc5dcf9cd21dd67b72)
and a CST variant from scratch on Speech Commands V2-35. The CST variant
replaces the 98-to-98 temporal projection in each of the 12 spatial gating
units. The original kernel-size-one convolution applies one temporal weight
matrix per feature channel. The CST variant retains the bias initialized to
one and matches the small initial weight scale. All other layers use the
official architecture.
The vendored model is covered by the [upstream license](upstream/LICENSE).

## RTX 4090 pilot

| Model | Median step | Peak allocated VRAM | Peak reserved VRAM | Parameters |
| --- | ---: | ---: | ---: | ---: |
| Original KW-MLP | 8.79 ms | 1,234.2 MiB | 1,460 MiB | 424,811 |
| CST, 128 Atoms per projection | 28.41 ms | 1,239.8 MiB | 1,470 MiB | 315,707 |

Batch size 256, seed 17, FP32 with TF32 enabled, PyTorch 2.6.0 and CUDA 12.6.
Each model ran in two fresh processes with 15 warmup updates and 60 measured
updates each. The median pools 120 measurements. VRAM is the maximum of
the two rounds using PyTorch allocator statistics. Timing covers the full
optimizer update on an already prepared GPU batch, including forward,
backward, and gradient clearing. Both models used the same examples and the
same initial values for shared parameters. Each also completed a separate
one-epoch run that saved and reloaded the validation-selected checkpoint
and evaluated it on the test split.

## Prepare and run

Use Python 3.10 or newer and a matching CUDA-enabled PyTorch/torchaudio pair.
The cluster pilot used PyTorch and torchaudio 2.6.0+cu126.

```bash
python -m pip install -r requirements.txt
mkdir -p data/speech_commands_v0.02
curl -fL https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz \
  -o data/speech_commands_v0.02.tar.gz
tar -xzf data/speech_commands_v0.02.tar.gz -C data/speech_commands_v0.02
python data.py --data data/speech_commands_v0.02 --cache data/scv2-mfcc
```

The feature cache uses the official train, validation, and test split (84,843,
9,981, and 11,005 clips). The 40-by-98 MFCC frontend uses a 30 ms window and a
10 ms hop. SpecAugment is shared across both training variants.

```bash
# Short measurement
for variant in baseline cst; do
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python train.py --variant "$variant" \
    --cache data/scv2-mfcc --output "output/pilot-$variant" \
    --batch-size 256 --seed 17 --atoms 128 --pilot-warmup 15 --pilot-updates 60
done

# Training from scratch
for variant in baseline cst; do
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python train.py --variant "$variant" \
    --cache data/scv2-mfcc --output "output/full-$variant" \
    --batch-size 256 --seed 17 --atoms 128 --epochs 140
done
```

The full run uses 10 epochs of learning-rate warmup and cosine decay. Ordinary
parameters start from a base learning rate of 0.001 and the CST Atom group
uses 0.0048. The CST backend defaults to `materialized`. Each run writes
`metrics.json`, and full training also writes the best validation checkpoint
to `best.pt`.
