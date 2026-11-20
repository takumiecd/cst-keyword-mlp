"""Original KW-MLP and its CST temporal projection variant."""

from __future__ import annotations

import hashlib

import torch
from torch import nn
from torchcst import BandwidthBounds, CSTLinear, TriweightSpec, chart_presets, kernels

from upstream.kwmlp import KW_MLP, SpatialGatingUnit


class CSTTemporalProjection(nn.Module):
    def __init__(self, original: nn.Conv1d, atoms: int, backend: str):
        super().__init__()
        frames = original.in_channels
        # Frame coordinates are in units of the 10 ms feature hop.
        chart = chart_presets.regular_grid(frames, spacing=1.0)
        kernel = kernels.product(
            kernels.l2(TriweightSpec()),
            coordinates=kernels.polar(
                amplitude_max=1.0,
                w_c=0.1,
                bounds=BandwidthBounds(
                    minimum=0.5, birth=3.5, maximum=6.0, upper_floor=0.5
                ),
            ),
        )
        self.linear = CSTLinear(charts=(chart, chart), atoms=atoms, kernel=kernel, backend=backend)
        with torch.no_grad():
            # Match the small initial gate weights while retaining the Polar radius.
            target_rms = original.weight.square().mean().sqrt()
            current_rms = self.linear.operator.weight().square().mean().sqrt()
            polar = self.linear.atoms.p[:, :2]
            radius = polar.norm(dim=1)
            direction = (polar[:, 0] / radius * (target_rms / current_rms)).clamp(
                -1 + 1e-6, 1 - 1e-6
            )
            polar[:, 0] = radius * direction
            polar[:, 1] = radius * (1 - direction.square()).sqrt()
        self.bias = original.bias

    def forward(self, gate):
        return (self.linear(gate.transpose(1, 2)) + self.bias).transpose(1, 2)


def build_model(variant, atoms=128, backend="materialized"):
    model = KW_MLP()
    if variant == "cst":
        for module in list(model.modules()):
            if isinstance(module, SpatialGatingUnit):
                module.proj = CSTTemporalProjection(module.proj, atoms, backend)
    return model


def shared_initial_sha256(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if ".sgu.proj." in name and not name.endswith(".sgu.proj.bias"):
            continue
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
