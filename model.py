"""Original KW-MLP and its CST temporal projection variant."""

from __future__ import annotations

import hashlib

from torch import nn
from torchcst import CSTLinear, ChebyshevSpec, RadiusBounds, chart_presets, init, kernels

from upstream.kwmlp import KW_MLP, SpatialGatingUnit


class CSTTemporalProjection(nn.Module):
    def __init__(self, original: nn.Conv1d, atoms: int, backend: str):
        super().__init__()
        frames = original.in_channels
        # Frame coordinates are in units of the 10 ms feature hop.
        chart = chart_presets.regular_grid(frames, spacing=1.0)
        kernel = kernels.product(
            kernels.cell_mass(ChebyshevSpec(modes=8)),
            coordinates=kernels.polar(
                amplitude_max=1.0,
                w_c=0.1,
                bounds=RadiusBounds(
                    minimum=0.5, birth=3.5, maximum=6.0, upper_floor=0.5
                ),
            ),
        )
        self.linear = CSTLinear(charts=(chart, chart), atoms=atoms, kernel=kernel, backend=backend)
        init.variance_scaling_(self.linear, (1e-3 / frames) ** 2 / 3, q=0.25)
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
