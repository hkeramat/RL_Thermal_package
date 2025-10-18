from __future__ import annotations
from typing import Sequence, Tuple, List
import numpy as np
from .specs import LayerSpec
from .materials import MATERIALS

def k_eff_parallel(k: Sequence[float], areas: Sequence[float], t: float) -> float:
    if t <= 0:
        raise ValueError("t must be > 0")
    G_tot = sum(ki * ai / t for ki, ai in zip(k, areas))
    A_tot = sum(areas)
    return (G_tot * t) / A_tot

def k_eff_series_from_layers(layer_keffs: Sequence[float], ts: Sequence[float]) -> float:
    num = sum(ts)
    den = sum(ti / ki for ti, ki in zip(ts, layer_keffs))
    return num / den

def layer_keff_and_cost(layer: LayerSpec, mat_ids: List[str], t: float, A_total: float = 1.0) -> Tuple[float, float]:
    ks, As, cost = [], [], 0.0
    for area, m in zip(layer.areas, mat_ids):
        ks.append(MATERIALS[m]["k"])
        Ai = area.frac * A_total
        As.append(Ai)
        cost += MATERIALS[m]["cost"] * Ai * t
    return k_eff_parallel(ks, As, t), cost
