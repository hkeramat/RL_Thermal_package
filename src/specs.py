from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class AreaSpec:
    frac: float
    choices: List[str]

@dataclass
class LayerSpec:
    areas: List[AreaSpec]

LAYER_SPECS_BASE: List[LayerSpec] = [
    LayerSpec(areas=[AreaSpec(0.2, ["c1","c2","c3"]),
                     AreaSpec(0.7, ["c2","c4"]),
                     AreaSpec(0.1, ["d2","d4"])]),
    LayerSpec(areas=[AreaSpec(0.2, ["c2","c4"]),
                     AreaSpec(0.8, ["d1","d2"])]),
    LayerSpec(areas=[AreaSpec(0.3, ["c1","c3","c5"]),
                     AreaSpec(0.1, ["c1","c2","c3"]),
                     AreaSpec(0.6, ["d1","d3","d4"])])
]

def build_layer_specs(num_layers: Optional[int]) -> List[LayerSpec]:
    base = LAYER_SPECS_BASE
    if num_layers is None:
        return base
    if num_layers <= len(base):
        return base[:num_layers]
    out = list(base)
    i = 0
    while len(out) < num_layers:
        out.append(base[i % len(base)])
        i += 1
    return out
