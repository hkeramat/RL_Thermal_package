from __future__ import annotations
from typing import List
import os, random, numpy as np

def mixedradix_bases(max_choices_per_pos: List[int], thickness_bins: int) -> List[int]:
    return [thickness_bins] + max_choices_per_pos

def decode_action(a: int, bases: List[int]) -> List[int]:
    digits = []
    for b in bases:
        digits.append(a % b)
        a //= b
    return digits

def action_space_size(bases: List[int]) -> int:
    n = 1
    for b in bases:
        n *= b
    return n

def set_global_seeds(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
