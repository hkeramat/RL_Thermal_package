from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..specs import build_layer_specs, LayerSpec
from ..physics import k_eff_series_from_layers, layer_keff_and_cost
from ..utils import mixedradix_bases, decode_action, action_space_size

T_MIN_GLOBAL_DEFAULT   = 5e-5
T_MAX_GLOBAL_DEFAULT   = 5e-3
THICKNESS_BINS_DEFAULT = 21
IMPORTANCE_RATIO_DEFAULT = 1.0
COST_CAP_DEFAULT = 2.0
CONSTRAINT_PENALTY_DEFAULT = 1e1

class ThermalStackRLLibEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, env_config: Dict = None):
        cfg = dict(env_config or {})
        self.mode: str = cfg.get("mode", "scalarized")
        assert self.mode in ("scalarized", "constrained")

        self.layer_specs: List[LayerSpec] = cfg.get("layer_specs", build_layer_specs(None))
        self.L = len(self.layer_specs)

        self.t_min_global   = float(cfg.get("t_min_global", T_MIN_GLOBAL_DEFAULT))
        self.t_max_global   = float(cfg.get("t_max_global", T_MAX_GLOBAL_DEFAULT))
        self.thickness_bins = int(cfg.get("thickness_bins", THICKNESS_BINS_DEFAULT))
        assert self.thickness_bins >= 2
        self.t_grid = np.linspace(self.t_min_global, self.t_max_global, self.thickness_bins)

        default_per_layer = [self.t_min_global] * self.L
        self.t_min_per_layer: List[float] = list(cfg.get("t_min_per_layer", default_per_layer))
        if len(self.t_min_per_layer) != self.L:
            self.t_min_per_layer = default_per_layer

        self.T_total_min: Optional[float]    = cfg.get("T_total_min", None)
        self.T_total_target: Optional[float] = cfg.get("T_total_target", None)
        self.lambda_cost = float(cfg.get("lambda_cost", 5.0e-3))
        self.penalty_total_min = float(cfg.get("penalty_total_min", 5.0e-6))
        self.penalty_total_target = float(cfg.get("penalty_total_target", 5.0e-6))
        ratio = float(cfg.get("importance_ratio_k_to_cost", IMPORTANCE_RATIO_DEFAULT))
        ratio = max(ratio, 0.0)
        denom = 1.0 + ratio
        self.w_k = ratio / denom
        self.w_cost = 1.0 / denom
        self.cost_cap = float(cfg.get("cost_cap", COST_CAP_DEFAULT))
        self.constraint_penalty = float(cfg.get("constraint_penalty", CONSTRAINT_PENALTY_DEFAULT))

        max_areas = max(len(l.areas) for l in self.layer_specs)
        per_pos_max: List[int] = []
        for pos in range(max_areas):
            mx = 1
            for l in self.layer_specs:
                if pos < len(l.areas):
                    mx = max(mx, len(l.areas[pos].choices))
            per_pos_max.append(mx)

        self.bases = mixedradix_bases(per_pos_max, self.thickness_bins)
        self.n_actions = action_space_size(self.bases)
        self.action_space = spaces.Discrete(self.n_actions)

        high = np.array([1e3, 1e3, 1e12, 1e12, 1e3, 1e3], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.reset()

    def _obs(self):
        remaining_t = (self.T_total_target if self.T_total_target is not None else
                       (self.T_total_min if self.T_total_min is not None else 0.0)) - self.sum_t
        return np.array([float(self.step_idx), self.sum_t, self.sum_cost, self.partial_den,
                         remaining_t, self.last_k_layer], dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.step_idx = 0
        self.sum_t = 0.0
        self.sum_cost = 0.0
        self.partial_den = 0.0
        self.layer_keffs: List[float] = []
        self.ts: List[float] = []
        self.last_k_layer = 0.0
        self.last_info = {}
        self.per_layer_log = []
        return self._obs(), {}

    def step(self, action: int):
        digits = decode_action(int(action), self.bases)
        thick_idx = digits[0]
        t = float(self.t_grid[thick_idx])

        layer_idx = self.step_idx
        t = max(t, float(self.t_min_per_layer[layer_idx]))

        layer = self.layer_specs[layer_idx]
        mat_ids: List[str] = []
        for pos, area in enumerate(layer.areas):
            idx = digits[1 + pos] % len(area.choices)
            mat_ids.append(area.choices[idx])

        k_layer, c_layer = layer_keff_and_cost(layer, mat_ids, t)

        self.per_layer_log.append((layer_idx + 1, mat_ids, t, k_layer, c_layer))

        self.layer_keffs.append(k_layer)
        self.ts.append(t)
        self.sum_t += t
        self.sum_cost += c_layer
        self.partial_den += t / k_layer
        self.last_k_layer = k_layer

        self.step_idx += 1
        terminated = self.step_idx >= self.L
        truncated = False

        reward = 0.0
        if terminated:
            k_series = k_eff_series_from_layers(self.layer_keffs, self.ts)
            if self.mode == "scalarized":
                reward = (self.w_k * k_series) - (self.w_cost * self.sum_cost)
            else:
                over = max(0.0, self.sum_cost - self.cost_cap)
                reward = k_series - self.constraint_penalty * over

            if self.T_total_target is not None:
                dev = abs(self.sum_t - float(self.T_total_target))
                reward -= self.penalty_total_target * dev
            elif self.T_total_min is not None:
                deficit = max(0.0, float(self.T_total_min) - self.sum_t)
                reward -= self.penalty_total_min * deficit

            self.last_info = {
                "k_series": float(k_series),
                "sum_cost": float(self.sum_cost),
                "sum_t": float(self.sum_t),
                "layer_keffs": np.array(self.layer_keffs, float),
                "ts": np.array(self.ts, float),
                "per_layer_log": list(self.per_layer_log),
                "reward": float(reward),
                "mode": self.mode,
                "w_k": float(self.w_k),
                "w_cost": float(self.w_cost),
                "cost_cap": float(self.cost_cap),
            }

        return self._obs(), float(reward), terminated, truncated, self.last_info
