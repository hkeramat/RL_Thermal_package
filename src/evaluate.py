from __future__ import annotations
import os, argparse, json
from typing import Dict, Any

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

from .envs.thermal_stack_env import ThermalStackRLLibEnv
from .train import make_env_config, load_config

def greedy_rollout(algorithm: Algorithm, env_config: Dict[str, Any], episodes: int = 1):
    env = ThermalStackRLLibEnv(env_config)
    reports = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        total_r = 0.0
        info = {}
        while not done:
            action = algorithm.compute_single_action(obs, explore=False)
            obs, r, done, trunc, info = env.step(action)
            total_r += r
        reports.append({"total_reward": float(total_r), "info": info})
    return reports

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="conf/default.yaml")
    parser.add_argument("--mode", default="scalarized", choices=["scalarized", "constrained"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_config = make_env_config(cfg["env"])
    env_config["mode"] = args.mode

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    register_env("ThermalStack-v0", lambda env_config: ThermalStackRLLibEnv(env_config))
    algo = Algorithm.from_checkpoint(args.checkpoint)

    reps = greedy_rollout(algo, env_config, episodes=3)
    out_dir = os.path.dirname(args.checkpoint)
    with open(os.path.join(out_dir, f"eval_{args.mode}.json"), "w") as f:
        json.dump(reps, f, indent=2)

    print(json.dumps(reps, indent=2))

    algo.stop()
    ray.shutdown()
