from __future__ import annotations
import os, argparse, json, time
from typing import Dict, Any

import yaml
import numpy as np

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm

from .envs.thermal_stack_env import ThermalStackRLLibEnv
from .specs import build_layer_specs
from .utils import set_global_seeds

def _apply_runner_api(cfg: PPOConfig, num_rollout_workers: int, num_envs_per_worker: int) -> PPOConfig:
    """Set rollout workers/vectorization; compatible across RLlib versions."""
    if hasattr(cfg, "env_runners"):
        return cfg.env_runners(num_env_runners=num_rollout_workers,
                               num_envs_per_env_runner=num_envs_per_worker)
    else:
        return cfg.rollouts(num_rollout_workers=num_rollout_workers,
                            num_envs_per_worker=num_envs_per_worker)

def _apply_training_api(cfg: PPOConfig, **kwargs) -> PPOConfig:
    """Handle RLlib API differences (num_sgd_iter→num_epochs, sgd_minibatch_size→minibatch_size)."""
    try:
        return cfg.training(**kwargs)
    except TypeError:
        translated = dict(kwargs)
        if "minibatch_size" in translated:
            translated["sgd_minibatch_size"] = translated.pop("minibatch_size")
        if "num_epochs" in translated and "num_sgd_iter" not in translated:
            translated["num_sgd_iter"] = translated.pop("num_epochs")
        return cfg.training(**translated)

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def merge_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in overrides.items():
        sec, key = k.split(".", 1)
        if sec not in cfg:
            cfg[sec] = {}
        try:
            vC = json.loads(v)
        except Exception:
            vC = v
        if isinstance(cfg[sec], dict):
            cfg[sec][key] = vC
        else:
            cfg[sec] = vC
    return cfg

def make_env_config(env_cfg: Dict[str, Any]) -> Dict[str, Any]:
    layer_specs = build_layer_specs(env_cfg.get("num_layers_override"))
    return dict(
        layer_specs=layer_specs,
        t_min_global=env_cfg["t_min_global"],
        t_max_global=env_cfg["t_max_global"],
        thickness_bins=env_cfg["thickness_bins"],
        t_min_per_layer=env_cfg["t_min_per_layer"],
        T_total_min=env_cfg["T_total_min"],
        T_total_target=env_cfg["T_total_target"],
        lambda_cost=env_cfg["lambda_cost"],
        penalty_total_min=env_cfg["penalty_total_min"],
        penalty_total_target=env_cfg["penalty_total_target"],
        importance_ratio_k_to_cost=env_cfg["importance_ratio_k_to_cost"],
        cost_cap=env_cfg["cost_cap"],
        constraint_penalty=env_cfg["constraint_penalty"],
        mode="scalarized",
    )

def _maybe_init_wandb(cfg: Dict[str, Any], mode: str, run_id: str):
    wbcfg = cfg.get("wandb", {})
    if not wbcfg.get("enabled", False):
        return None
    try:
        import wandb
    except Exception as e:
        print(f"[W&B] import failed: {e}")
        return None
    name = wbcfg.get("run_name") or f"{run_id}-{mode}"
    group = wbcfg.get("group") or run_id
    project = wbcfg.get("project", "thermal-stack-rl")
    entity = wbcfg.get("entity")
    tags = wbcfg.get("tags", [])
    run = wandb.init(project=project, entity=entity, name=name, group=group, tags=tags,
                     config=cfg, reinit=True)
    return run

def _wandb_log(run, payload: Dict[str, Any], step: int | None = None):
    if run is None:
        return
    try:
        run.log(payload, step=step)
    except Exception as e:
        print(f"[W&B] log error: {e}")

def _wandb_artifact(run, path: str, name: str, type_: str = "artifact"):
    if run is None:
        return
    try:
        import wandb
        art = wandb.Artifact(name=name, type=type_)
        if os.path.isdir(path):
            art.add_dir(path)
        else:
            art.add_file(path)
        run.log_artifact(art)
    except Exception as e:
        print(f"[W&B] artifact error: {e}")

def train_once(config_path: str, overrides: Dict[str, str], run_id: str, sweep: bool = False):
    base = load_config(config_path) if config_path else {}
    cfg = merge_overrides(base, overrides)

    seed = int(cfg["run"].get("seed", 0))
    run_mode = cfg["run"].get("mode", "scalarized")

    artifacts_dir = os.path.join(cfg["run"].get("artifacts_dir", "artifacts"), run_id)
    os.makedirs(artifacts_dir, exist_ok=True)

    set_global_seeds(seed)

    # Init Ray (local or cluster)
    ray_addr = cfg.get("ray", {}).get("address", None)
    ray.init(address=ray_addr or None, ignore_reinit_error=True, include_dashboard=False, log_to_driver=True)

    # Register env
    env_name = "ThermalStack-v0"
    register_env(env_name, lambda env_config: ThermalStackRLLibEnv(env_config))

    # Shared env config
    env_config = make_env_config(cfg["env"])

    # RLlib config
    p = cfg["ppo"]
    r = cfg["ray"]

    algo_cfg = PPOConfig()
    algo_cfg = algo_cfg.framework("torch")
    algo_cfg = algo_cfg.environment(env=env_name, env_config=env_config)
    algo_cfg = _apply_runner_api(algo_cfg, r["num_rollout_workers"], r["num_envs_per_worker"])

    # Prefer new param names (num_epochs/minibatch_size), but support old ones.
    num_epochs = p.get("num_epochs", p.get("num_sgd_iter", 10))
    algo_cfg = _apply_training_api(
        algo_cfg,
        gamma=p["gamma"],
        lr=p["lr"],
        train_batch_size=p["train_batch_size"],
        minibatch_size=p["minibatch_size"],
        num_epochs=num_epochs,
        vf_clip_param=p["vf_clip_param"],
        model=p["model"],
    )
    algo_cfg = algo_cfg.resources(num_gpus=cfg["ray"]["num_gpus"])

    # ---- Evaluation config (always via .evaluation()) ----
    algo_cfg = algo_cfg.evaluation(
        evaluation_interval=r["evaluation_interval"],
        evaluation_duration=r["evaluation_duration"],
        evaluation_num_workers=r["evaluation_num_workers"],
        evaluation_parallel_to_training=r["evaluation_parallel_to_training"],
        evaluation_config={"env_config": dict(env_config, mode=run_mode), "explore": True},
    )

    algo_cfg = algo_cfg.rollouts(enable_connectors=False)
    algo_cfg = algo_cfg.debugging(seed=seed)

    regimes = [run_mode] if run_mode != "both" else ["scalarized", "constrained"]

    # Optional sweeps via Ray Tune
    if sweep:
        search_space = {}
        for sec in ("ppo", "env"):
            for k, v in cfg[sec].items():
                if isinstance(v, str) and "," in v:
                    vals = [json.loads(x) if _is_jsonable(x) else x for x in v.split(",")]
                    search_space[f"{sec}.{k}"] = tune.grid_search(vals)
        if search_space:
            tune_config = tune.TuneConfig(num_samples=1)
            tuner = tune.Tuner(
                trainable=_tune_trainable,
                param_space={"cfg": cfg, **search_space, "run_id": run_id},
                tune_config=tune_config,
            )
            tuner.fit()
            ray.shutdown()
            return

    results = {}
    for mode in regimes:
        mode_dir = os.path.join(artifacts_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)

        # Build algorithm per-mode
        local_cfg = algo_cfg.copy(copy_frozen=False)
        local_cfg.environment(env=env_name, env_config=dict(env_config, mode=mode))
        algo: Algorithm = local_cfg.build()

        # W&B run per mode
        wb_run = _maybe_init_wandb(cfg, mode=mode, run_id=run_id)

        iters = int(cfg["train"]["iterations"])
        save_every = int(cfg["train"]["save_every"])
        reward_curve = []

        for it in range(1, iters + 1):
            res = algo.train()
            rmean = float(res.get("episode_reward_mean", np.nan))
            reward_curve.append(rmean if not np.isnan(rmean) else 0.0)
            print(f"[{mode}] iter={it:03d} mean_reward={rmean:.6f}")

            # W&B logging
            payload = {
                "iter": it,
                "mode": mode,
                "mean_reward": rmean,
                "timesteps_total": res.get("timesteps_total"),
                "episodes_total": res.get("episodes_total"),
            }
            eval_res = res.get("evaluation") or {}
            if isinstance(eval_res, dict) and "episode_reward_mean" in eval_res:
                payload["eval/episode_reward_mean"] = eval_res.get("episode_reward_mean")
            _wandb_log(wb_run, payload, step=it)

            # Save checkpoints periodically
            if it % save_every == 0 or it == iters:
                ckpt = algo.save(mode_dir)
                ckpt_path = ckpt.checkpoint.path
                with open(os.path.join(mode_dir, "last_checkpoint"), "w") as f:
                    f.write(ckpt_path)
                _wandb_artifact(wb_run, ckpt_path, name=f"{run_id}-{mode}-ckpt-{it}", type_="model")

        # Persist curves
        reward_curve_path = os.path.join(mode_dir, "reward_curve.json")
        with open(reward_curve_path, "w") as f:
            json.dump(reward_curve, f)
        _wandb_artifact(wb_run, reward_curve_path, name=f"{run_id}-{mode}-curves")

        results[mode] = {
            "last_checkpoint": os.path.join(mode_dir, "last_checkpoint"),
            "reward_curve": reward_curve,
        }

        # Close W&B
        if wb_run is not None:
            try:
                wb_run.finish()
            except Exception:
                pass

        algo.stop()

    with open(os.path.join(artifacts_dir, "summary.json"), "w") as f:
        json.dump({"results": results, "cfg": cfg}, f, indent=2)

    ray.shutdown()

def _is_jsonable(x: str) -> bool:
    try:
        json.loads(x)
        return True
    except Exception:
        return False

def _tune_trainable(config: Dict[str, Any]):
    cfg = config["cfg"]
    run_id = config.get("run_id", f"tune-{int(time.time())}")
    for k, v in list(config.items()):
        if k in ("cfg", "run_id"):
            continue
        sec, key = k.split(".", 1)
        cfg[sec][key] = v
    train_once(config_path=None, overrides={}, run_id=run_id, sweep=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf/default.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("overrides", nargs="*", help="section.key=value overrides")
    args = parser.parse_args()

    overrides = {}
    for ov in (args.overrides or []):
        k, v = ov.split("=", 1)
        overrides[k] = v

    run_id = args.run_id or f"run-{int(time.time())}"

    train_once(config_path=args.config, overrides=overrides, run_id=run_id, sweep=args.sweep)
