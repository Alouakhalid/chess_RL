
from __future__ import annotations
from pathlib import Path
from typing import Any
import sys
import gymnasium as gym
import numpy as np
sys.modules.setdefault("tensorflow", None)
from stable_baselines3 import A2C, DDPG, DQN, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO, QRDQN
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from chess_env import ChessEnv, ContinuousLegalMoveWrapper, LegalMoveIndexWrapper
from model_support import PROJECT_ROOT, action_mask, normalise_model_name, predict_legal_move
def mask_fn(env: gym.Env) -> np.ndarray:
    return action_mask(env)
def make_chess_env(
    model_type: str = "ppo",
    *,
    mode: str = "agent_vs_opponent",
    player_color: str = "random",
    opponent_type: str = "heuristic",
    max_steps: int = 200,
) -> gym.Env:
    name = normalise_model_name(model_type)
    env: gym.Env = ChessEnv(
        mode=mode,
        player_color=player_color,
        opponent_type=opponent_type,
        max_steps=max_steps,
    )
    if name == "ppo":
        env = ActionMasker(env, mask_fn)
    elif name in {"a2c", "dqn", "ddqn"}:
        env = LegalMoveIndexWrapper(env)
    else:
        env = ContinuousLegalMoveWrapper(env)
    return Monitor(env)
DiscreteToContinuousWrapper = ContinuousLegalMoveWrapper
def _paths(model_name: str, save_dir: str | Path | None) -> tuple[Path, Path]:
    name = normalise_model_name(model_name)
    directory = Path(save_dir) if save_dir else PROJECT_ROOT / "models" / name
    directory.mkdir(parents=True, exist_ok=True)
    best_directory = directory / "best_model"
    best_directory.mkdir(parents=True, exist_ok=True)
    return directory, best_directory
def _train(
    model_name: str,
    total_timesteps: int,
    save_dir: str | Path | None = None,
    *,
    seed: int | None = None,
):
    name = normalise_model_name(model_name)
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    directory, best_directory = _paths(name, save_dir)
    env = DummyVecEnv([lambda: make_chess_env(name)])
    eval_env = make_chess_env(name, player_color="white")
    constructors: dict[str, Any] = {
        "ppo": lambda: MaskablePPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, seed=seed),
        "a2c": lambda: A2C("MlpPolicy", env, verbose=1, learning_rate=7e-4, seed=seed),
        "dqn": lambda: DQN("MlpPolicy", env, verbose=1, learning_rate=1e-4, buffer_size=20_000, learning_starts=1_000, seed=seed),
        "ddqn": lambda: QRDQN("MlpPolicy", env, verbose=1, learning_rate=1e-4, buffer_size=20_000, learning_starts=1_000, seed=seed),
        "ddpg": lambda: DDPG("MlpPolicy", env, verbose=1, learning_rate=1e-3, learning_starts=1_000, seed=seed),
        "sac": lambda: SAC("MlpPolicy", env, verbose=1, learning_rate=3e-4, learning_starts=1_000, seed=seed),
    }
    model = constructors[name]()
    eval_frequency = min(10_000, max(1_000, total_timesteps // 5))
    callback_type = MaskableEvalCallback if name == "ppo" else EvalCallback
    evaluation_callback = callback_type(
        eval_env,
        best_model_save_path=str(best_directory),
        log_path=str(directory),
        eval_freq=eval_frequency,
        n_eval_episodes=5,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_frequency,
        save_path=str(directory),
        name_prefix=f"{name}_checkpoint",
    )
    model.learn(total_timesteps=total_timesteps, callback=[evaluation_callback, checkpoint_callback])
    model.save(directory / f"{name}_final")
    return model
def train_ppo(total_timesteps: int = 100_000, save_dir: str | Path | None = None):
    return _train("ppo", total_timesteps, save_dir)
def train_a2c(total_timesteps: int = 100_000, save_dir: str | Path | None = None):
    return _train("a2c", total_timesteps, save_dir)
def train_dqn(total_timesteps: int = 100_000, save_dir: str | Path | None = None):
    return _train("dqn", total_timesteps, save_dir)
def train_ddqn(total_timesteps: int = 100_000, save_dir: str | Path | None = None):
    return _train("ddqn", total_timesteps, save_dir)
def train_ddpg(total_timesteps: int = 100_000, save_dir: str | Path | None = None):
    return _train("ddpg", total_timesteps, save_dir)
def train_sac(total_timesteps: int = 100_000, save_dir: str | Path | None = None):
    return _train("sac", total_timesteps, save_dir)
def render_trained_model(model_type: str = "ppo", model_path: str | Path | None = None, max_steps: int = 20) -> None:
    from model_support import load_model
    name = normalise_model_name(model_type)
    model = load_model(name, model_path)
    if model is None:
        raise FileNotFoundError(f"No {name.upper()} checkpoint could be loaded.")
    env = ChessEnv(mode="agent_vs_opponent", player_color="white", render_mode="human")
    observation, _ = env.reset()
    env.render()
    for step in range(max_steps):
        decision = predict_legal_move(model, name, observation, env)
        if decision.move is None:
            break
        observation, reward, terminated, truncated, info = env.step(decision.move)
        print(
            f"Step {step + 1}: {decision.move.uci()} "
            f"({decision.action_encoding}, reward={reward:.2f})"
        )
        env.render()
        if terminated or truncated:
            print("Game over:", env.board.outcome(claim_draw=True))
            break
if __name__ == "__main__":
    train_ppo(total_timesteps=100_000)
