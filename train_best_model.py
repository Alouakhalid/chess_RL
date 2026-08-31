import os
import sys
import time
import numpy as np

sys.modules['tensorflow'] = None
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gymnasium as gym
from chess_env import ChessEnv
from stable_baselines3 import PPO, A2C, DQN
from sb3_contrib import MaskablePPO, QRDQN
from sb3_contrib.common.wrappers import ActionMasker


def mask_fn(env: gym.Env) -> np.ndarray:
    if hasattr(env, 'get_action_mask'):
        return env.get_action_mask()
    return env.get_wrapper_attr('get_action_mask')()


def train_and_update_best_model(timesteps=30000):
    print("=" * 65)
    print("  TRAINING CHESS RL AGENTS TO MAXIMIZE REWARD (SELF-UPDATE)  ")
    print("=" * 65)

    env = ChessEnv(mode="ai_vs_ai", player_color="white", opponent_type="heuristic")
    env = ActionMasker(env, mask_fn)

    best_reward = -float('inf')
    best_model_name = "ppo"

    for model_name, cls in [("ppo", MaskablePPO), ("a2c", A2C), ("dqn", DQN), ("ddqn", QRDQN)]:
        print(f"\n--- Fine-tuning {model_name.upper()} ({timesteps} steps) ---")
        best_dir = f"./models/{model_name}/best_model"
        os.makedirs(best_dir, exist_ok=True)

        path = f"{best_dir}/best_model.zip"
        if os.path.exists(path):
            try:
                model = cls.load(path, env=env)
            except Exception:
                model = cls("MlpPolicy", env, verbose=0)
        else:
            model = cls("MlpPolicy", env, verbose=0)

        model.learn(total_timesteps=timesteps)
        model.save(f"{best_dir}/best_model.zip")
        print(f"Saved updated {model_name.upper()} model to {best_dir}/best_model.zip")

        test_rewards = []
        for _ in range(10):
            obs, info = env.reset()
            ep_rew = 0
            done = False
            while not done:
                if model_name == "ppo":
                    mask = env.get_wrapper_attr('action_masks')().reshape(1, -1)
                    act, _ = model.predict(obs, action_masks=mask, deterministic=True)
                else:
                    act, _ = model.predict(obs, deterministic=True)
                if hasattr(act, 'item'):
                    act = int(act.item())
                obs, reward, term, trunc, info = env.step(act)
                ep_rew += reward
                done = term or trunc
            test_rewards.append(ep_rew)

        avg_rew = float(np.mean(test_rewards))
        print(f"{model_name.upper()} Test Mean Reward: {avg_rew:.2f}")

        if avg_rew > best_reward:
            best_reward = avg_rew
            best_model_name = model_name

    print("\n" + "=" * 65)
    print(f"  BEST OVERALL MODEL: {best_model_name.upper()} (Reward: {best_reward:.2f})")
    print("=" * 65)

    with open("./models/best_model_info.txt", "w") as f:
        f.write(best_model_name)

    return best_model_name, best_reward


if __name__ == "__main__":
    train_and_update_best_model(timesteps=20000)
