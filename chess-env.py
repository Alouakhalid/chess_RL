
import numpy as np
from chess_env import ChessEnv
if __name__ == "__main__":
    env = ChessEnv(mode="agent_vs_opponent", player_color="white")
    observation, _ = env.reset(seed=7)
    for step in range(10):
        actions = np.flatnonzero(env.get_action_mask())
        observation, reward, terminated, truncated, info = env.step(int(actions[0]))
        print(f"Step {step + 1}: {info['last_move']} / {info['last_opponent_move']} ({reward:.2f})")
        if terminated or truncated:
            break
