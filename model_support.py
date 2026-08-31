
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional
import sys
import warnings
import chess
import numpy as np
from gymnasium import spaces
sys.modules.setdefault("tensorflow", None)
from stable_baselines3 import A2C, DDPG, DQN, SAC
from sb3_contrib import MaskablePPO, QRDQN
from chess_env import BuiltInOpponent, ChessEnv, MAX_LEGAL_MOVES
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_CLASSES = {
    "ppo": MaskablePPO,
    "a2c": A2C,
    "dqn": DQN,
    "ddqn": QRDQN,
    "ddpg": DDPG,
    "sac": SAC,
}
@dataclass(frozen=True)
class ModelDecision:
    raw_action: Any
    move: Optional[chess.Move]
    action_encoding: str
    was_illegal: bool = False
    was_aliased: bool = False
    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["move"] = self.move.uci() if self.move else None
        return result
def normalise_model_name(model_name: str) -> str:
    name = model_name.lower().strip()
    if name not in MODEL_CLASSES:
        choices = ", ".join(sorted(MODEL_CLASSES))
        raise ValueError(f"Unknown model '{model_name}'. Choose one of: {choices}.")
    return name
def checkpoint_path(model_name: str) -> Optional[Path]:
    name = normalise_model_name(model_name)
    candidates = (
        PROJECT_ROOT / "models" / name / "best_model" / "best_model.zip",
        PROJECT_ROOT / "models" / name / f"{name}_final.zip",
    )
    return next((path for path in candidates if path.exists()), None)
def load_model(model_name: str, model_path: Optional[str | Path] = None):
    name = normalise_model_name(model_name)
    path = Path(model_path) if model_path is not None else checkpoint_path(name)
    if path is None or not path.exists():
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This system does not have apparently enough memory.*")
        return MODEL_CLASSES[name].load(path)
def action_mask(env: Any) -> np.ndarray:
    if hasattr(env, "action_masks"):
        return np.asarray(env.action_masks(), dtype=bool)
    if hasattr(env, "get_action_mask"):
        return np.asarray(env.get_action_mask(), dtype=bool)
    if hasattr(env, "get_wrapper_attr"):
        return np.asarray(env.get_wrapper_attr("action_masks")(), dtype=bool)
    raise TypeError("Environment does not expose an action mask.")
def tactical_fallback(board: chess.Board) -> Optional[chess.Move]:
    return BuiltInOpponent(mode="heuristic").choose_move(board)
def _scalar_action(action: Any) -> Any:
    array = np.asarray(action)
    if array.size == 1:
        return array.reshape(-1)[0].item()
    return array.tolist()
def _move_from_compact_index(board: chess.Board, index: int) -> tuple[Optional[chess.Move], bool]:
    legal_moves = sorted(board.legal_moves, key=lambda candidate: candidate.uci())
    if not legal_moves:
        return None, False
    return legal_moves[index % len(legal_moves)], not 0 <= index < len(legal_moves)
def predict_legal_move(
    model: Any,
    model_name: str,
    observation: np.ndarray,
    env: ChessEnv,
    *,
    deterministic: bool = True,
) -> ModelDecision:
    name = normalise_model_name(model_name)
    board = env.board
    if board.is_game_over(claim_draw=True):
        return ModelDecision(raw_action=None, move=None, action_encoding="game_over")
    if model is None:
        return ModelDecision(
            raw_action=None,
            move=tactical_fallback(board),
            action_encoding="heuristic_fallback",
        )
    model_space = getattr(model, "action_space", None)
    if name == "ppo" and isinstance(model_space, spaces.Discrete):
        raw_action, _ = model.predict(
            observation,
            action_masks=action_mask(env).reshape(1, -1),
            deterministic=deterministic,
        )
    else:
        raw_action, _ = model.predict(observation, deterministic=deterministic)
    value = _scalar_action(raw_action)
    if isinstance(model_space, spaces.Box):
        try:
            continuous_value = float(np.asarray(raw_action).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            continuous_value = 0.0
        legal_moves = sorted(board.legal_moves, key=lambda candidate: candidate.uci())
        if not legal_moves:
            return ModelDecision(value, None, "continuous_legal_index")
        clipped = float(np.clip(continuous_value, -1.0, 1.0))
        index = min(int((clipped + 1.0) * 0.5 * len(legal_moves)), len(legal_moves) - 1)
        return ModelDecision(value, legal_moves[index], "continuous_legal_index")
    try:
        discrete_value = int(value)
    except (TypeError, ValueError, OverflowError):
        return ModelDecision(value, tactical_fallback(board), "invalid_model_output", was_illegal=True)
    if isinstance(model_space, spaces.Discrete) and model_space.n == MAX_LEGAL_MOVES:
        move, aliased = _move_from_compact_index(board, discrete_value)
        return ModelDecision(value, move, "compact_legal_index", was_aliased=aliased)
    move = env.parse_action(discrete_value)
    if move is not None:
        return ModelDecision(value, move, "legacy_4096")
    return ModelDecision(
        value,
        tactical_fallback(board),
        "legacy_4096",
        was_illegal=True,
    )
def legal_move_from_decision(env: ChessEnv, decision: ModelDecision) -> chess.Move:
    if decision.move is None:
        raise RuntimeError("No legal move is available in this position.")
    if decision.move not in env.board.legal_moves:
        raise RuntimeError("Move selection produced a non-legal move.")
    return decision.move
