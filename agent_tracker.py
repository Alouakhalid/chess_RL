
from __future__ import annotations
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import chess
import gymnasium as gym
import numpy as np
sys.modules.setdefault("tensorflow", None)
from stable_baselines3 import A2C, DDPG, DQN, SAC
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO, QRDQN
from sb3_contrib.common.wrappers import ActionMasker
from chess_env import (
    ChessEnv,
    ContinuousLegalMoveWrapper,
    LegalMoveIndexWrapper,
    PIECE_VALUES,
)
from model_support import (
    PROJECT_ROOT,
    ModelDecision,
    action_mask,
    checkpoint_path,
    load_model,
    normalise_model_name,
    predict_legal_move,
)
TRACKING_DIR = PROJECT_ROOT / "tracking"
def mask_fn(env: gym.Env) -> np.ndarray:
    return action_mask(env)
def captured_value(board: chess.Board, move: chess.Move) -> float:
    if not board.is_capture(move):
        return 0.0
    captured_piece = board.piece_at(move.to_square)
    if captured_piece is None and board.is_en_passant(move):
        return PIECE_VALUES[chess.PAWN]
    return PIECE_VALUES.get(captured_piece.piece_type, 0.0) if captured_piece else 0.0
def move_profile(board: chess.Board, move: chess.Move, perspective: chess.Color) -> dict[str, float]:
    capture = captured_value(board, move)
    material_before = ChessEnv._calculate_material_balance(board, perspective)
    board.push(move)
    material_after = ChessEnv._calculate_material_balance(board, perspective)
    gives_check = float(board.is_check())
    centre = float(move.to_square in {chess.D4, chess.E4, chess.D5, chess.E5})
    opponent_capture = max(
        (captured_value(board, reply) for reply in board.legal_moves if board.is_capture(reply)),
        default=0.0,
    )
    board.pop()
    score = (material_after - material_before) + 0.20 * gives_check + 0.05 * centre - 1.10 * opponent_capture
    return {
        "score": round(score, 4),
        "capture_value": capture,
        "opponent_best_capture": opponent_capture,
    }
@dataclass
class MistakeRecord:
    timestamp: str
    model_name: str
    episode: int
    ply: int
    fen: str
    turn: str
    raw_action: Any
    action_encoding: str
    selected_move: Optional[str]
    best_move: Optional[str]
    issue_types: list[str]
    selected_score: Optional[float]
    best_score: Optional[float]
    score_gap: Optional[float]
    selected_capture_value: Optional[float]
    best_capture_value: Optional[float]
    opponent_best_capture: Optional[float]
    legal_move_count: int
class MistakeJournal:
    def __init__(self, model_name: str, directory: Path = TRACKING_DIR):
        self.model_name = model_name
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{model_name}_mistakes.jsonl"
    def append(self, record: MistakeRecord) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(asdict(record), sort_keys=True) + "\n")
def recent_mistake_fens(model_name: str, limit: int = 250) -> list[str]:
    path = TRACKING_DIR / f"{normalise_model_name(model_name)}_mistakes.jsonl"
    if not path.exists():
        return []
    positions: list[str] = []
    seen: set[str] = set()
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        try:
            fen = json.loads(line).get("fen")
            board = chess.Board(fen)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if fen not in seen and board.is_valid() and not board.is_game_over(claim_draw=True):
            positions.append(fen)
            seen.add(fen)
        if len(positions) >= limit:
            break
    return positions
class ModelErrorTracker:
    def __init__(self, model_name: str = "ppo", model_path: Optional[str | Path] = None):
        self.model_name = normalise_model_name(model_name)
        self.model_path = Path(model_path) if model_path else checkpoint_path(self.model_name)
        self.model = load_model(self.model_name, self.model_path)
        self.journal = MistakeJournal(self.model_name)
    def _analyse_decision(
        self,
        board: chess.Board,
        decision: ModelDecision,
        *,
        episode: int,
        ply: int,
    ) -> Optional[MistakeRecord]:
        legal_moves = sorted(board.legal_moves, key=lambda move: move.uci())
        if not legal_moves or decision.move is None:
            return None
        perspective = board.turn
        profiles = {move: move_profile(board, move, perspective) for move in legal_moves}
        selected = profiles[decision.move]
        best_move = max(legal_moves, key=lambda move: profiles[move]["score"])
        best = profiles[best_move]
        score_gap = best["score"] - selected["score"]
        issues: list[str] = []
        if decision.was_illegal:
            issues.append("illegal_action_recovered")
        if decision.was_aliased:
            issues.append("compact_action_aliased")
        if best["capture_value"] >= selected["capture_value"] + 1.0:
            issues.append("missed_material_capture")
        if selected["opponent_best_capture"] >= 3.0:
            issues.append("hung_major_or_minor_piece")
        if score_gap >= 1.25:
            issues.append("tactical_inaccuracy")
        if not issues:
            return None
        return MistakeRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_name=self.model_name,
            episode=episode,
            ply=ply,
            fen=board.fen(),
            turn="white" if board.turn == chess.WHITE else "black",
            raw_action=decision.raw_action,
            action_encoding=decision.action_encoding,
            selected_move=decision.move.uci(),
            best_move=best_move.uci(),
            issue_types=issues,
            selected_score=selected["score"],
            best_score=best["score"],
            score_gap=round(score_gap, 4),
            selected_capture_value=selected["capture_value"],
            best_capture_value=best["capture_value"],
            opponent_best_capture=selected["opponent_best_capture"],
            legal_move_count=len(legal_moves),
        )
    def analyze_game(
        self,
        num_episodes: int = 5,
        max_steps: int = 200,
        *,
        model: Any = None,
        persist: bool = True,
    ) -> dict[str, Any]:
        active_model = self.model if model is None else model
        if active_model is None:
            return {"error": f"No {self.model_name.upper()} checkpoint was found."}
        if num_episodes <= 0:
            raise ValueError("num_episodes must be positive")
        env = ChessEnv(
            mode="agent_vs_opponent",
            player_color="white",
            opponent_type="heuristic",
            max_steps=max_steps,
        )
        counts = {
            "illegal_attempts": 0,
            "aliased_actions": 0,
            "blunders": 0,
            "missed_captures": 0,
            "tactical_inaccuracies": 0,
        }
        total_moves = 0
        total_reward = 0.0
        mistakes_written = 0
        for episode in range(num_episodes):
            observation, _ = env.reset(seed=episode)
            done = False
            while not done:
                decision = predict_legal_move(active_model, self.model_name, observation, env)
                record = self._analyse_decision(env.board, decision, episode=episode, ply=total_moves + 1)
                if record:
                    if "illegal_action_recovered" in record.issue_types:
                        counts["illegal_attempts"] += 1
                    if "compact_action_aliased" in record.issue_types:
                        counts["aliased_actions"] += 1
                    if "hung_major_or_minor_piece" in record.issue_types:
                        counts["blunders"] += 1
                    if "missed_material_capture" in record.issue_types:
                        counts["missed_captures"] += 1
                    if "tactical_inaccuracy" in record.issue_types:
                        counts["tactical_inaccuracies"] += 1
                    if persist:
                        self.journal.append(record)
                        mistakes_written += 1
                if decision.move is None:
                    break
                observation, reward, terminated, truncated, _ = env.step(decision.move)
                total_reward += reward
                total_moves += 1
                done = terminated or truncated
        error_total = (
            counts["illegal_attempts"] * 3.0
            + counts["aliased_actions"] * 0.5
            + counts["blunders"] * 1.5
            + counts["missed_captures"]
            + counts["tactical_inaccuracies"] * 0.25
        )
        return {
            "model_name": self.model_name,
            "episodes": num_episodes,
            "total_moves": total_moves,
            **counts,
            "mean_reward": total_reward / max(1, num_episodes),
            "mistakes_written": mistakes_written,
            "mistake_log": str(self.journal.path),
            "quality_score": (total_reward / max(1, num_episodes)) - error_total / max(1, total_moves),
        }
def make_training_env(model_name: str, *, max_steps: int = 200) -> gym.Env:
    name = normalise_model_name(model_name)
    replay_fens = recent_mistake_fens(name)
    env: gym.Env = ChessEnv(
        mode="agent_vs_opponent",
        player_color="random",
        opponent_type="heuristic",
        max_steps=max_steps,
        replay_fens=replay_fens,
        replay_probability=0.35 if replay_fens else 0.0,
    )
    if name == "ppo":
        env = ActionMasker(env, mask_fn)
    elif name in {"ddpg", "sac"}:
        env = ContinuousLegalMoveWrapper(env)
    else:
        env = LegalMoveIndexWrapper(env)
    return Monitor(env)
def build_or_reuse_model(model_name: str, env: gym.Env, existing: Any = None):
    name = normalise_model_name(model_name)
    if existing is not None and existing.action_space == env.action_space:
        existing.set_env(env)
        return existing
    constructors = {
        "ppo": lambda: MaskablePPO("MlpPolicy", env, verbose=0, learning_rate=3e-4),
        "a2c": lambda: A2C("MlpPolicy", env, verbose=0, learning_rate=7e-4),
        "dqn": lambda: DQN("MlpPolicy", env, verbose=0, learning_rate=1e-4, buffer_size=20_000, learning_starts=1_000),
        "ddqn": lambda: QRDQN("MlpPolicy", env, verbose=0, learning_rate=1e-4, buffer_size=20_000, learning_starts=1_000),
        "ddpg": lambda: DDPG("MlpPolicy", env, verbose=0, learning_rate=1e-3, learning_starts=1_000),
        "sac": lambda: SAC("MlpPolicy", env, verbose=0, learning_rate=3e-4, learning_starts=1_000),
    }
    return constructors[name]()
class SelfUpdatingAgent:
    def __init__(self, model_name: str = "ppo"):
        self.model_name = normalise_model_name(model_name)
        self.tracker = ModelErrorTracker(self.model_name)
    def update_model(self, timesteps: int = 25_000, evaluation_episodes: int = 8) -> tuple[dict[str, Any], dict[str, Any]]:
        if timesteps <= 0:
            raise ValueError("timesteps must be positive")
        before = self.tracker.analyze_game(num_episodes=evaluation_episodes, persist=True)
        env = make_training_env(self.model_name)
        candidate = build_or_reuse_model(self.model_name, env, self.tracker.model)
        candidate.learn(total_timesteps=timesteps, reset_num_timesteps=False)
        after = self.tracker.analyze_game(
            num_episodes=evaluation_episodes,
            model=candidate,
            persist=True,
        )
        candidate_dir = TRACKING_DIR / "candidates" / self.model_name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate_path = candidate_dir / f"candidate_{stamp}"
        candidate.save(candidate_path)
        after["candidate_path"] = str(candidate_path.with_suffix(".zip"))
        should_promote = "error" in before or after["quality_score"] >= before["quality_score"]
        after["promoted"] = should_promote
        if should_promote:
            best_dir = PROJECT_ROOT / "models" / self.model_name / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)
            candidate.save(best_dir / "best_model")
            self.tracker = ModelErrorTracker(self.model_name)
        return before, after
def run_full_pipeline(timesteps: int = 25_000) -> None:
    print("Chess RL error tracker — mistakes are written to tracking/*.jsonl")
    for name in ("ppo", "a2c", "dqn", "ddqn"):
        updater = SelfUpdatingAgent(name)
        before, after = updater.update_model(timesteps=timesteps)
        print(
            f"{name.upper():4} quality {before.get('quality_score', float('nan')):.3f} "
            f"→ {after.get('quality_score', float('nan')):.3f} "
            f"promoted={after.get('promoted')}"
        )
if __name__ == "__main__":
    run_full_pipeline()
