
from __future__ import annotations
import random
from typing import Any, Optional
import chess
import gymnasium as gym
import numpy as np
from gymnasium import spaces
PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}
MAX_LEGAL_MOVES = 218
AUTO_OPPONENT_MODES = {"ai_vs_ai", "agent_vs_opponent"}
def colour_from_name(value: str) -> chess.Color:
    value = value.lower().strip()
    if value == "white":
        return chess.WHITE
    if value == "black":
        return chess.BLACK
    raise ValueError("player_color must be 'white', 'black', or 'random'")
class BuiltInOpponent:
    def __init__(self, mode: str = "heuristic", rng: Optional[random.Random] = None):
        if mode not in {"heuristic", "random"}:
            raise ValueError("opponent_type must be 'heuristic' or 'random'")
        self.mode = mode
        self.rng = rng or random.Random()
    def choose_move(self, board: chess.Board) -> Optional[chess.Move]:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        if self.mode == "random":
            return self.rng.choice(legal_moves)
        perspective = board.turn
        scores = [(self.score_move(board, move, perspective), move) for move in legal_moves]
        best_score = max(score for score, _ in scores)
        return self.rng.choice([move for score, move in scores if score == best_score])
    @classmethod
    def score_move(cls, board: chess.Board, move: chess.Move, perspective: chess.Color) -> float:
        is_capture = board.is_capture(move)
        captured_piece = board.piece_at(move.to_square)
        capture_bonus = (
            PIECE_VALUES.get(captured_piece.piece_type, 0.0) if captured_piece else 1.0
        ) if is_capture else 0.0
        promotion_bonus = (
            PIECE_VALUES.get(move.promotion, PIECE_VALUES[chess.PAWN])
            - PIECE_VALUES[chess.PAWN]
            if move.promotion
            else 0.0
        )
        board.push(move)
        score = cls.evaluate(board, perspective) + (0.25 if board.is_check() else 0.0)
        board.pop()
        return score + 0.15 * capture_bonus + 0.05 * promotion_bonus
    @staticmethod
    def evaluate(board: chess.Board, perspective: chess.Color) -> float:
        if board.is_checkmate():
            return 1_000.0 if board.turn != perspective else -1_000.0
        if board.is_game_over(claim_draw=True):
            return 0.0
        score = 0.0
        for square, piece in board.piece_map().items():
            value = PIECE_VALUES[piece.piece_type]
            file_distance = abs(chess.square_file(square) - 3.5)
            rank_distance = abs(chess.square_rank(square) - 3.5)
            centrality = max(0.0, 3.5 - file_distance - rank_distance) * 0.04
            score += value + centrality if piece.color == perspective else -(value + centrality)
        return score
class ChessEnv(gym.Env):
    metadata = {"render_modes": ["human", "unicode", "ascii"], "render_fps": 4}
    def __init__(
        self,
        mode: str = "agent_vs_opponent",
        player_color: str = "white",
        opponent_type: str = "heuristic",
        reward_config: Optional[dict[str, float]] = None,
        render_mode: Optional[str] = None,
        max_steps: int = 200,
        invalid_move_ends_episode: bool = False,
        replay_fens: Optional[list[str]] = None,
        replay_probability: float = 0.0,
    ):
        super().__init__()
        self.mode = mode.lower().strip()
        if self.mode == "human_vs_ai":
            self.mode = "manual"
        if self.mode not in AUTO_OPPONENT_MODES | {"manual", "human_vs_human"}:
            raise ValueError("Unsupported mode. Use agent_vs_opponent, ai_vs_ai, manual, or human_vs_human.")
        self.requested_player_color = player_color.lower().strip()
        if self.requested_player_color not in {"white", "black", "random"}:
            raise ValueError("player_color must be 'white', 'black', or 'random'")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not 0.0 <= replay_probability <= 1.0:
            raise ValueError("replay_probability must be between 0 and 1")
        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.invalid_move_ends_episode = bool(invalid_move_ends_episode)
        self.replay_fens = list(replay_fens or [])
        self.replay_probability = float(replay_probability)
        self.action_space = spaces.Discrete(4096)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(14, 8, 8), dtype=np.float32)
        self.rewards = {
            "win": 10.0,
            "loss": -10.0,
            "draw": 0.0,
            "illegal_move": -1.0,
            "step_penalty": -0.01,
            "check_bonus": 0.2,
            "center_control_bonus": 0.05,
            "material_weight": 1.0,
        }
        if reward_config:
            self.rewards.update(reward_config)
        self.board = chess.Board()
        self.current_step = 0
        self.agent_color = chess.WHITE
        self._rng = random.Random()
        self.opponent = BuiltInOpponent(opponent_type, self._rng)
        self.last_action: Optional[int] = None
        self.last_move: Optional[chess.Move] = None
        self.last_opponent_move: Optional[chess.Move] = None
        self.invalid_action_count = 0
        self.was_reset_from_replay = False
    @property
    def auto_opponent(self) -> bool:
        return self.mode in AUTO_OPPONENT_MODES
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)
        self.board.reset()
        self.current_step = 0
        self.invalid_action_count = 0
        self.last_action = self.last_move = self.last_opponent_move = None
        self.was_reset_from_replay = False
        self.agent_color = (
            self._rng.choice([chess.WHITE, chess.BLACK])
            if self.requested_player_color == "random"
            else colour_from_name(self.requested_player_color)
        )
        requested_fen = (options or {}).get("fen")
        replay_fen = requested_fen
        if replay_fen is None and self.replay_fens and self._rng.random() < self.replay_probability:
            replay_fen = self._rng.choice(self.replay_fens)
        if replay_fen is not None:
            try:
                self.board.set_fen(replay_fen)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid replay FEN: {replay_fen!r}") from error
            if self.board.is_game_over(claim_draw=True):
                raise ValueError("Replay FEN must have at least one legal continuation.")
            self.agent_color = self.board.turn
            self.was_reset_from_replay = True
        elif self.auto_opponent and self.agent_color == chess.BLACK:
            self.last_opponent_move = self.opponent.choose_move(self.board)
            if self.last_opponent_move:
                self.board.push(self.last_opponent_move)
        return self._get_observation(), self._get_info()
    def step(self, action: Any):
        if self.board.is_game_over(claim_draw=True):
            return self._get_observation(), 0.0, True, False, self._get_info()
        if self.auto_opponent and self.board.turn != self.agent_color:
            info = self._get_info()
            info["wrong_turn"] = True
            return self._get_observation(), float(self.rewards["illegal_move"]), True, False, info
        move = self.parse_action(action)
        if move is None:
            self.current_step += 1
            self.invalid_action_count += 1
            truncated = self.current_step >= self.max_steps
            info = self._get_info()
            info.update(
                {
                    "illegal_move": True,
                    "attempted_action": self._serialise_action(action),
                    "invalid_action_count": self.invalid_action_count,
                }
            )
            return (
                self._get_observation(),
                float(self.rewards["illegal_move"]),
                self.invalid_move_ends_episode,
                truncated and not self.invalid_move_ends_episode,
                info,
            )
        self.current_step += 1
        self.last_action = self.move_to_action(move)
        self.last_move, self.last_opponent_move = move, None
        material_before = self._calculate_material_balance(self.board, self.agent_color)
        self.board.push(move)
        material_after_move = self._calculate_material_balance(self.board, self.agent_color)
        material_delta = material_after_move - material_before
        terminated = self.board.is_game_over(claim_draw=True)
        truncated = self.current_step >= self.max_steps
        reward = self._compute_reward(move, material_delta, terminated)
        if self.auto_opponent and not (terminated or truncated):
            self.last_opponent_move = self.opponent.choose_move(self.board)
            if self.last_opponent_move is not None:
                self.board.push(self.last_opponent_move)
            material_after_reply = self._calculate_material_balance(self.board, self.agent_color)
            reward += (material_after_reply - material_after_move) * self.rewards["material_weight"]
            terminated = self.board.is_game_over(claim_draw=True)
            if terminated:
                reward += self._terminal_reward()
        return self._get_observation(), float(reward), terminated, truncated, self._get_info()
    def parse_action(self, action: Any) -> Optional[chess.Move]:
        if isinstance(action, chess.Move):
            return action if action in self.board.legal_moves else None
        if isinstance(action, str):
            try:
                move = chess.Move.from_uci(action.strip().lower())
            except ValueError:
                return None
            return move if move in self.board.legal_moves else None
        if isinstance(action, (int, np.integer, np.ndarray)):
            try:
                value = int(action.item()) if hasattr(action, "item") else int(action)
            except (TypeError, ValueError):
                return None
            if not 0 <= value < self.action_space.n:
                return None
            from_square, to_square = divmod(value, 64)
            candidates = [
                move for move in self.board.legal_moves
                if move.from_square == from_square and move.to_square == to_square
            ]
            if not candidates:
                return None
            return next((move for move in candidates if move.promotion == chess.QUEEN), candidates[0])
        return None
    @staticmethod
    def move_to_action(move: chess.Move) -> int:
        return move.from_square * 64 + move.to_square
    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
        for move in self.board.legal_moves:
            mask[self.move_to_action(move)] = True
        return mask
    def legal_moves_sorted(self) -> list[chess.Move]:
        return sorted(self.board.legal_moves, key=lambda move: move.uci())
    def _compute_reward(self, move: chess.Move, material_delta: float, terminated: bool) -> float:
        reward = self.rewards["step_penalty"] + material_delta * self.rewards["material_weight"]
        if move.to_square in {chess.D4, chess.E4, chess.D5, chess.E5}:
            reward += self.rewards["center_control_bonus"]
        if self.board.is_check():
            reward += self.rewards["check_bonus"]
        if terminated:
            reward += self._terminal_reward()
        return float(reward)
    def _terminal_reward(self) -> float:
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return float(self.rewards["draw"])
        return float(self.rewards["win"] if outcome.winner == self.agent_color else self.rewards["loss"])
    @staticmethod
    def _serialise_action(action: Any) -> Any:
        if isinstance(action, np.ndarray):
            return action.tolist()
        if isinstance(action, np.generic):
            return action.item()
        return str(action) if isinstance(action, chess.Move) else action
    @staticmethod
    def _calculate_material_balance(board: chess.Board, perspective: chess.Color) -> float:
        balance = 0.0
        for piece in board.piece_map().values():
            if piece.piece_type != chess.KING:
                value = PIECE_VALUES[piece.piece_type]
                balance += value if piece.color == perspective else -value
        return balance
    def _get_observation(self) -> np.ndarray:
        observation = np.zeros((14, 8, 8), dtype=np.float32)
        piece_to_plane = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
        }
        for square, piece in self.board.piece_map().items():
            rank, file = divmod(square, 8)
            plane = piece_to_plane[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
            observation[plane, rank, file] = 1.0
        if self.board.turn == chess.WHITE:
            observation[12, :, :] = 1.0
        if self.board.has_queenside_castling_rights(chess.WHITE):
            observation[13, 0, 0] = 1.0
        if self.board.has_kingside_castling_rights(chess.WHITE):
            observation[13, 0, 7] = 1.0
        if self.board.has_queenside_castling_rights(chess.BLACK):
            observation[13, 7, 0] = 1.0
        if self.board.has_kingside_castling_rights(chess.BLACK):
            observation[13, 7, 7] = 1.0
        return observation
    def _get_info(self) -> dict[str, Any]:
        return {
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "agent_color": "white" if self.agent_color == chess.WHITE else "black",
            "legal_moves": [move.uci() for move in self.board.legal_moves],
            "legal_actions_mask": self.get_action_mask(),
            "is_check": self.board.is_check(),
            "is_game_over": self.board.is_game_over(claim_draw=True),
            "last_move": self.last_move.uci() if self.last_move else None,
            "last_opponent_move": self.last_opponent_move.uci() if self.last_opponent_move else None,
            "move_count": self.current_step,
            "replay_position": self.was_reset_from_replay,
        }
    def render(self):
        lines = ["", "  a b c d e f g h", " +-----------------+"]
        for index, line in enumerate(str(self.board).splitlines()):
            lines.append(f"{8 - index}| {line} |{8 - index}")
        lines.extend(
            [
                " +-----------------+",
                "  a b c d e f g h",
                f"Turn: {'White' if self.board.turn else 'Black'} | FEN: {self.board.fen()}",
            ]
        )
        text = "\n".join(lines)
        if self.render_mode in {"unicode", "ascii"}:
            return text
        print(text)
        return None
class LegalMoveIndexWrapper(gym.Wrapper):
    def __init__(self, env: ChessEnv):
        super().__init__(env)
        self.action_space = spaces.Discrete(MAX_LEGAL_MOVES)
    def step(self, action: Any):
        try:
            index = int(np.asarray(action).item())
        except (TypeError, ValueError):
            index = 0
        legal_moves = self.unwrapped.legal_moves_sorted()
        if not legal_moves:
            return self.env.step(0)
        was_aliased = not 0 <= index < len(legal_moves)
        move = legal_moves[index % len(legal_moves)]
        observation, reward, terminated, truncated, info = self.env.step(move)
        info.update({"legal_move_index": index, "action_was_aliased": was_aliased, "selected_move": move.uci()})
        return observation, reward, terminated, truncated, info
    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[: len(self.unwrapped.legal_moves_sorted())] = True
        return mask
class ContinuousLegalMoveWrapper(gym.Wrapper):
    def __init__(self, env: ChessEnv):
        super().__init__(env)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    def step(self, action: Any):
        try:
            value = float(np.asarray(action).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            value = 0.0
        legal_moves = self.unwrapped.legal_moves_sorted()
        if not legal_moves:
            return self.env.step(0)
        clipped = float(np.clip(value, -1.0, 1.0))
        index = min(int((clipped + 1.0) * 0.5 * len(legal_moves)), len(legal_moves) - 1)
        move = legal_moves[index]
        observation, reward, terminated, truncated, info = self.env.step(move)
        info.update({"continuous_choice": clipped, "selected_move": move.uci()})
        return observation, reward, terminated, truncated, info
if __name__ == "__main__":
    env = ChessEnv(mode="agent_vs_opponent", player_color="white")
    observation, _ = env.reset(seed=7)
    for _ in range(5):
        actions = np.flatnonzero(env.get_action_mask())
        observation, reward, terminated, truncated, info = env.step(int(actions[0]))
        print(info["last_move"], info["last_opponent_move"], f"reward={reward:.2f}")
        if terminated or truncated:
            break
