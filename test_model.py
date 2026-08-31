
from __future__ import annotations
import chess
from chess_env import ChessEnv
from model_support import load_model, normalise_model_name, predict_legal_move
def _move_from_text(board: chess.Board, text: str) -> chess.Move | None:
    try:
        move = chess.Move.from_uci(text.strip().lower())
    except ValueError:
        return None
    return move if move in board.legal_moves else None
def play_human_vs_ai(model_type: str = "ppo", player_color: str = "white") -> None:
    name = normalise_model_name(model_type)
    human = chess.WHITE if player_color.lower() == "white" else chess.BLACK
    model = load_model(name)
    env = ChessEnv(mode="manual", render_mode="human", max_steps=300)
    observation, _ = env.reset()
    print(f"Human: {'White' if human else 'Black'} | AI: {name.upper()}")
    while not env.board.is_game_over(claim_draw=True) and env.current_step < env.max_steps:
        env.render()
        if env.board.turn == human:
            move = _move_from_text(env.board, input("Your move (UCI, e.g. e2e4): "))
            if move is None:
                print("That is not a legal move.")
                continue
        else:
            decision = predict_legal_move(model, name, observation, env)
            move = decision.move
            if move is None:
                break
            print(f"AI: {move.uci()} [{decision.action_encoding}]")
        observation, _, terminated, truncated, _ = env.step(move)
        if terminated or truncated:
            break
    env.render()
    print("Result:", env.board.outcome(claim_draw=True) or "move limit reached")
def play_ai_vs_ai(white_model: str = "ppo", black_model: str = "a2c", max_steps: int = 100) -> None:
    white_name, black_name = normalise_model_name(white_model), normalise_model_name(black_model)
    models = {chess.WHITE: load_model(white_name), chess.BLACK: load_model(black_name)}
    names = {chess.WHITE: white_name, chess.BLACK: black_name}
    env = ChessEnv(mode="manual", render_mode="human", max_steps=max_steps)
    observation, _ = env.reset()
    while not env.board.is_game_over(claim_draw=True) and env.current_step < max_steps:
        colour = env.board.turn
        decision = predict_legal_move(models[colour], names[colour], observation, env)
        if decision.move is None:
            break
        observation, reward, terminated, truncated, _ = env.step(decision.move)
        side = "White" if colour else "Black"
        print(f"{side} {names[colour].upper()}: {decision.move.uci()}  reward={reward:.2f}")
        if terminated or truncated:
            break
    env.render()
    print("Result:", env.board.outcome(claim_draw=True) or "move limit reached")
if __name__ == "__main__":
    play_ai_vs_ai()
