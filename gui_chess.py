
from __future__ import annotations
from pathlib import Path
import time
from typing import Optional
import chess
import pygame
from chess_env import ChessEnv
from model_support import ModelDecision, load_model, predict_legal_move
MODEL_NAMES = ("ppo", "a2c", "dqn", "ddqn", "ddpg", "sac")
PROMOTIONS = {
    pygame.K_q: chess.QUEEN,
    pygame.K_r: chess.ROOK,
    pygame.K_b: chess.BISHOP,
    pygame.K_n: chess.KNIGHT,
}
class ChessGUIApp:
    def __init__(self) -> None:
        pygame.init()
        pygame.font.init()
        self.board_size = 512
        self.square_size = 64
        self.width, self.height = 860, 576
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Chess RL — playable model arena")
        self.clock = pygame.time.Clock()
        apple_symbols = Path("/System/Library/Fonts/Apple Symbols.ttf")
        self.piece_font_has_glyphs = apple_symbols.exists()
        self.font_piece = (
            pygame.font.Font(str(apple_symbols), 46)
            if self.piece_font_has_glyphs
            else pygame.font.SysFont("arial", 28, bold=True)
        )
        self.font_ui = pygame.font.SysFont("arial", 14, bold=True)
        self.font_small = pygame.font.SysFont("arial", 12)
        self.font_title = pygame.font.SysFont("arial", 18, bold=True)
        self.LIGHT = (240, 217, 181)
        self.DARK = (181, 136, 99)
        self.SELECT = (130, 180, 105)
        self.LEGAL = (170, 210, 140)
        self.LAST = (205, 210, 106)
        self.BG = (30, 33, 36)
        self.PANEL = (45, 49, 54)
        self.BUTTON = (65, 72, 82)
        self.ACTIVE = (74, 132, 85)
        self.ACCENT = (77, 141, 204)
        self.TEXT = (240, 240, 240)
        self.MUTED = (180, 185, 190)
        self.WARNING = (242, 181, 63)
        self.unicode_pieces = {
            "P": "♙", "R": "♖", "N": "♘", "B": "♗", "Q": "♕", "K": "♔",
            "p": "♟", "r": "♜", "n": "♞", "b": "♝", "q": "♛", "k": "♚",
        }
        best_model_name = "ppo"
        if Path("./models/best_model_info.txt").exists():
            best_model_name = Path("./models/best_model_info.txt").read_text().strip()
        self.mode = "human_vs_ai"
        self.player_color = chess.WHITE
        self.model_types = {chess.WHITE: best_model_name, chess.BLACK: best_model_name}
        self.models: dict[chess.Color, object | None] = {chess.WHITE: None, chess.BLACK: None}
        self.active_slot = chess.BLACK
        self.promotion_piece = chess.QUEEN
        self.selected_square: Optional[chess.Square] = None
        self.legal_destinations: list[chess.Square] = []
        self.last_decision: Optional[ModelDecision] = None
        self.status_message = "Select a piece to move. Promotion defaults to Queen."
        self.last_ai_time = 0.0
        self.done = False
        self.env: ChessEnv
        self.observation = None
        self.reset_game()
        self.load_models()
    @property
    def board(self) -> chess.Board:
        return self.env.board
    def load_models(self) -> None:
        for colour, model_type in self.model_types.items():
            try:
                self.models[colour] = load_model(model_type)
            except Exception as error:
                self.models[colour] = None
                self.status_message = f"Could not load {model_type.upper()}: {error}"
    def reset_game(self) -> None:
        self.env = ChessEnv(mode="manual", render_mode="gui", max_steps=300)
        self.observation, _ = self.env.reset()
        self.selected_square = None
        self.legal_destinations = []
        self.last_decision = None
        self.done = False
        self.last_ai_time = time.monotonic()
    def set_model(self, model_type: str) -> None:
        self.model_types[self.active_slot] = model_type
        try:
            self.models[self.active_slot] = load_model(model_type)
            if self.models[self.active_slot] is None:
                self.status_message = f"{model_type.upper()} checkpoint missing — heuristic fallback is active."
            else:
                self.status_message = f"Loaded {model_type.upper()} for {'White' if self.active_slot else 'Black'}."
        except Exception as error:
            self.models[self.active_slot] = None
            self.status_message = f"Could not load {model_type.upper()}: {error}"
    def display_is_flipped(self) -> bool:
        return self.mode == "human_vs_ai" and self.player_color == chess.BLACK
    def square_from_screen(self, position: tuple[int, int]) -> chess.Square:
        file_index = position[0] // self.square_size
        rank_from_top = position[1] // self.square_size
        if self.display_is_flipped():
            return chess.square(7 - file_index, rank_from_top)
        return chess.square(file_index, 7 - rank_from_top)
    def screen_rect_for_square(self, square: chess.Square) -> pygame.Rect:
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        if self.display_is_flipped():
            x = (7 - file_index) * self.square_size
            y = rank_index * self.square_size
        else:
            x = file_index * self.square_size
            y = (7 - rank_index) * self.square_size
        return pygame.Rect(x, y, self.square_size, self.square_size)
    def ai_colour_to_move(self) -> Optional[chess.Color]:
        if self.mode == "ai_vs_ai":
            return self.board.turn
        if self.mode == "human_vs_ai" and self.board.turn != self.player_color:
            return self.board.turn
        return None
    def step_ai(self) -> None:
        if self.done:
            return
        colour = self.ai_colour_to_move()
        if colour is None:
            return
        model_type = self.model_types[colour]
        decision = predict_legal_move(self.models[colour], model_type, self.observation, self.env)
        if decision.move is None:
            self.done = True
            self.status_message = "No legal move remains."
            return
        self.last_decision = decision
        self.observation, _, terminated, truncated, _ = self.env.step(decision.move)
        self.done = terminated or truncated
        side = "White" if colour == chess.WHITE else "Black"
        recovered = " — recovered illegal action" if decision.was_illegal else ""
        aliased = " — compact action aliased" if decision.was_aliased else ""
        self.status_message = f"{side} {model_type.upper()}: {decision.move.uci()}{recovered}{aliased}"
    def play_human_move(self, move: chess.Move) -> None:
        self.observation, _, terminated, truncated, _ = self.env.step(move)
        self.done = terminated or truncated
        self.selected_square = None
        self.legal_destinations = []
        self.last_ai_time = time.monotonic()
        self.status_message = f"You played {move.uci()}."
    def handle_board_click(self, position: tuple[int, int]) -> None:
        if self.done or self.mode == "ai_vs_ai":
            return
        if self.mode == "human_vs_ai" and self.board.turn != self.player_color:
            return
        clicked = self.square_from_screen(position)
        piece = self.board.piece_at(clicked)
        if self.selected_square is None:
            if piece and piece.color == self.board.turn:
                self.selected_square = clicked
                self.legal_destinations = [move.to_square for move in self.board.legal_moves if move.from_square == clicked]
            return
        candidates = [
            move for move in self.board.legal_moves
            if move.from_square == self.selected_square and move.to_square == clicked
        ]
        chosen = next((move for move in candidates if move.promotion == self.promotion_piece), None)
        if chosen is None:
            chosen = next((move for move in candidates if move.promotion is None), None)
        if chosen is not None:
            self.play_human_move(chosen)
            return
        if piece and piece.color == self.board.turn:
            self.selected_square = clicked
            self.legal_destinations = [move.to_square for move in self.board.legal_moves if move.from_square == clicked]
        else:
            self.selected_square = None
            self.legal_destinations = []
    def handle_panel_click(self, position: tuple[int, int]) -> None:
        x, y = position
        if not 530 <= x <= 840:
            return
        if 56 <= y < 86:
            if x < 685:
                self.mode = "human_vs_ai"
                self.status_message = "Human vs AI. Choose your colour below."
            else:
                self.mode = "ai_vs_ai"
                self.status_message = "AI vs AI uses the white and black models shown below."
            self.reset_game()
        elif 94 <= y < 124:
            if x < 685:
                self.mode = "human_vs_human"
                self.status_message = "Human vs Human."
                self.reset_game()
            else:
                self.reset_game()
                self.status_message = "Game reset."
        elif 180 <= y < 208:
            if x < 685:
                self.player_color = chess.WHITE
                self.active_slot = chess.BLACK
            else:
                self.player_color = chess.BLACK
                self.active_slot = chess.WHITE
            self.reset_game()
        elif 264 <= y < 294:
            self.active_slot = chess.WHITE if x < 685 else chess.BLACK
        elif 344 <= y < 428:
            column = 0 if x < 685 else 1
            row = (y - 344) // 30
            index = int(row * 2 + column)
            if 0 <= index < len(MODEL_NAMES):
                self.set_model(MODEL_NAMES[index])
        elif 486 <= y < 526:
            self.reset_game()
            self.status_message = "Game reset."
    def handle_click(self, position: tuple[int, int]) -> None:
        if position[0] < self.board_size and position[1] < self.board_size:
            self.handle_board_click(position)
        else:
            self.handle_panel_click(position)
    def draw_button(self, label: str, rect: pygame.Rect, *, active: bool = False, accent: bool = False) -> None:
        colour = self.ACCENT if accent else (self.ACTIVE if active else self.BUTTON)
        pygame.draw.rect(self.screen, colour, rect, border_radius=4)
        text = self.font_ui.render(label, True, self.TEXT)
        self.screen.blit(text, text.get_rect(center=rect.center))
    def draw_board(self) -> None:
        last_move = self.board.peek() if self.board.move_stack else None
        for square in chess.SQUARES:
            rect = self.screen_rect_for_square(square)
            file_index, rank_index = chess.square_file(square), chess.square_rank(square)
            colour = self.LIGHT if (file_index + rank_index) % 2 else self.DARK
            if last_move and square in {last_move.from_square, last_move.to_square}:
                colour = self.LAST
            if square == self.selected_square:
                colour = self.SELECT
            elif square in self.legal_destinations:
                colour = self.LEGAL
            pygame.draw.rect(self.screen, colour, rect)
            piece = self.board.piece_at(square)
            if piece:
                text_colour = (250, 250, 250) if piece.color == chess.WHITE else (18, 18, 18)
                symbol = self.unicode_pieces[piece.symbol()] if self.piece_font_has_glyphs else piece.symbol().upper()
                glyph = self.font_piece.render(symbol, True, text_colour)
                self.screen.blit(glyph, glyph.get_rect(center=rect.center))
        files = "hgfedcba" if self.display_is_flipped() else "abcdefgh"
        ranks = "12345678" if self.display_is_flipped() else "87654321"
        for index, label in enumerate(files):
            text = self.font_small.render(label, True, (75, 75, 75))
            self.screen.blit(text, (index * 64 + 3, 496))
        for index, label in enumerate(ranks):
            text = self.font_small.render(label, True, (75, 75, 75))
            self.screen.blit(text, (2, index * 64 + 2))
    def draw_panel(self) -> None:
        pygame.draw.rect(self.screen, self.PANEL, (512, 0, self.width - 512, self.height))
        self.screen.blit(self.font_title.render("CHESS RL ARENA", True, self.TEXT), (530, 18))
        self.draw_button("Human vs AI", pygame.Rect(530, 56, 145, 30), active=self.mode == "human_vs_ai")
        self.draw_button("AI vs AI", pygame.Rect(685, 56, 155, 30), active=self.mode == "ai_vs_ai")
        self.draw_button("Human vs Human", pygame.Rect(530, 94, 145, 30), active=self.mode == "human_vs_human")
        self.draw_button("Reset game", pygame.Rect(685, 94, 155, 30), accent=True)
        self.screen.blit(self.font_ui.render("PLAYER VIEW", True, self.MUTED), (530, 144))
        self.draw_button("Play White", pygame.Rect(530, 180, 145, 28), active=self.player_color == chess.WHITE)
        self.draw_button("Play Black", pygame.Rect(685, 180, 155, 28), active=self.player_color == chess.BLACK)
        self.screen.blit(self.font_ui.render("MODEL SLOT", True, self.MUTED), (530, 240))
        self.draw_button("White model", pygame.Rect(530, 264, 145, 30), active=self.active_slot == chess.WHITE)
        self.draw_button("Black model", pygame.Rect(685, 264, 155, 30), active=self.active_slot == chess.BLACK)
        slot = "White" if self.active_slot == chess.WHITE else "Black"
        self.screen.blit(self.font_ui.render(f"Select {slot} model", True, self.MUTED), (530, 322))
        for index, name in enumerate(MODEL_NAMES):
            col, row = index % 2, index // 2
            rect = pygame.Rect(530 + col * 155, 344 + row * 30, 145 if col == 0 else 155, 25)
            active = self.model_types[self.active_slot] == name
            loaded = self.models[self.active_slot] is not None and active
            self.draw_button(name.upper(), rect, active=active, accent=loaded)
        turn = "White" if self.board.turn == chess.WHITE else "Black"
        game_status = "GAME OVER" if self.done else f"Turn: {turn}"
        if self.board.is_check() and not self.done:
            game_status += " — CHECK"
        self.screen.blit(self.font_ui.render(game_status, True, self.WARNING), (530, 444))
        promoted_name = chess.piece_name(self.promotion_piece).title()
        self.screen.blit(self.font_small.render(f"Promotion: {promoted_name}  (Q/R/B/N)", True, self.MUTED), (530, 465))
        self.draw_button("RESET GAME", pygame.Rect(530, 486, 310, 40), accent=True)
        status = self.status_message[:52]
        self.screen.blit(self.font_small.render(status, True, self.TEXT), (530, 536))
        if self.last_decision and self.last_decision.was_illegal:
            self.screen.blit(self.font_small.render("Illegal model output recovered safely.", True, self.WARNING), (530, 554))
    def draw(self) -> None:
        self.screen.fill(self.BG)
        self.draw_board()
        self.draw_panel()
        pygame.display.flip()
    def run(self) -> None:
        running = True
        while running:
            self.clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in PROMOTIONS:
                    self.promotion_piece = PROMOTIONS[event.key]
                    self.status_message = f"Promotion piece set to {chess.piece_name(self.promotion_piece)}."
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)
            if self.ai_colour_to_move() is not None and not self.done:
                if time.monotonic() - self.last_ai_time >= 0.55:
                    self.step_ai()
                    self.last_ai_time = time.monotonic()
            self.draw()
        pygame.quit()
if __name__ == "__main__":
    ChessGUIApp().run()
