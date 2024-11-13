import chess
import chess.pgn
import chess.engine
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import openai
import anthropic
from dataclasses import dataclass

@dataclass
class ChessConfig:
    """Configuration for chess games"""
    stockfish_path: str = "/opt/homebrew/bin/stockfish"
    verbose: bool = True
    output_path: str = "game.pgn"

@dataclass
class LMConfig:
    """Configuration for LM-based chess models"""
    model_name: str
    api_key: str
    type: str
    max_retries: int = 3
    is_white: bool = True

class ChessModel(ABC):
    """Abstract base class for chess models"""
    @abstractmethod
    def get_move(self, board: chess.Board) -> chess.Move:
        """Get the next move from the model given a board state"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the model name"""
        pass

class StockfishModel(ChessModel):
    """Example implementation of a chess engine model"""
    def __init__(self, path: str, time_limit: float = 0.01):
        self.path = path
        self.time_limit = time_limit
        self.engine = chess.engine.SimpleEngine.popen_uci(path)

    def get_move(self, board: chess.Board) -> chess.Move:
        result = self.engine.play(board, chess.engine.Limit(time=self.time_limit))
        return result.move

    @property
    def name(self) -> str:
        return "Stockfish"

    def __del__(self):
        if hasattr(self, 'engine'):
            self.engine.quit()

class GPT4ChessModel(ChessModel):
    """
    GPT-4 implementation of a chess model.
    Uses structured prompting to get valid moves.
    """
    def __init__(self, config: LMConfig):
        self.config = config
        openai.api_key = config.api_key
        self._role = "White" if config.is_white else "Black"

    @property
    def name(self) -> str:
        return f"{self.config.model_name}"

    def create_prompt(self, board: chess.Board) -> str:
        """Create a structured prompt for the current board state"""
        legal_moves = [board.san(move) for move in board.legal_moves]

        return f"""As a chess engine playing {self._role}, analyze this position:

Current Board State:
{board}

PGN: {str(board)}

Legal moves: {', '.join(legal_moves)}

Important:
1. Consider piece activity, king safety, and pawn structure
2. Choose the best move from the legal moves list
3. Respond ONLY with the chosen move in standard algebraic notation (e.g., "e4" or "Nf3")
4. No explanations or additional text"""

    def get_move(self, board: chess.Board) -> chess.Move:
        """Get the next move from GPT-4"""
        prompt = self.create_prompt(board)

        for _ in range(self.config.max_retries):
            response = openai.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "You are a chess engine. Respond only with moves in algebraic notation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10
            )

            move_str = response.choices[0].message.content.strip()
            try:
                move = board.parse_san(move_str)
            except Exception as e:
                print(f"Error parsing move: {e}")
                continue

            if move in board.legal_moves:
                return move


class O1ChessModel(ChessModel):
    """
    O1 implementation of a chess model.
    Uses structured prompting to get valid moves.
    """
    def __init__(self, config: LMConfig):
        self.config = config
        openai.api_key = config.api_key
        self._role = "White" if config.is_white else "Black"

    @property
    def name(self) -> str:
        return f"{self.config.model_name}"

    def create_prompt(self, board: chess.Board) -> str:
        """Create a structured prompt for the current board state"""
        legal_moves = [board.san(move) for move in board.legal_moves]

        return f"""As a chess engine playing {self._role}, analyze this position:

Current Board State:
{board}

PGN: {str(board)}

Legal moves: {', '.join(legal_moves)}

Important:
1. Consider piece activity, king safety, and pawn structure
2. Choose the best move from the legal moves list
3. Respond ONLY with the chosen move in standard algebraic notation (e.g., "e4" or "Nf3")
4. No explanations or additional text"""

    def get_move(self, board: chess.Board) -> chess.Move:
        """Get the next move from GPT-4"""
        prompt = self.create_prompt(board)

        for _ in range(self.config.max_retries):
            response = openai.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            move_str = response.choices[0].message.content.strip()
            try:
                move = board.parse_san(move_str)
            except Exception as e:
                print(f"Error parsing move: {e}")
                continue

            if move in board.legal_moves:
                return move


class ClaudeChessModel(ChessModel):
    """
    Claude implementation of a chess model.
    Uses structured prompting to get valid moves.
    """
    def __init__(self, config: LMConfig):
        self.config = config
        self._role = "White" if config.is_white else "Black"
        self.client = anthropic.Anthropic(api_key=config.api_key)

    @property
    def name(self) -> str:
        return f"{self.config.model_name}"

    def create_prompt(self, board: chess.Board) -> str:
        """Create a structured prompt for the current board state"""
        legal_moves = [board.san(move) for move in board.legal_moves]

        return f"""As a chess engine playing {self._role}, analyze this position:

Current Board State:
{board}

PGN: {str(board)}

Legal moves: {', '.join(legal_moves)}

Important:
1. Consider piece activity, king safety, and pawn structure
2. Choose the best move from the legal moves list
3. Respond ONLY with the chosen move in standard algebraic notation (e.g., "e4" or "Nf3")
4. No explanations or additional text"""

    def get_move(self, board: chess.Board) -> chess.Move:
        """Get the next move from Claude"""
        prompt = self.create_prompt(board)

        for _ in range(self.config.max_retries):
            message = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=10,
                system="You are a chess engine. Respond only with moves in algebraic notation.",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
            )

            move_str = message.content[0].text.strip()
            try:
                move = board.parse_san(move_str)
            except Exception as e:
                print(f"Error parsing move: {e}")
                continue

            if move in board.legal_moves:
                return move

class ChessGame:
    """Manages a chess game between two models"""
    def __init__(self, white_player: ChessModel, black_player: ChessModel,
                 evaluator: Optional[ChessModel] = None):
        self.white_player = white_player
        self.black_player = black_player
        self.board = chess.Board()
        self.evaluator = evaluator
        self.evaluations = []

    def play_game(self, verbose: bool = True) -> Tuple[str, Dict[str, Any]]:
        """Play a complete game and return the PGN and performance stats"""
        move_count = 0

        while not self.board.is_game_over():
            is_white = self.board.turn == chess.WHITE
            current_player = self.white_player if is_white else self.black_player

            if verbose:
                print(f"\n{'White' if is_white else 'Black'} ({current_player.name}) to move")
                print(self.board)

            move = current_player.get_move(self.board)
            if not move:
                print(f"Invalid move from {current_player.name}")
                return None, None
            if verbose:
                print(f"Move: {self.board.san(move)}")

            self.board.push(move)
            move_count += 1

        pgn = self.create_pgn()

        stats = None # TODO: calculate average centipawn loss

        return pgn, stats


    def create_pgn(self) -> str:
        """Create a PGN string from the game"""
        game = chess.pgn.Game()
        game.headers["Event"] = f"{self.white_player.name} vs {self.black_player.name}"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["White"] = self.white_player.name
        game.headers["Black"] = self.black_player.name
        game.headers["Result"] = self.board.result()

        node = game
        for move in self.board.move_stack:
            node = node.add_variation(move)

        return str(game)

def play_chess_game(config: ChessConfig, lm_config: LMConfig) -> None:
    """Utility function to quickly set up and play a game"""
    # Create models
    lm_model = None
    if "openai" in lm_config.type:
        if "o1" in lm_config.model_name:
            lm_model = O1ChessModel(lm_config)
        elif "gpt" in lm_config.model_name:
            lm_model = GPT4ChessModel(lm_config)
    elif "anthropic" in lm_config.type:
        lm_model = ClaudeChessModel(lm_config)
    assert lm_model
    stockfish = StockfishModel(config.stockfish_path)

    # Set up players based on color
    if lm_config.is_white:
        white_player = lm_model
        black_player = stockfish
    else:
        white_player = stockfish
        black_player = lm_model

    # Create evaluator for analysis
    evaluator = StockfishModel(config.stockfish_path)

    # Create and play game
    game = ChessGame(
        white_player=white_player,
        black_player=black_player,
        evaluator=evaluator
    )

    pgn, stats = game.play_game(verbose=config.verbose)
    if not pgn or not stats:
        return None

    # Save game
    with open(config.output_path, "x") as f:
        f.write(pgn)
    return stats
