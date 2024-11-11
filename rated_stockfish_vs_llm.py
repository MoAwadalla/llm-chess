import chess
import chess.pgn
import chess.engine
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
import math
import openai
from dataclasses import dataclass

@dataclass
class ChessConfig:
    """Configuration for chess games"""
    stockfish_path: str = "/opt/homebrew/bin/stockfish"
    verbose: bool = True
    output_path: str = "game.pgn"

@dataclass
class LLMConfig:
    """Configuration for LLM-based chess models"""
    model_name: str
    api_key: str
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
    def __init__(self, config: LLMConfig):
        self.config = config
        openai.api_key = config.api_key
        self._role = "White" if config.is_white else "Black"

    @property
    def name(self) -> str:
        return f"GPT4-{self.config.model_name}"

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

        for attempt in range(self.config.max_retries):
            try:
                response = openai.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "system", "content": "You are a chess engine. Respond only with moves in algebraic notation."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=10
                )

                move_str = response.choices[0].message.content.strip()
                move = board.parse_san(move_str)

                if move in board.legal_moves:
                    return move

            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    # If all retries failed, make a fallback move
                    return self._get_fallback_move(board)
                continue

        return self._get_fallback_move(board)

    def _get_fallback_move(self, board: chess.Board) -> chess.Move:
        """Make a simple fallback move if GPT-4 fails"""
        # Prioritize captures, then checks, then any legal move
        legal_moves = list(board.legal_moves)

        # Try captures first
        captures = [move for move in legal_moves if board.is_capture(move)]
        if captures:
            return captures[0]

        # Then checks
        checks = [move for move in legal_moves if board.gives_check(move)]
        if checks:
            return checks[0]

        # Finally, any legal move
        return legal_moves[0]

    def _get_material_count(self, board: chess.Board) -> Dict[str, int]:
        """Calculate material count for both sides"""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9
        }

        white_material = sum(
            len(board.pieces(piece_type, chess.WHITE)) * value
            for piece_type, value in piece_values.items()
        )

        black_material = sum(
            len(board.pieces(piece_type, chess.BLACK)) * value
            for piece_type, value in piece_values.items()
        )

        return {"White": white_material, "Black": black_material}

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
            if verbose:
                print(f"Move: {self.board.san(move)}")

            self.board.push(move)
            move_count += 1

            # Get evaluation if available
            if self.evaluator and is_white:
                eval_score = self.get_position_evaluation()
                if eval_score is not None:
                    self.evaluations.append(eval_score)
                    if verbose:
                        print(f"Position evaluation: {eval_score} centipawns")

        stats = self.calculate_performance_stats()
        pgn = self.create_pgn()

        if verbose:
            self.print_game_result(stats)

        return pgn, stats

    def get_position_evaluation(self) -> Optional[int]:
        """Get position evaluation if an evaluator is available"""
        if isinstance(self.evaluator, StockfishModel):
            info = self.evaluator.engine.analyse(self.board, chess.engine.Limit(depth=20))
            if "score" in info:
                score = info["score"].white()
                if score.is_mate():
                    mate_in = score.mate()
                    return 10000 - (mate_in * 100) if mate_in > 0 else -10000 - (mate_in * 100)
                return score.score()
        return None

    def calculate_performance_stats(self) -> Dict[str, Any]:
        """Calculate performance statistics"""
        if not self.evaluations:
            return {}

        valid_evals = [e for e in self.evaluations if e is not None]
        if not valid_evals:
            return {}

        avg_loss = abs(sum(valid_evals) / len(valid_evals))
        return {
            'avg_centipawn_loss': avg_loss,
            'max_loss': max(abs(e) for e in valid_evals),
            'min_loss': min(abs(e) for e in valid_evals),
            'good_moves': len([e for e in valid_evals if abs(e) < 50]),
            'blunders': len([e for e in valid_evals if abs(e) > 200]),
            'total_moves': len(valid_evals),
            'estimated_rating': self._estimate_rating(avg_loss)
        }

    @staticmethod
    def _estimate_rating(avg_centipawn_loss: float, baseline: int = 3000) -> float:
        """Estimate rating based on average centipawn loss"""
        return baseline - (avg_centipawn_loss * 2)

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

    def print_game_result(self, stats: Dict[str, Any]):
        """Print game results and statistics"""
        print("\n=== Game Complete ===")
        print(f"Result: {self.board.result()}")
        if stats:
            print(f"\nStatistics:")
            print(f"Average centipawn loss: {stats['avg_centipawn_loss']:.2f}")
            # print(f"Estimated rating: {stats['estimated_rating']:.0f}")
            print(f"Good moves: {stats['good_moves']}")
            print(f"Blunders: {stats['blunders']}")

def play_chess_game(config: ChessConfig, llm_config: LLMConfig) -> None:
    """Utility function to quickly set up and play a game"""
    # Create models
    gpt_model = GPT4ChessModel(llm_config)
    stockfish = StockfishModel(config.stockfish_path)

    # Set up players based on color
    if llm_config.is_white:
        white_player = gpt_model
        black_player = stockfish
    else:
        white_player = stockfish
        black_player = gpt_model

    # Create evaluator for analysis
    evaluator = StockfishModel(config.stockfish_path)

    # Create and play game
    game = ChessGame(
        white_player=white_player,
        black_player=black_player,
        evaluator=evaluator
    )

    pgn, stats = game.play_game(verbose=config.verbose)

    # Save game
    with open(config.output_path, "w") as f:
        f.write(pgn)

if __name__ == "__main__":
    import os

    chess_config = ChessConfig(
        stockfish_path="/opt/homebrew/bin/stockfish",  # Adjust path as needed
        verbose=True,
        output_path="game.pgn"
    )

    llm_config = LLMConfig(
        model_name="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        is_white=True  # GPT-4 plays as White
    )

    play_chess_game(chess_config, llm_config)