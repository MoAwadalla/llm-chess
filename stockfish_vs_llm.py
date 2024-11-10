import chess
import chess.pgn
import openai
import time
from datetime import datetime

def create_board_state_prompt(board, is_white):
    """Create a concise but effective prompt for GPT about the current board state"""
    color = "White" if is_white else "Black"

    # Get simplified game state
    in_check = board.is_check()

    prompt = f"""Chess position analysis - playing as {color}:

Board PGN: {create_pgn(board, get_game_result(board))}

Only Legal moves: {', '.join(board.san(move) for move in board.legal_moves)}

Instructions:
- Analyze position and select best legal move
- Return ONLY the move in algebraic notation (e.g., "e4" or "Nf3")
- No explanations or additional text"""

    return prompt

def get_last_move(board):
    """Get the last move played"""
    if not board.move_stack:
        return "None"
    last_move = board.move_stack[-1]
    game_copy = chess.Board()
    for move in board.move_stack[:-1]:
        game_copy.push(move)
    return f"{game_copy.san(last_move)}"

def get_gpt_move(prompt, retries=3, delay=1):
    """Get a move from GPT with retry logic"""
    for attempt in range(retries):
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a chess engine. Respond only with moves in algebraic notation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=5
            )

            move_str = response.choices[0].message.content.strip()
            return move_str

        except Exception as e:
            if attempt < retries - 1:
                print(f"Error getting move from GPT (attempt {attempt + 1}): {e}")
                time.sleep(delay)
            else:
                raise Exception(f"Failed to get move from GPT after {retries} attempts: {e}")

def white_get_move(board):
    """Get White's move from GPT"""
    prompt = create_board_state_prompt(board, is_white=True)
    move_str = get_gpt_move(prompt)

    try:
        move = board.parse_san(move_str)
        if move in board.legal_moves:
            return move
        else:
            raise ValueError(f"GPT suggested illegal move: {move_str}")
    except ValueError as e:
        raise Exception(f"Error processing GPT move: {e}")

def black_get_move(board):
    """Get Black's move from GPT"""
    prompt = create_board_state_prompt(board, is_white=False)
    move_str = get_gpt_move(prompt)

    try:
        move = board.parse_san(move_str)
        if move in board.legal_moves:
            return move
        else:
            raise ValueError(f"GPT suggested illegal move: {move_str}")
    except ValueError as e:
        raise Exception(f"Error processing GPT move: {e}")

def create_pgn(board, result):
    """Create a PGN string from the game"""
    game = chess.pgn.Game()

    # Set headers
    game.headers["Event"] = "GPT vs GPT Chess Game"
    game.headers["Site"] = "OpenAI GPT-4o mini"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = "1"
    game.headers["White"] = "GPT-4o mini (White)"
    game.headers["Black"] = "GPT-4o mini (Black)"
    game.headers["Result"] = result

    # Reconstruct the game
    node = game
    for move in board.move_stack:
        node = node.add_variation(move)

    # Return PGN string
    return str(game)

def get_game_result(board):
    """Determine the game result in PGN format"""
    if board.is_checkmate():
        return "1-0" if board.turn == chess.BLACK else "0-1"
    elif board.is_stalemate() or board.is_insufficient_material() or \
         board.is_fifty_moves() or board.is_repetition():
        return "1/2-1/2"
    else:
        return "*"

def play_chess():
    board = chess.Board()

    print("Starting chess game: GPT vs GPT")

    try:
        while not board.is_game_over():
            # White's turn
            print("\nWhite is thinking...")
            move = white_get_move(board)
            print(f"White plays: {board.san(move)}")
            board.push(move)

            if board.is_game_over():
                break

            # Black's turn
            print("\nBlack is thinking...")
            move = black_get_move(board)
            print(f"Black plays: {board.san(move)}")
            board.push(move)

    except Exception as e:
        print(f"\nGame ended due to error: {e}")

    # Generate result
    result = get_game_result(board)

    # Create PGN
    pgn = create_pgn(board, result)

    # Save PGN to file
    with open("gpt_chess_game.pgn", "w") as f:
        f.write(pgn)

    print("\nGame Over!")
    print(board.result())
    print(f"\nGame saved to 'gpt_chess_game.pgn'")
    print("\nYou can import this PGN file directly into Lichess at https://lichess.org/paste")

    return board, pgn

def main():
    openai.api_key = "key"
    final_board, pgn = play_chess()
    print("\nPGN Output (copy this to Lichess):")
    print("\n" + pgn)
    return final_board

if __name__ == "__main__":
    main()