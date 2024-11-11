from typing import List, Dict, Tuple
import os
from pathlib import Path
import matplotlib.pyplot as plt
from statistics import mean
from rated_stockfish_vs_llm import play_chess_game, ChessConfig, LMConfig

import argparse

def run_openai_model(model_name: str, trials: int, stockfish_path: str = "/opt/homebrew/bin/stockfish") -> Tuple[List[Dict], List[Dict]]:
    """Run chess games for OpenAI model playing as both white and black."""
    output_dir = Path("output") / model_name.replace(".", "-")
    output_dir.mkdir(parents=True, exist_ok=True)

    white_results = []
    black_results = []

    # Run white games
    for trial in range(trials):
        chess_config = ChessConfig(
            stockfish_path=stockfish_path,
            verbose=False,
            output_path=str(output_dir / f"{trial}_white.pgn")
        )
        lm_config = LMConfig(
            model_name=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            is_white=True,
            type="openai"
        )
        stats = play_chess_game(chess_config, lm_config)
        if stats:
          white_results.append(stats)

    # Run black games
    for trial in range(trials):
        chess_config = ChessConfig(
            stockfish_path=stockfish_path,
            verbose=False,
            output_path=str(output_dir / f"{trial}_black.pgn")
        )
        lm_config = LMConfig(
            model_name=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            is_white=False,
            type="openai"
        )
        stats = play_chess_game(chess_config, lm_config)
        if stats:
          black_results.append(stats)

    return white_results, black_results

def run_anthropic_model(model_name: str, trials: int, stockfish_path: str = "/opt/homebrew/bin/stockfish") -> Tuple[List[Dict], List[Dict]]:
    """Run chess games for Anthropic model playing as both white and black."""
    output_dir = Path("output") / model_name.replace(".", "-")
    output_dir.mkdir(parents=True, exist_ok=True)

    white_results = []
    black_results = []

    # Run white games
    for trial in range(trials):
        chess_config = ChessConfig(
            stockfish_path=stockfish_path,
            verbose=False,
            output_path=str(output_dir / f"{trial}_white.pgn")
        )
        lm_config = LMConfig(
            model_name=model_name,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            is_white=True,
            type="anthropic"
        )
        stats = play_chess_game(chess_config, lm_config)
        if stats:
          white_results.append(stats)

    # Run black games
    for trial in range(trials):
        chess_config = ChessConfig(
            stockfish_path=stockfish_path,
            verbose=False,
            output_path=str(output_dir / f"{trial}_black.pgn")
        )
        lm_config = LMConfig(
            model_name=model_name,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            is_white=False,
            type="anthropic"
        )
        stats = play_chess_game(chess_config, lm_config)
        if stats:
          black_results.append(stats)

    return white_results, black_results

def plot_comparative_results(results: Dict[str, List[Dict]]) -> None:
    """Create comparative box plots of model performance."""
    metrics = {
        'Estimated Rating': lambda x: x['estimated_rating'],
        'Good Moves %': lambda x: (x['good_moves'] / x['total_moves'] * 100),
        'Blunder Rate %': lambda x: (x['blunders'] / x['total_moves'] * 100),
        'Avg Centipawn Loss': lambda x: x['avg_centipawn_loss']
    }

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, y=0.95)

    for (i, j), (metric_name, metric_func) in zip(
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        metrics.items()
    ):
        ax = axs[i, j]
        model_data = []
        model_names = []
        colors = []

        for model_key, stats_list in results.items():
            model_names.append(model_key)
            values = [metric_func(stat) for stat in stats_list]
            model_data.append(values)
            colors.append('lightblue' if 'white' in model_key else 'lightgreen')

        bp = ax.boxplot(model_data, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_title(metric_name)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Run chess model comparisons')
    parser.add_argument('--trials', '-t', type=int, default=5, help='Number of trials per model')
    parser.add_argument('--stockfish-path', type=str, default='/opt/homebrew/bin/stockfish',
                      help='Path to Stockfish executable')
    args = parser.parse_args()

    openai_models = [
      # 'o1-preview',
      # 'o1-mini',
      # 'gpt-4o-mini',
      # 'gpt-4o',
      'gpt-4-turbo'
    ]

    anthropic_models = [
        'claude-3-5-sonnet-latest',
        # 'claude-3-5-haiku-latest'
    ]

    results = {}

    # Run OpenAI models
    for model in openai_models:
        print(f"\nTesting {model}...")
        white_results, black_results = run_openai_model(model, args.trials, args.stockfish_path)
        results[f"{model}_white"] = white_results
        results[f"{model}_black"] = black_results

    # Run Anthropic models
    for model in anthropic_models:
        print(f"\nTesting {model}...")
        white_results, black_results = run_anthropic_model(model, args.trials, args.stockfish_path)
        results[f"{model}_white"] = white_results
        results[f"{model}_black"] = black_results

    # Plot comparative results
    plot_comparative_results(results)

    # Print summary statistics
    print("\nSummary Statistics:")
    for model_key, stats_list in results.items():
        print(f"\n{model_key}:")
        print(f"Average Good Move %: {mean(stat['good_moves']/stat['total_moves']*100 for stat in stats_list):.1f}%")
        print(f"Average Blunder %: {mean(stat['blunders']/stat['total_moves']*100 for stat in stats_list):.1f}%")
        print(f"Average Centipawn Loss: {mean(stat['avg_centipawn_loss'] for stat in stats_list):.1f}")

if __name__ == "__main__":
    main()