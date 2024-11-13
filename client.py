from typing import List, Dict, Tuple
import os
from pathlib import Path
import matplotlib.pyplot as plt
from statistics import mean
from stockfish_vs_llm import play_chess_game, ChessConfig, LMConfig

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

def plot_comparative_results(results):
    """
    Plot comparative results for different models playing as both white and black.

    Args:
        results: Dict of format {model_name_color: [list of game stats]}
    """
    # Prepare data for plotting
    model_stats = {}
    for model_key, games in results.items():
        if not games:  # Skip if no games were played
            continue

        model_name = model_key.rsplit('_', 1)[0]  # Remove _white/_black suffix
        color = model_key.split('_')[-1]

        # Calculate average centipawn loss across all games
        avg_losses = [game['llm_avg_centipawn_loss'] for game in games]
        mean_loss = mean(avg_losses)
        std_loss = (sum((x - mean_loss) ** 2 for x in avg_losses) / len(avg_losses)) ** 0.5

        if model_name not in model_stats:
            model_stats[model_name] = {}
        model_stats[model_name][color] = {
            'mean': mean_loss,
            'std': std_loss,
            'games': len(games)
        }

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Chess Model Performance vs Stockfish\nLower is Better', fontsize=14)

    # Plot white games
    models = list(model_stats.keys())
    x_pos = range(len(models))
    white_means = [model_stats[model]['white']['mean'] for model in models]
    white_stds = [model_stats[model]['white']['std'] for model in models]

    ax1.bar(x_pos, white_means, yerr=white_stds, capsize=5, alpha=0.8)
    ax1.set_ylabel('Average Centipawn Loss')
    ax1.set_title('Playing as White')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=45, ha='right')

    # Plot black games
    black_means = [model_stats[model]['black']['mean'] for model in models]
    black_stds = [model_stats[model]['black']['std'] for model in models]

    ax2.bar(x_pos, black_means, yerr=black_stds, capsize=5, alpha=0.8, color='gray')
    ax2.set_ylabel('Average Centipawn Loss')
    ax2.set_title('Playing as Black')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, rotation=45, ha='right')

    plt.tight_layout()

    # Save the plot
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'chess_performance_comparison.png', dpi=300, bbox_inches='tight')

    # Print summary statistics
    print("\nPerformance Summary:")
    print("-" * 80)
    print(f"{'Model':<25} {'Color':<8} {'Avg Loss':>10} {'Std Dev':>10} {'Games':>8}")
    print("-" * 80)

    for model in models:
        for color in ['white', 'black']:
            stats = model_stats[model][color]
            print(f"{model:<25} {color:<8} {stats['mean']:>10.2f} {stats['std']:>10.2f} {stats['games']:>8}")

    return model_stats

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

    model_stats = plot_comparative_results(results)

    # Print overall conclusions
    print("\nOverall Conclusions:")
    print("-" * 80)
    best_white = min((stats['white']['mean'], model) for model, stats in model_stats.items())
    best_black = min((stats['black']['mean'], model) for model, stats in model_stats.items())

    print(f"Best performance as White: {best_white[1]} (Average centipawn loss: {best_white[0]:.2f})")
    print(f"Best performance as Black: {best_black[1]} (Average centipawn loss: {best_black[0]:.2f})")

if __name__ == "__main__":
    main()