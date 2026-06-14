import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_makespan_over_episodes(log_file=None):
    """
    Plot the makespan over episodes from a training log file.
    
    Args:
        log_file: Path to the log file. If None, uses the most recent log file.
    """
    # If no log file specified, find the most recent one
    if log_file is None:
        log_dir = Path("logs")
        if not log_dir.exists():
            print("No logs directory found. Run training first.")
            return
        
        log_files = sorted(log_dir.glob("training_log_*.txt"))
        if not log_files:
            print("No log files found in logs directory.")
            return
        
        log_file = log_files[-1]
        print(f"Using log file: {log_file}")
    
    # Read the CSV file
    df = pd.read_csv(log_file)
    
    # Create figure and axis
    plt.figure(figsize=(12, 6))
    plt.plot(df['episode'], df['makespan'], linewidth=1, alpha=0.7)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Makespan (time units)', fontsize=12)
    plt.title('Makespan Over Episodes', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add a rolling average line
    window = 100
    df['makespan_rolling_avg'] = df['makespan'].rolling(window=window, center=True).mean()
    plt.plot(df['episode'], df['makespan_rolling_avg'], linewidth=2, 
             label=f'{window}-episode rolling average', color='red')
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Save the plot
    plot_file = log_file.parent / f"makespan_plot_{log_file.stem.split('_', 2)[2]}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    
    plt.show()


def plot_reward_over_episodes(log_file=None):
    """
    Plot the reward over episodes from a training log file.
    
    Args:
        log_file: Path to the log file. If None, uses the most recent log file.
    """
    # If no log file specified, find the most recent one
    if log_file is None:
        log_dir = Path("logs")
        if not log_dir.exists():
            print("No logs directory found. Run training first.")
            return
        
        log_files = sorted(log_dir.glob("training_log_*.txt"))
        if not log_files:
            print("No log files found in logs directory.")
            return
        
        log_file = log_files[-1]
        print(f"Using log file: {log_file}")
    
    # Read the CSV file
    df = pd.read_csv(log_file)
    
    # Create figure and axis
    plt.figure(figsize=(12, 6))
    plt.plot(df['episode'], df['reward'], linewidth=1, alpha=0.7)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Episode Reward', fontsize=12)
    plt.title('Reward Over Episodes', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add a rolling average line
    window = 100
    df['reward_rolling_avg'] = df['reward'].rolling(window=window, center=True).mean()
    plt.plot(df['episode'], df['reward_rolling_avg'], linewidth=2, 
             label=f'{window}-episode rolling average', color='red')
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Save the plot
    plot_file = log_file.parent / f"reward_plot_{log_file.stem.split('_', 2)[2]}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    
    plt.show()


def plot_all_metrics(log_file=None):
    """
    Plot all metrics (makespan, reward, epsilon) in a single figure.
    
    Args:
        log_file: Path to the log file. If None, uses the most recent log file.
    """
    # If no log file specified, find the most recent one
    if log_file is None:
        log_dir = Path("logs")
        if not log_dir.exists():
            print("No logs directory found. Run training first.")
            return
        
        log_files = sorted(log_dir.glob("training_log_*.txt"))
        if not log_files:
            print("No log files found in logs directory.")
            return
        
        log_file = log_files[-1]
        print(f"Using log file: {log_file}")
    
    # Read the CSV file
    df = pd.read_csv(log_file)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Makespan
    axes[0].plot(df['episode'], df['makespan'], linewidth=1, alpha=0.7, color='blue')
    axes[0].set_ylabel('Makespan', fontsize=11)
    axes[0].set_title('Training Metrics Over Episodes', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Reward
    axes[1].plot(df['episode'], df['reward'], linewidth=1, alpha=0.7, color='green')
    axes[1].set_ylabel('Reward', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Epsilon
    axes[2].plot(df['episode'], df['epsilon'], linewidth=1, alpha=0.7, color='red')
    axes[2].set_xlabel('Episode', fontsize=11)
    axes[2].set_ylabel('Epsilon', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = log_file.parent / f"all_metrics_plot_{log_file.stem.split('_', 2)[2]}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    
    plt.show()


if __name__ == "__main__":
    # Plot all available metrics
    plot_all_metrics()
    
    # Or plot individual metrics:
    # plot_makespan_over_episodes()
    # plot_reward_over_episodes()
