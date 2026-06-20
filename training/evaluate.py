import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.decoder_env import DecoderEnv

def denormalize_observation(obs, true_env):
    """
    Helper function to convert a normalized observation dictionary back into 
    raw values for human debugging and logging, without breaking the model.
    """
    # Recalculate the scaling factors used in the environment
    max_possible_time = np.sum(true_env.packets) + 1.0
    max_packet_cost = 20.0 # From your randint(1, 20)
    
    return {
        "raw_decoder_time_left": np.round(obs["decoder_time_left"] * max_possible_time).astype(int),
        "raw_next_cost": np.round(obs["next_cost"] * max_packet_cost).astype(int),
        "raw_packets_remaining": np.round(obs["packets_remaining"] * true_env.num_packets).astype(int),
    }

def evaluate_ppo_model(model_path=None, env_config=None):
    """
    Load a PPO model and evaluate it on the DecoderEnv to get the final makespan.
    """
    # Default environment config
    if env_config is None:
        env_config = {'width': 200, 'height': 200, 'comp_ratio': 2}
    
    # If no model path specified, find the most recent one
    if model_path is None:
        model_dir = Path("models")
        if not model_dir.exists():
            print("No models directory found. Train a model first.")
            return None
        
        model_files = sorted(model_dir.glob("ppo_agent_*.zip"))
        # Filter out the 'resumed' or 'finished' if you want a specific one, 
        # but [-1] grabs the latest chronologically if sorted by timestamp.
        if not model_files:
            print("No PPO model files found in models directory.")
            return None
        
        model_path = model_files[-1]
    
    print(f"Loading PPO model from: {model_path}")
    model = PPO.load(model_path)
    
    # Create the raw environment (No DummyVecEnv or VecNormalize needed now!)
    env = DecoderEnv(**env_config)
    
    # Run a single episode with the model
    obs, info = env.reset()
    done = False
    step_count = 0
    
    print("\n--- Starting Evaluation Episode ---")
    
    while not done:
        # 1. Model predicts using the NORMALIZED observation
        action, _states = model.predict(obs, deterministic=True)
        
        # 2. Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        
        # Optional: Print the first few steps to verify denormalization works
        if step_count <= 3:
            raw_obs = denormalize_observation(obs, env)
            print(f"Step {step_count}: Model assigned packet to Decoder {action}.")
            print(f"   Normalized State seen by model: {obs['decoder_time_left']}")
            print(f"   Raw State evaluated by helper:  {raw_obs['raw_decoder_time_left']}")

    
    # Calculate final makespan directly from the underlying environment variables
    ppo_makespan = max(env.decoder_time_left)
    
    # Calculate greedy baseline for comparison
    env_greedy = DecoderEnv(**env_config)
    # Re-using your environment's packets to ensure a fair comparison
    env_greedy.packets = env.packets.copy() 
    env_greedy.num_packets = env.num_packets
    
    greedy_done = False
    greedy_obs, _ = env_greedy.reset()
    while not greedy_done:
        # Pure greedy heuristic: pick the decoder with the minimum time
        best_action = np.argmin(env_greedy.decoder_time_left)
        _, _, g_terminated, g_truncated, _ = env_greedy.step(best_action)
        greedy_done = g_terminated or g_truncated
        
    greedy_makespan = max(env_greedy.decoder_time_left)
    
    # Calculate performance ratio
    ratio = ppo_makespan / greedy_makespan if greedy_makespan > 0 else float('inf')
    
    print(f"\n=== PPO Model Evaluation ===")
    print(f"Model: {model_path}")
    print(f"Steps taken: {step_count}")
    print(f"PPO Makespan: {ppo_makespan}")
    print(f"Greedy Makespan (Baseline): {greedy_makespan}")
    print(f"Performance Ratio (PPO/Greedy): {ratio:.4f}")
    print(f"PPO Final Decoder times: {env.decoder_time_left}")
    print(f"Greedy Final Decoder times: {env_greedy.decoder_time_left}")
    print(f"============================\n")
    
    return ppo_makespan

if __name__ == "__main__":
    evaluate_ppo_model()