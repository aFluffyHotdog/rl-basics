import sys
import os
import argparse
import csv
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Add parent directory to path so imports work from any location
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.decoder_env_v2 import DecoderEnvV2, NUM_DECODERS

def run_simulation(env, raw_env, is_naive=False, model=None):
    """
    Runs a single simulation episode until completion or deadlock.
    Returns the total cycles (makespan) or 'DEADLOCK'.
    """
    obs = env.reset()
    done = False
    step_idx = 0
    
    while not done:
        if is_naive:
            # Naive round-robin scheduling (0, 1, 2, ..., 7, 0, 1...)
            action = np.array([step_idx % NUM_DECODERS])
        else:
            # Model prediction (deterministic mode for static scheduling)
            # Fetch masks from the raw environment and shape them for the DummyVecEnv batch
            action_masks = np.array([raw_env.action_masks()])
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
            
        obs, rewards, dones, infos = env.step(action)
        done = dones[0]
        step_idx += 1
        
        # Check if the environment safely intercepted a deadlock
        if infos[0].get("deadlock", False):
            return "DEADLOCK"
            
    # Because 1 step = 1 clock cycle, the step count is exactly the makespan
    return step_idx

def compare_model_vs_naive(model_path: str, data_dir: str, output_csv: str):
    print(f"Loading environment to discover datasets in: {data_dir}")
    
    # Instantiate with train_split_pct=1.0 to load ALL datasets into the pool
    raw_env = DecoderEnvV2(data_dir=data_dir, train_split_pct=1.0, is_eval=False, seed=42)
    env = DummyVecEnv([lambda: raw_env])
    
    print(f"Loading trained model from: {model_path}")
    try:
        model = MaskablePPO.load(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Recover the dataset names in the exact shuffled order the env loaded them
    folder_names = []
    for test_folder in sorted(Path(data_dir).iterdir()):
        if test_folder.is_dir() and (test_folder / "beats_hex").is_dir():
            folder_names.append(test_folder.name)
            
    rng = random.Random(42) # Default seed used in DecoderEnvV2
    rng.shuffle(folder_names)
    
    # Store the original loaded pool
    original_dataset_pool = raw_env.dataset_pool.copy()
    results = []

    print("\nStarting evaluation across all datasets...")
    
    for i, dataset in enumerate(tqdm(original_dataset_pool, desc="Evaluating Datasets")):
        dataset_name = folder_names[i]
        
        # Hack to force the environment to use this specific dataset on reset
        raw_env.dataset_pool = [dataset]
        
        # Run Model Simulation
        model_cycles = run_simulation(env, raw_env, is_naive=False, model=model)
        
        # Run Naive Simulation
        naive_cycles = run_simulation(env, raw_env, is_naive=True, model=None)
        
        # Calculate Improvement
        if isinstance(model_cycles, int) and isinstance(naive_cycles, int):
            improvement = ((naive_cycles - model_cycles) / naive_cycles) * 100
            imp_str = f"{improvement:.2f}%"
        elif model_cycles == "DEADLOCK" and naive_cycles == "DEADLOCK":
            imp_str = "Both Deadlocked"
        elif model_cycles == "DEADLOCK":
            imp_str = "Model Failed"
        elif naive_cycles == "DEADLOCK":
            imp_str = "Naive Failed"
            
        results.append({
            "Dataset": dataset_name,
            "Model_Cycles": model_cycles,
            "Naive_Cycles": naive_cycles,
            "Improvement": imp_str
        })

    print(f"\nWriting results to: {output_csv}")
    try:
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Dataset", "Model_Cycles", "Naive_Cycles", "Improvement_Pct"])
            
            for res in results:
                writer.writerow([
                    res["Dataset"], 
                    res["Model_Cycles"], 
                    res["Naive_Cycles"], 
                    res["Improvement"]
                ])
        print("Done! Evaluation complete.")
    except Exception as e:
        print(f"Failed to write CSV: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare PPO Model vs Naive Round-Robin scheduling.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained .zip model file.")
    parser.add_argument("--data_dir", type=str, default="sample_data",
                        help="Path to the top-level folder containing all test image folders.")
    parser.add_argument("--output", type=str, default="model_vs_naive_results.csv",
                        help="Path to save the generated CSV comparison file.")
    
    args = parser.parse_args()
    
    compare_model_vs_naive(args.model_path, args.data_dir, args.output)