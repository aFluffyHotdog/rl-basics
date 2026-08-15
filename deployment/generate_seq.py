import sys
import os
import argparse
import numpy as np
from tqdm import tqdm
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Add parent directory to path so imports work from any location
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.decoder_env_v2 import DecoderEnvV2, NUM_DECODERS

def generate_sequence(model_path: str, data_dir: str, output_path: str):
    print(f"Loading environment with target dataset from: {data_dir}")
    
    # Initialize env in inference mode to specifically target a single dataset
    raw_env = DecoderEnvV2(data_dir=data_dir, is_inference=True)
    env = DummyVecEnv([lambda: raw_env])
    
    print(f"Loading trained MaskablePPO model from: {model_path}")
    try:
        model = MaskablePPO.load(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("\nStarting generation...")
    obs = env.reset()
    done = False
    
    schedule = []
    
    # Setup variables to track deadlocks and progress
    initial_cmds = sum(len(q) for q in raw_env.cmds)
    last_remaining_cmds = initial_cmds
    cycles_stalled = 0
    
    pbar = tqdm(total=initial_cmds, desc="Routing Commands", unit="cmd")
    
    while not done:
        # Fetch the dynamic mask from the environment
        action_masks = np.array([raw_env.action_masks()])
        
        # deterministic=True forces the network to pick the optimal route instead of exploring
        action, _states = model.predict(obs, deterministic=True, action_masks=action_masks)
        action_scalar = int(action[0])
        
        obs, rewards, dones, infos = env.step(action)
        done = dones[0]
        
        # Only record the action in the final schedule if data actually crossed the bus!
        # This ignores stalled cycles and turns the schedule into a robust, ordered data log.
        if infos[0].get("tokens_pushed", 0) > 0:
            schedule.append(action_scalar)
        
        if done:
            pbar.update(last_remaining_cmds)
            break
            
        # Update progress and check for deadlocks
        current_remaining_cmds = sum(len(q) for q in raw_env.cmds)
        consumed = last_remaining_cmds - current_remaining_cmds
        
        if consumed > 0:
            pbar.update(consumed)
            cycles_stalled = 0
        else:
            cycles_stalled += 1
            
        if cycles_stalled >= 500:
            pbar.close()
            print("\n[ERROR] Model hit a deadlock! Cannot generate a complete sequence.")
            return
            
        last_remaining_cmds = current_remaining_cmds
        
    pbar.close()
        
    final_info = infos[0]
    makespan = final_info.get("makespan", len(schedule))
    
    print("\n" + "="*50)
    print("SEQUENCE GENERATION COMPLETE")
    print("="*50)
    print(f"Total Cycles (Makespan): {makespan}")
    print(f"Total Actions Scheduled: {len(schedule)}")
        
    # Write the schedule line-by-line using newlines
    try:
        with open(output_path, 'w') as f:
            for a in schedule:
                f.write(f"{a}\n")
        print(f"\nStatic routing sequence successfully saved to: {output_path}")
    except Exception as e:
        print(f"\nFailed to save sequence: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a line-by-line schedule sequence for udp_send_beats.py")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained .zip model file.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the specific image folder (should contain the 'beats_hex' directory).")
    parser.add_argument("--output", type=str, default="generated_sequence.txt",
                        help="Path to save the generated sequence text file.")
    
    args = parser.parse_args()
    
    generate_sequence(args.model_path, args.data_dir, args.output)

# USAGE: python3  .\deployment\generate_seq.py --model_path models\checkpoints_20260811_111851\ppo_makespan_resumed_33000000_steps.zip --data_dir sample_data_9_lanes\test_11_lantern\beats_hex
