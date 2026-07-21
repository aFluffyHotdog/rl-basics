import sys
import os
import argparse
import numpy as np
import gymnasium as gym
from typing import Callable
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

# Add parent directory to path so imports work from any location
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.decoder_env_v2 import DecoderEnvV2, NUM_DECODERS

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def resume_training(checkpoint_path: str, env_setup, log_file: str, additional_timesteps: int):
    """
    Resumes training from a saved checkpoint model.
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    model = PPO.load(checkpoint_path, env=env_setup, tensorboard_log="./logs/tensorboard")
    
    checkpoint_dir = os.path.dirname(checkpoint_path)
    stats_callback = EpisodeStatsCallback(log_file)
    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000, 
        save_path=checkpoint_dir,
        name_prefix="ppo_makespan_resumed"
    )
    callback_list = CallbackList([stats_callback, checkpoint_callback])
    
    print(f"Resuming training for {additional_timesteps} timesteps...")
    
    model.learn(
        total_timesteps=additional_timesteps, 
        callback=callback_list, 
        reset_num_timesteps=False,
        progress_bar=True
    )
    
    final_save_path = checkpoint_path.replace(".zip", "_finished.zip")
    model.save(final_save_path)
    print(f"Resumed training complete! Saved to {final_save_path}")
    
    return model

# ---------------------------------------------------------
# ADVANCED TELEMETRY CALLBACK
# ---------------------------------------------------------
class EpisodeStatsCallback(BaseCallback):
    def __init__(self, log_file):
        super().__init__()
        self.log_file = log_file
        self.episode_count = 0
        self.action_counts = np.zeros(NUM_DECODERS) # Track how often each decoder is picked
        
    def _init_callback(self):
        with open(self.log_file, "w") as f:
            f.write("step,episode_reward,episode_length,makespan\n")
    
    def _on_step(self):
        # 1. Track which action the agent took in this exact step
        action = self.locals["actions"][0]
        self.action_counts[action] += 1

        # 2. Log exploration stats continuously so TensorBoard has data immediately
        if self.num_timesteps % 2048 == 0:
            total_actions = np.sum(self.action_counts)
            if total_actions > 0:
                for i in range(NUM_DECODERS):
                    pct = (self.action_counts[i] / total_actions) * 100
                    self.logger.record(f"exploration/decoder_{i}_pct", pct)

        # 3. Check if the episode is done
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_count += 1
                
                # Extract data
                episode_reward = info["episode"]["r"]
                episode_length = info["episode"]["l"]
                
                makespan = episode_length
                decoder_times = info.get("decoder_times", [0] * NUM_DECODERS) 
                
                # ==========================================
                # TENSORBOARD LOGGING (End of Episode)
                # ==========================================
                # Log global makespan
                self.logger.record("env/makespan", makespan)
                
                # Log individual final loads for all 8 decoders
                for i, time_val in enumerate(decoder_times):
                    self.logger.record(f"decoder_loads/decoder_{i}", time_val)
                
                # ==========================================
                # CSV AND CONSOLE LOGGING
                # ==========================================
                with open(self.log_file, "a") as f:
                    f.write(f"{self.num_timesteps},{episode_reward:.4f},{episode_length},{makespan}\n")
                
                # Print debug stats to console every 100 episodes
                if self.episode_count % 100 == 0:
                    print(f"\nStep {self.num_timesteps} | Episode {self.episode_count}")
                    print(f"   Makespan: {makespan} | Reward: {episode_reward:.3f}")
                    
                    total_actions = np.sum(self.action_counts)
                    # Format a neat string showing percentage of usage
                    dist_str = " | ".join([f"D{i}:{int(pct)}%" for i, pct in enumerate((self.action_counts/total_actions)*100)])
                    print(f"   Usage Spread: {dist_str}")
                    print(f"   Final Loads:  {decoder_times}")
                    
                    # Reset action counts so we only see RECENT exploration behavior
                    self.action_counts = np.zeros(NUM_DECODERS)
                    
        return True

if __name__ == "__main__":
    # Add argparse to allow running with different datasets from the command line
    ap = argparse.ArgumentParser()
    ap.add_argument('--cmd_path', type=str, default='sample_data/test_11_lantern/beats_hex',
                    help="Path to the hex folder dataset for the environment.")
    args = ap.parse_args()

    # Create logs and models directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_log_{timestamp}.txt"
    model_file = f"models/ppo_agent_{timestamp}"
    checkpoint_dir = f"models/checkpoints_{timestamp}/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Hyperparameters for PPO
    learning_rate = linear_schedule(3e-4)         
    n_steps = 4096                
    batch_size = 128
    ent_coef = 0.01               
    n_epochs = 10                 
    gamma = 0.99                  
    gae_lambda = 0.95             
    clip_range = 0.2              
    total_timesteps = 30_000_000      
    
    # Wrapped DecoderEnvV2 in Monitor
    # Swapped initialization parameters to use the hex path
    env = DummyVecEnv([lambda: Monitor(DecoderEnvV2(cmd_path=args.cmd_path))])
    
    # Bring back VecNormalize ONLY for rewards to prevent gradient explosion!
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    
    # Initialize PPO agent
    model = PPO(
        policy="MultiInputPolicy",  
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        ent_coef=ent_coef,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        verbose=1,
        tensorboard_log="./logs/tensorboard"
    )

    print(f"Starting PPO training on dataset: {args.cmd_path}")
    print(f"Logging to: {log_file}")
    
    # Initialize both callbacks
    stats_callback = EpisodeStatsCallback(log_file)
    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000, 
        save_path=checkpoint_dir,
        name_prefix="ppo_makespan"
    )
    callback_list = CallbackList([stats_callback, checkpoint_callback])
    
    # Train the model with the combined callback list
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        progress_bar=True
    )

    print("Training complete!")
    print(f"Final model saved to: {model_file}")
    model.save(model_file)