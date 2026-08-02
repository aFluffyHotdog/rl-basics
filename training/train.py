import sys
import os
import argparse
import numpy as np
import gymnasium as gym
from typing import Callable
from datetime import datetime

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

# Import Action Masking modules
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# Add parent directory to path so imports work from any location
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.decoder_env_v2 import DecoderEnvV2, NUM_DECODERS

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def resume_training(checkpoint_path: str, env_setup, log_file: str, additional_timesteps: int):
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Must use MaskablePPO.load now!
    custom_objects = {"learning_rate": linear_schedule(3e-4)}
    model = MaskablePPO.load(checkpoint_path, env=env_setup, custom_objects=custom_objects, tensorboard_log="./logs/tensorboard")
    
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

class EpisodeStatsCallback(BaseCallback):
    def __init__(self, log_file):
        super().__init__()
        self.log_file = log_file
        self.episode_count = 0
        self.action_counts = np.zeros(NUM_DECODERS) 
        
    def _init_callback(self):
        with open(self.log_file, "w") as f:
            f.write("step,episode_reward,episode_length,makespan\n")
    
    def _on_step(self):
        action = self.locals["actions"][0]
        self.action_counts[action] += 1

        if self.num_timesteps % 2048 == 0:
            total_actions = np.sum(self.action_counts)
            if total_actions > 0:
                for i in range(NUM_DECODERS):
                    pct = (self.action_counts[i] / total_actions) * 100
                    self.logger.record(f"exploration/decoder_{i}_pct", pct)

        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_count += 1
                episode_reward = info["episode"]["r"]
                episode_length = info["episode"]["l"]
                makespan = info.get("makespan", episode_length)
                decoder_times = info.get("decoder_times", [0] * NUM_DECODERS) 
                
                self.logger.record("env/makespan", makespan)
                for i, time_val in enumerate(decoder_times):
                    self.logger.record(f"decoder_loads/decoder_{i}", time_val)
                
                with open(self.log_file, "a") as f:
                    f.write(f"{self.num_timesteps},{episode_reward:.4f},{episode_length},{makespan}\n")
                
                if self.episode_count % 100 == 0:
                    print(f"\nStep {self.num_timesteps} | Episode {self.episode_count}")
                    print(f"   Makespan: {makespan} | Reward: {episode_reward:.3f}")
                    self.action_counts = np.zeros(NUM_DECODERS)
        return True

def mask_fn(env: gym.Env) -> np.ndarray:
    """Helper function required by ActionMasker to pull masks from the base env."""
    return env.action_masks()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, default='sample_data', help="Path to dataset folder")
    ap.add_argument('--train_split_pct', type=float, default=0.8, help="Split data for training vs eval")
    ap.add_argument('--is_eval', type=bool, default=False)
    ap.add_argument('--resume_path', type=str, default=None, help="Path to checkpoint model to resume")
    args = ap.parse_args()

    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_log_{timestamp}.txt"
    model_file = f"models/ppo_agent_{timestamp}"
    checkpoint_dir = f"models/checkpoints_{timestamp}/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Hyperparameters
    learning_rate = linear_schedule(3e-4)         
    n_steps = 4096                
    batch_size = 128
    ent_coef = 0.01               
    n_epochs = 10                 
    gamma = 0.99                  
    gae_lambda = 0.95             
    clip_range = 0.2              
    total_timesteps = 180_000_000 
    
    # Create Env -> ActionMasker -> Monitor -> DummyVecEnv -> VecNormalize
    def make_env():
        env = DecoderEnvV2(data_dir=args.data_dir, train_split_pct=args.train_split_pct, is_eval=args.is_eval)
        env = ActionMasker(env, mask_fn) # Wrap with Action Masking
        return Monitor(env)
        
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    if args.resume_path:
        model = resume_training(args.resume_path, env, log_file, total_timesteps)
    else:
        # Increase the network capacity to process the 64-token lookahead window
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

        # Initialize Maskable PPO agent
        model = MaskablePPO(
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
            tensorboard_log="./logs/tensorboard",
            policy_kwargs=policy_kwargs
        )

        print(f"Starting MASKABLE PPO training on dataset: {args.data_dir}")
        print(f"Logging to: {log_file}")
        
        stats_callback = EpisodeStatsCallback(log_file)
        checkpoint_callback = CheckpointCallback(
            save_freq=1_000_000, 
            save_path=checkpoint_dir,
            name_prefix="ppo_makespan"
        )
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList([stats_callback, checkpoint_callback]),
            progress_bar=True
        )

        print("Training complete!")
        model.save(model_file)