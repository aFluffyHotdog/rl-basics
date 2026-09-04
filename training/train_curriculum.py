import os
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Import your newly updated environment
from decoder_env_v3 import DecoderEnvV3

class CurriculumCallback(BaseCallback):
    """
    Custom SB3 Callback that automatically increases the environment's 
    action chunk size at a specified timestep interval.
    """
    def __init__(self, step_interval: int, verbose=1):
        super().__init__(verbose)
        self.step_interval = step_interval

    def _on_step(self) -> bool:
        # Check if we have hit the interval threshold
        if self.n_calls % self.step_interval == 0:
            if self.verbose > 0:
                print(f"\n[Curriculum] Timestep {self.num_timesteps}: Increasing action chunk size!")
            
            # env_method() broadcasts the function call to the environment inside the VecEnv
            self.training_env.env_method("increase_curriculum")
            
        return True

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    progress_remaining starts at 1.0 and decreases to 0.0 at the end of training.
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def main():
    # --- Configuration ---
    DATA_DIR = "./sample_data_9_lanes"  # Replace with the path to your dataset folders
    TOTAL_TIMESTEPS = 60_000_000
    CURRICULUM_INTERVAL = 200_000 # Increase chunk size every 500k steps
    
    # --- 1. Single Environment Setup ---
    print("Initializing Single Environment...")
    env_1 = DecoderEnvV2(data_dir=DATA_DIR, is_eval=False)
    
    # Wrap it in a DummyVecEnv. This is required for the Callback to use env_method()
    vec_env = DummyVecEnv([lambda: env_1])
    
    # --- 2. Callback Setup ---
    curriculum_callback = CurriculumCallback(step_interval=CURRICULUM_INTERVAL)
    
    # --- 3. Model Initialization ---
    print("Initializing PPO Model...")
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        learning_rate=linear_schedule(3e-4), 
        ent_coef=0.01,                       
        verbose=1,
        tensorboard_log="./ppo_decoder_logs/"
    )
    
    # --- 4. Training Loop ---
    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS, 
        callback=curriculum_callback,
        progress_bar=True
    )
    
    # --- 5. Save Model ---
    model_path = "ppo_decoder_chunking_single_env"
    model.save(model_path)
    print(f"Training complete! Model saved to {model_path}.zip")

if __name__ == "__main__":
    main()