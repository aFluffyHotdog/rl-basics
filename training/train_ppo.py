import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# 1. Register or initialize your environment
# (Replace 'CartPole-v1' with your custom scheduling environment ID or class)
env_id = "CartPole-v1"

# 2. Vectorize the environment for faster on-policy training
# PPO relies heavily on parallel data collection. Running 4 instances in 
# parallel helps stabilize the policy gradient updates.
vec_env = make_vec_env(env_id, n_envs=4)

# 3. Instantiate the PPO Agent
# 'MlpPolicy' creates a standard Multi-Layer Perceptron neural network.
# You can tweak learning_rate, batch_size, and n_steps (rollout buffer size).
model = PPO(
    policy="MlpPolicy", 
    env=vec_env, 
    verbose=1,
    learning_rate=3e-4,
    batch_size=64,
    n_steps=2048
)

# 4. Train the agent
print("Starting training...")
model.learn(total_timesteps=100_000)
print("Training complete!")

# 5. Save the trained weights
model.save("ppo_scheduling_agent")

# 6. Evaluate the trained policy
# Always evaluate on a single, isolated environment instance
eval_env = gym.make(env_id, render_mode="human")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# 7. Close the environments clean
vec_env.close()
eval_env.close()