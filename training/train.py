import sys
import os
import pickle
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path so imports work from any location
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.decoder_env import DecoderEnv
from agents.q_agent import DecoderAgent

if __name__ == "__main__":
    # Create logs and models directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_log_{timestamp}.txt"
    model_file = f"models/agent_{timestamp}.pkl"
    
    # hyperparameters should go here
    learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
    n_episodes = 10000        # Number of hands to practice
    start_epsilon = 1.0         # Start with 100% random actions
    epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
    final_epsilon = 0.1         # Always keep some exploration

    # Initialize environment
    env = DecoderEnv(width=200, height=200, comp_ratio=2)

    # Initialize agent
    agent = DecoderAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=0.99
    )

    # Training loop
    print("Starting training...")
    print(f"Logging to: {log_file}")
    print(f"Model will be saved to: {model_file}")
    episode_rewards = []

    # Write header to log file
    with open(log_file, "w") as f:
        f.write("episode,reward,makespan,epsilon\n")

    for episode in range(n_episodes):
        obs, info = env.reset()
        state = agent.get_state(obs)
        episode_reward = 0
        
        # Progress bar for packets in this episode
        pbar = tqdm(total=env.num_packets, desc=f"Episode {episode + 1}", leave=False)
        while True:
            # Choose and perform action
            action = agent.get_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = agent.get_state(next_obs)

            # Print decoder info
            # print(f"Packet sent to decoder {action} | Decoder times: {env.decoder_time_left}")

            # Update agent
            agent.update_q_table(state, action, reward, next_state)

            episode_reward += reward
            state = next_state

            # Update progress bar
            pbar.update(1)
            if terminated or truncated:
                break

        pbar.close()
        
        # Decay epsilon
        agent.epsilon = max(final_epsilon, agent.epsilon - epsilon_decay)
        episode_rewards.append(episode_reward)
        
        # Calculate makespan
        makespan = max(env.decoder_time_left)
        
        # Log reward and makespan to file
        with open(log_file, "a") as f:
            f.write(f"{episode + 1},{episode_reward:.4f},{makespan},{agent.epsilon:.6f}\n")
        
        # Print progress every 1000 episodes
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(episode_rewards[-1000:])
            print(f"Episode {episode + 1}/{n_episodes} | Avg Reward: {avg_reward:.3f} | Makespan: {makespan} | Epsilon: {agent.epsilon:.3f}")
            
            # Save model every 1000 episodes
            with open(model_file, "wb") as f:
                pickle.dump(agent, f)
            print(f"Model saved to {model_file}")

    print("Training complete!")
    print(f"Final model saved to {model_file}")
    print(f"Logs saved to {log_file}")
