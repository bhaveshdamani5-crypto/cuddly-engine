import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

# Import to register the environment
import env.safeguard_env

def train_rl_agent():
    print("Initializing Meta OpenEnv RL Training...")
    
    # Create the environment
    environment = gym.make('SafeGuardEnv-v0')
    
    # Verify environment conforms to Gymnasium API
    check_env(environment)
    print("Environment validation: PASSED")

    # Initialize PPO Agent (using Proximal Policy Optimization)
    # MLP policy is perfect for this continuous observation space
    model = PPO("MlpPolicy", environment, verbose=1, learning_rate=0.0003)

    # Train the Agent
    print("Starting PPO Training for 50,000 timesteps...")
    model.learn(total_timesteps=50000)

    # Evaluate the trained agent
    print("Evaluating RL Agent policy...")
    mean_reward, std_reward = evaluate_policy(model, environment, n_eval_episodes=10)
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Save the agent
    os.makedirs('models', exist_ok=True)
    model.save("models/ppo_safeguard")
    print("RL Agent saved successfully to models/ppo_safeguard.zip")

if __name__ == "__main__":
    train_rl_agent()
