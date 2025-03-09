"""
Reinforcement Learning Example: Q-Learning and DQN

This example demonstrates the implementation of two reinforcement learning algorithms:
1. Q-Learning - A classic value-based method for discrete state spaces
2. Deep Q-Network (DQN) - A deep learning approach for complex state spaces

We use OpenAI Gym environments to show both algorithms in action:
- FrozenLake for tabular Q-Learning (discrete state space)
- CartPole for DQN (continuous state space)

The example includes visualization of training progress, comparison between algorithms,
and a demonstration of how to use pre-trained models for decision-making tasks.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import gym
import pickle
import time
import tensorflow as tf
from gym.wrappers import RecordVideo
from IPython.display import clear_output

# Add the parent directory to the path so we can import from models
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the reinforcement learning models
from models.reinforcement.reinforcement_learning import QLearningModel, DQNModel


def create_output_dir(subdir='reinforcement_learning'):
    """Create output directory for models and visualizations."""
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output', subdir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def train_q_learning_frozen_lake(render=False, n_episodes=5000):
    """
    Train a Q-Learning agent on the FrozenLake environment.
    
    FrozenLake is a grid world where the agent must navigate from the start to the goal
    without falling into holes. It's a discrete state space with 16 states (4x4 grid)
    and 4 actions (left, down, right, up).
    """
    print("\n--- Training Q-Learning on FrozenLake-v1 ---")
    
    # Create the FrozenLake environment
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
    
    # Define parameters
    n_states = env.observation_space.n  # 16 states (4x4 grid)
    n_actions = env.action_space.n      # 4 actions (left, down, right, up)
    
    # Initialize the Q-Learning model
    q_learning = QLearningModel(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=0.1,
        discount_factor=0.99,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration_rate=0.01
    )
    
    # Train the agent
    start_time = time.time()
    q_learning.train(env, n_episodes=n_episodes, max_steps=100, render=render)
    training_time = time.time() - start_time
    
    # Evaluate the trained agent
    print("\nEvaluating trained Q-Learning agent...")
    metrics = q_learning.evaluate(env, n_episodes=100, render=False)
    
    print(f"\nTraining time: {training_time:.2f} seconds")
    print(f"Evaluation metrics: {metrics}")
    
    # Save the trained model
    output_dir = create_output_dir()
    model_path = os.path.join(output_dir, 'q_learning_frozen_lake.pkl')
    q_learning.save_model(model_path)
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    q_learning.plot_training_history(window_size=100)
    
    # Plot Q-values
    plt.figure(figsize=(10, 8))
    q_learning.plot_q_values()
    
    return q_learning


def visualize_q_learning_policy(q_learning):
    """
    Visualize the policy learned by the Q-Learning agent on the FrozenLake environment.
    
    This creates a visual representation of the actions the agent would take in each state.
    """
    # Create a 4x4 grid for the FrozenLake environment
    policy_grid = np.zeros((4, 4), dtype=object)
    
    # Define action symbols
    action_symbols = ['←', '↓', '→', '↑']
    
    # Get the best action for each state from the Q-table
    for state in range(16):
        row, col = state // 4, state % 4
        action = np.argmax(q_learning.q_table[state])
        policy_grid[row, col] = action_symbols[action]
    
    # Visualization settings
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_xticks(np.arange(-.5, 4, 1))
    ax.set_yticks(np.arange(-.5, 4, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(color='black', linestyle='-', linewidth=1)
    
    # Create the 4x4 grid
    for i in range(4):
        for j in range(4):
            state = i * 4 + j
            
            # Different colors for different tile types in FrozenLake
            if (i, j) == (0, 0):  # Start (S)
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, color='lightblue')
                ax.text(j, i, 'S', ha='center', va='center', fontsize=20, color='black')
            elif (i, j) == (3, 3):  # Goal (G)
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, color='lightgreen')
                ax.text(j, i, 'G', ha='center', va='center', fontsize=20, color='black')
            elif (i, j) in [(1, 1), (1, 3), (2, 3), (3, 0)]:  # Holes (H)
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, color='orange')
                ax.text(j, i, 'H', ha='center', va='center', fontsize=20, color='black')
            else:  # Frozen (F)
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, color='white')
                ax.text(j, i, 'F', ha='center', va='center', fontsize=20, color='black')
            
            ax.add_patch(rect)
            
            # Add policy direction
            if not ((i, j) == (3, 3) or (i, j) in [(1, 1), (1, 3), (2, 3), (3, 0)]):
                ax.text(j, i - 0.25, policy_grid[i, j], ha='center', va='center', fontsize=25, color='blue')
    
    plt.tight_layout()
    plt.title('Q-Learning Policy on FrozenLake', fontsize=15)
    
    # Save the visualization
    output_dir = create_output_dir()
    plt.savefig(os.path.join(output_dir, 'q_learning_policy.png'))
    plt.show()


def train_dqn_cartpole(render=False, n_episodes=500):
    """
    Train a DQN agent on the CartPole environment.
    
    CartPole is a classic control problem where a pole is attached to a cart.
    The goal is to balance the pole by moving the cart left or right.
    It has a continuous state space (position, velocity, angle, angular velocity)
    and 2 discrete actions (move left or right).
    """
    print("\n--- Training DQN on CartPole-v1 ---")
    
    # Create the CartPole environment
    env = gym.make('CartPole-v1')
    
    # Define parameters
    state_shape = env.observation_space.shape  # (4,) - position, velocity, angle, angular velocity
    n_actions = env.action_space.n            # 2 actions - left or right
    
    # Initialize the DQN model
    dqn = DQNModel(
        state_shape=state_shape,
        n_actions=n_actions,
        learning_rate=0.001,
        discount_factor=0.99,
        exploration_rate=1.0,
        exploration_decay=0.995,
        min_exploration_rate=0.01,
        replay_buffer_size=10000,
        batch_size=64,
        target_update_freq=100,
        network_architecture='mlp'
    )
    
    # Train the agent
    start_time = time.time()
    dqn.train(env, n_episodes=n_episodes, max_steps=500, render=render)
    training_time = time.time() - start_time
    
    # Evaluate the trained agent
    print("\nEvaluating trained DQN agent...")
    metrics = dqn.evaluate(env, n_episodes=10, render=False)
    
    print(f"\nTraining time: {training_time:.2f} seconds")
    print(f"Evaluation metrics: {metrics}")
    
    # Save the trained model
    output_dir = create_output_dir()
    model_path = os.path.join(output_dir, 'dqn_cartpole')
    dqn.save_model(model_path)
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    dqn.plot_training_history(window_size=20)
    
    return dqn


def compare_models_visualization():
    """
    Create a comparison visualization between Q-Learning and DQN performance.
    """
    # Setup plot
    plt.figure(figsize=(15, 8))
    
    # Load data from saved models or recreate if needed
    output_dir = create_output_dir()
    ql_model_path = os.path.join(output_dir, 'q_learning_frozen_lake.pkl')
    dqn_model_path = os.path.join(output_dir, 'dqn_cartpole')
    
    try:
        # Load Q-Learning data
        with open(ql_model_path, 'rb') as f:
            ql_data = pickle.load(f)
        ql_history = ql_data['training_history']
        
        # Load DQN data
        with open(dqn_model_path + '_data.pkl', 'rb') as f:
            dqn_data = pickle.load(f)
        dqn_history = dqn_data['training_history']
        
    except FileNotFoundError:
        print("Saved model data not found. Run training first.")
        return
    
    # Normalize histories to 0-1 range for comparison
    ql_normalized = np.array(ql_history) / max(1, max(ql_history))
    dqn_normalized = np.array(dqn_history) / max(1, max(dqn_history))
    
    # Plot Q-Learning history
    plt.subplot(1, 2, 1)
    plt.plot(ql_history, alpha=0.5, color='blue')
    # Add running average
    window_size = min(100, len(ql_history))
    running_avg = []
    for i in range(len(ql_history)):
        if i < window_size:
            running_avg.append(np.mean(ql_history[:i+1]))
        else:
            running_avg.append(np.mean(ql_history[i-window_size+1:i+1]))
    plt.plot(running_avg, color='darkblue', linewidth=2)
    plt.title('Q-Learning Training (FrozenLake)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    
    # Plot DQN history
    plt.subplot(1, 2, 2)
    plt.plot(dqn_history, alpha=0.5, color='green')
    # Add running average
    window_size = min(20, len(dqn_history))
    running_avg = []
    for i in range(len(dqn_history)):
        if i < window_size:
            running_avg.append(np.mean(dqn_history[:i+1]))
        else:
            running_avg.append(np.mean(dqn_history[i-window_size+1:i+1]))
    plt.plot(running_avg, color='darkgreen', linewidth=2)
    plt.title('DQN Training (CartPole)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rl_comparison.png'))
    plt.show()


def demo_trained_agent(model_type='dqn', render=True, n_episodes=3):
    """
    Demonstrate a trained reinforcement learning agent in its environment.
    
    Args:
        model_type: Either 'q_learning' or 'dqn'
        render: Whether to render the environment
        n_episodes: Number of episodes to run
    """
    output_dir = create_output_dir()
    
    if model_type == 'q_learning':
        # Load Q-Learning model and run on FrozenLake
        model_path = os.path.join(output_dir, 'q_learning_frozen_lake.pkl')
        try:
            q_learning = QLearningModel(n_states=16, n_actions=4)
            q_learning.load_model(model_path)
            
            env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode='human' if render else None)
            
            for episode in range(n_episodes):
                state = env.reset()
                if isinstance(state, tuple):  # Handle gym env reset returning (state, info)
                    state = state[0]
                    
                total_reward = 0
                done = False
                step = 0
                
                while not done and step < 100:
                    # Use the model to choose an action
                    action = q_learning.predict(state)
                    next_state, reward, done, *_ = env.step(action)
                    
                    state = next_state
                    total_reward += reward
                    step += 1
                    
                    # Clear the output for better visualization
                    if render:
                        clear_output(wait=True)
                        time.sleep(0.5)  # Slow down for better viewing
                
                print(f"Episode {episode+1}: Reward = {total_reward}, Steps = {step}")
                
        except FileNotFoundError:
            print(f"Model file {model_path} not found. Train the model first.")
            return
        
    elif model_type == 'dqn':
        # Load DQN model and run on CartPole
        model_path = os.path.join(output_dir, 'dqn_cartpole')
        try:
            # Create a model with the same parameters
            dqn = DQNModel(
                state_shape=(4,),
                n_actions=2,
                learning_rate=0.001,
                network_architecture='mlp'
            )
            dqn.load_model(model_path)
            
            # Create environment with video recording if rendering
            if render:
                env = gym.make('CartPole-v1', render_mode='human')
                # Optional: wrap with video recorder
                # video_dir = os.path.join(output_dir, 'videos')
                # env = RecordVideo(env, video_dir, episode_trigger=lambda x: True)
            else:
                env = gym.make('CartPole-v1')
            
            for episode in range(n_episodes):
                state = env.reset()
                if isinstance(state, tuple):  # Handle gym env reset returning (state, info)
                    state = state[0]
                    
                total_reward = 0
                done = False
                step = 0
                
                while not done and step < 500:
                    # Use the model to choose an action
                    action = dqn.predict(state)
                    next_state, reward, done, *_ = env.step(action)
                    
                    state = next_state
                    total_reward += reward
                    step += 1
                
                print(f"Episode {episode+1}: Reward = {total_reward}, Steps = {step}")
            
            env.close()
            
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading model: {e}. Train the model first.")
            return
    else:
        print(f"Unknown model type: {model_type}. Choose 'q_learning' or 'dqn'.")


def main():
    """Main function to run the reinforcement learning examples."""
    print("\n=== Reinforcement Learning Examples ===\n")
    print("This script demonstrates reinforcement learning algorithms in different environments.")
    
    # Part 1: Train and visualize Q-Learning on FrozenLake
    q_learning = train_q_learning_frozen_lake(render=False, n_episodes=5000)
    visualize_q_learning_policy(q_learning)
    
    # Part 2: Train and visualize DQN on CartPole
    dqn = train_dqn_cartpole(render=False, n_episodes=500)
    
    # Part 3: Compare models
    compare_models_visualization()
    
    # Part 4: Demo agents (uncomment to run interactively)
    # print("\nDemonstrating trained Q-Learning agent on FrozenLake:")
    # demo_trained_agent(model_type='q_learning', render=True, n_episodes=3)
    
    # print("\nDemonstrating trained DQN agent on CartPole:")
    # demo_trained_agent(model_type='dqn', render=True, n_episodes=3)


if __name__ == "__main__":
    main() 