"""
Reinforcement Learning Implementation

This module provides implementations of reinforcement learning algorithms including:
1. Q-Learning: A value-based method that learns the value of actions in states
2. Deep Q-Network (DQN): A deep learning approach to reinforcement learning that uses
   neural networks to approximate the Q-function

These models can be used for sequential decision-making tasks, game playing,
robotics control, and other reinforcement learning applications.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import os
import gym
from collections import deque

# For DQN
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.model_template import MLModel


class QLearningModel(MLModel):
    """
    Q-Learning reinforcement learning algorithm implementation.
    
    Q-Learning is a model-free, value-based reinforcement learning algorithm that learns
    the value of actions in states by exploring the environment and updating a Q-table
    based on the rewards received.
    
    Use case: Simple environments with discrete state and action spaces, like grid worlds,
    small games, and simple control problems.
    """
    
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.95, 
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01,
                 **kwargs):
        """
        Initialize the Q-Learning model.
        
        Args:
            n_states: Number of possible states or state space size
            n_actions: Number of possible actions
            learning_rate: Alpha - learning rate for Q-value updates (0-1)
            discount_factor: Gamma - discount factor for future rewards (0-1) 
            exploration_rate: Epsilon - initial exploration rate for epsilon-greedy policy
            exploration_decay: Rate at which exploration rate decays after each episode
            min_exploration_rate: Minimum exploration rate
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions))
        
        self.model_type = "Q-Learning"
        self.is_trained = False
        self.training_history = []
        self.model_params = {
            'n_states': n_states,
            'n_actions': n_actions,
            'learning_rate': learning_rate,
            'discount_factor': discount_factor,
            'exploration_rate': exploration_rate,
            'exploration_decay': exploration_decay,
            'min_exploration_rate': min_exploration_rate
        }
    
    def get_action(self, state):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        # Exploration: choose a random action
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.n_actions)
        
        # Exploitation: choose the best action from Q-table
        return np.argmax(self.q_table[state])
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value for a state-action pair using the Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        # Q-learning update formula: Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error
    
    def train(self, env, n_episodes=1000, max_steps=100, render=False, **kwargs):
        """
        Train the Q-learning agent in the provided environment.
        
        Args:
            env: Gym-like environment with discrete state and action spaces
            n_episodes: Number of episodes to train for
            max_steps: Maximum steps per episode
            render: Whether to render the environment (for visualization)
            **kwargs: Additional training parameters
            
        Returns:
            self: The trained model instance
        """
        episode_rewards = []
        
        for episode in range(n_episodes):
            state = env.reset()
            if isinstance(state, tuple):  # Handle gym env reset returning (state, info)
                state = state[0]
                
            state = self._preprocess_state(state)
            total_reward = 0
            done = False
            step = 0
            
            while not done and step < max_steps:
                if render:
                    env.render()
                
                # Select and perform action
                action = self.get_action(state)
                next_state, reward, done, *_ = env.step(action)  # Handle both old and new gym API
                next_state = self._preprocess_state(next_state)
                
                # Update Q-value
                self.update_q_value(state, action, reward, next_state)
                
                state = next_state
                total_reward += reward
                step += 1
            
            # Decay exploration rate
            self.exploration_rate = max(self.min_exploration_rate, 
                                       self.exploration_rate * self.exploration_decay)
            
            episode_rewards.append(total_reward)
            
            # Print progress every 100 episodes
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode+1}/{n_episodes}, Avg Reward: {avg_reward:.2f}, Exploration Rate: {self.exploration_rate:.4f}")
        
        self.training_history = episode_rewards
        self.is_trained = True
        
        return self
    
    def _preprocess_state(self, state):
        """
        Preprocess the state to ensure it can be used as an index in the Q-table.
        
        Args:
            state: The state from the environment
            
        Returns:
            Processed state that can be used as an index
        """
        # Convert numpy arrays, tuples, lists to a single integer if needed
        if isinstance(state, (np.ndarray, list, tuple)):
            # This is a simple hash function for small discrete state spaces
            # For more complex spaces, consider a more sophisticated approach
            state_hash = 0
            for i, val in enumerate(state):
                state_hash += hash(val) * (i + 1)
            return state_hash % self.n_states
        
        return state
    
    def predict(self, state):
        """
        Predict the best action for a given state based on the learned Q-values.
        
        Args:
            state: The current state
            
        Returns:
            The best action according to the Q-table
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        state = self._preprocess_state(state)
        return np.argmax(self.q_table[state])
    
    def evaluate(self, env, n_episodes=100, max_steps=100, render=False):
        """
        Evaluate the trained model on the environment.
        
        Args:
            env: Gym-like environment to evaluate on
            n_episodes: Number of episodes to evaluate
            max_steps: Maximum steps per episode
            render: Whether to render the environment during evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        episode_rewards = []
        episode_steps = []
        
        for episode in range(n_episodes):
            state = env.reset()
            if isinstance(state, tuple):  # Handle gym env reset returning (state, info)
                state = state[0]
                
            state = self._preprocess_state(state)
            total_reward = 0
            done = False
            step = 0
            
            while not done and step < max_steps:
                if render:
                    env.render()
                
                # Always choose the best action (no exploration)
                action = np.argmax(self.q_table[state])
                next_state, reward, done, *_ = env.step(action)
                next_state = self._preprocess_state(next_state)
                
                state = next_state
                total_reward += reward
                step += 1
            
            episode_rewards.append(total_reward)
            episode_steps.append(step)
        
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_steps': np.mean(episode_steps)
        }
        
        return metrics
    
    def plot_training_history(self, window_size=100, figsize=(12, 6)):
        """
        Plot the training history showing episode rewards and their running average.
        
        Args:
            window_size: Size of the window for calculating running average
            figsize: Size of the figure (width, height)
        """
        if not self.training_history:
            raise ValueError("No training history available. Train the model first.")
        
        plt.figure(figsize=figsize)
        
        # Plot episode rewards
        plt.plot(self.training_history, alpha=0.5, label='Episode Reward')
        
        # Calculate and plot the running average
        running_avg = []
        for i in range(len(self.training_history)):
            if i < window_size:
                running_avg.append(np.mean(self.training_history[:i+1]))
            else:
                running_avg.append(np.mean(self.training_history[i-window_size+1:i+1]))
        
        plt.plot(running_avg, color='red', label=f'{window_size}-Episode Running Average')
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'{self.model_type} Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_q_values(self, figsize=(12, 8)):
        """
        Plot a heatmap of the Q-values for visualization.
        
        Args:
            figsize: Size of the figure (width, height)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        plt.figure(figsize=figsize)
        
        # Create heatmap of Q-values
        plt.imshow(self.q_table, cmap='viridis')
        plt.colorbar(label='Q-Value')
        plt.xlabel('Action')
        plt.ylabel('State')
        plt.title('Q-Value Map')
        
        # If there aren't too many states/actions, show the grid
        if self.n_states <= 50 and self.n_actions <= 20:
            plt.grid(which='minor', color='w', linestyle='-', linewidth=1)
            plt.xticks(np.arange(self.n_actions))
            plt.yticks(np.arange(min(50, self.n_states)))
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the trained Q-learning model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        model_data = {
            'q_table': self.q_table,
            'model_params': self.model_params,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained Q-learning model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            self: The loaded model instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_table = model_data['q_table']
        self.model_params = model_data['model_params']
        self.training_history = model_data.get('training_history', [])
        self.is_trained = model_data['is_trained']
        
        # Update model parameters
        for key, value in self.model_params.items():
            setattr(self, key, value)
        
        print(f"Model loaded from {filepath}")
        return self


class DQNModel(MLModel):
    """
    Deep Q-Network (DQN) reinforcement learning algorithm implementation.
    
    DQN is a deep learning approach to reinforcement learning that uses neural
    networks to approximate the Q-function, allowing it to handle high-dimensional
    state spaces like images.
    
    Use case: Complex environments with continuous or high-dimensional state spaces,
    like Atari games, robotics, and complex control problems.
    """
    
    def __init__(self, state_shape, n_actions, learning_rate=0.001, discount_factor=0.99,
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01,
                 replay_buffer_size=10000, batch_size=64, target_update_freq=100,
                 network_architecture='mlp', **kwargs):
        """
        Initialize the DQN model.
        
        Args:
            state_shape: Shape of the state space (e.g., (4,) for CartPole)
            n_actions: Number of possible actions
            learning_rate: Alpha - learning rate for neural network updates
            discount_factor: Gamma - discount factor for future rewards (0-1)
            exploration_rate: Epsilon - initial exploration rate for epsilon-greedy policy
            exploration_decay: Rate at which exploration rate decays
            min_exploration_rate: Minimum exploration rate
            replay_buffer_size: Size of experience replay buffer
            batch_size: Batch size for neural network training
            target_update_freq: How often to update the target network (in steps)
            network_architecture: Type of network architecture ('mlp' or 'cnn')
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.network_architecture = network_architecture
        
        # Initialize replay buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        
        # Initialize main and target networks
        self.main_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()  # Initial target network update
        
        self.model_type = "Deep Q-Network (DQN)"
        self.is_trained = False
        self.training_history = []
        self.step_count = 0
        
        self.model_params = {
            'state_shape': state_shape,
            'n_actions': n_actions,
            'learning_rate': learning_rate,
            'discount_factor': discount_factor,
            'exploration_rate': exploration_rate,
            'exploration_decay': exploration_decay,
            'min_exploration_rate': min_exploration_rate,
            'replay_buffer_size': replay_buffer_size,
            'batch_size': batch_size,
            'target_update_freq': target_update_freq,
            'network_architecture': network_architecture
        }
    
    def _build_network(self):
        """
        Build the neural network model based on the specified architecture.
        
        Returns:
            A compiled Keras model
        """
        if self.network_architecture == 'mlp':
            # Multi-layer perceptron for simple state spaces
            model = Sequential([
                Input(shape=self.state_shape),
                Dense(64, activation='relu'),
                Dense(64, activation='relu'),
                Dense(self.n_actions, activation='linear')
            ])
        elif self.network_architecture == 'cnn':
            # Convolutional neural network for image-based state spaces
            if len(self.state_shape) != 3:
                raise ValueError("CNN architecture requires 3D state shape (height, width, channels)")
            
            model = Sequential([
                Input(shape=self.state_shape),
                Conv2D(32, (8, 8), strides=(4, 4), activation='relu'),
                Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
                Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
                Flatten(),
                Dense(512, activation='relu'),
                Dense(self.n_actions, activation='linear')
            ])
        else:
            raise ValueError(f"Unknown network architecture: {self.network_architecture}")
        
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=Huber())
        return model
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.set_weights(self.main_network.get_weights())
    
    def get_action(self, state):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        # Ensure state is in the correct shape for the network
        state = self._preprocess_state(state)
        
        # Exploration: choose a random action
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.n_actions)
        
        # Exploitation: choose the best action from Q-values predicted by the network
        q_values = self.main_network.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def _preprocess_state(self, state):
        """
        Preprocess the state to ensure it has the correct shape for the neural network.
        
        Args:
            state: The state from the environment
            
        Returns:
            Processed state with correct shape
        """
        # Convert to numpy array if it's not already
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        
        # Ensure state has the correct dimensions
        if state.shape != self.state_shape and state.shape != (1,) + self.state_shape:
            # If 1D or 2D array needs to be expanded for CNN
            if self.network_architecture == 'cnn' and len(state.shape) < 3:
                raise ValueError(f"CNN requires 3D state but got shape {state.shape}")
            
            # Add batch dimension if needed
            if state.shape == self.state_shape:
                state = np.expand_dims(state, axis=0)
        
        return state
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def replay(self):
        """
        Train the neural network on a batch of experiences from the replay buffer.
        
        Returns:
            Loss value from the training
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0
        
        # Sample a batch of experiences
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        
        states = np.array([self._preprocess_state(exp[0])[0] for exp in minibatch])
        actions = np.array([exp[1] for exp in minibatch])
        rewards = np.array([exp[2] for exp in minibatch])
        next_states = np.array([self._preprocess_state(exp[3])[0] for exp in minibatch])
        dones = np.array([exp[4] for exp in minibatch])
        
        # Compute target Q values
        target_q_values = self.target_network.predict(next_states, verbose=0)
        max_target_q_values = np.max(target_q_values, axis=1)
        
        # Q-learning update rule
        target_for_current_states = rewards + (1 - dones) * self.discount_factor * max_target_q_values
        
        # Train the network
        current_q_values = self.main_network.predict(states, verbose=0)
        for i, action in enumerate(actions):
            current_q_values[i][action] = target_for_current_states[i]
        
        history = self.main_network.fit(states, current_q_values, verbose=0, batch_size=self.batch_size)
        return history.history['loss'][0]
    
    def train(self, env, n_episodes=1000, max_steps=500, render=False, **kwargs):
        """
        Train the DQN agent in the provided environment.
        
        Args:
            env: Gym-like environment
            n_episodes: Number of episodes to train for
            max_steps: Maximum steps per episode
            render: Whether to render the environment (for visualization)
            **kwargs: Additional training parameters
            
        Returns:
            self: The trained model instance
        """
        episode_rewards = []
        
        for episode in range(n_episodes):
            state = env.reset()
            if isinstance(state, tuple):  # Handle gym env reset returning (state, info)
                state = state[0]
            
            total_reward = 0
            done = False
            step = 0
            losses = []
            
            while not done and step < max_steps:
                if render:
                    env.render()
                
                # Select and perform action
                action = self.get_action(state)
                next_state, reward, done, *_ = env.step(action)  # Handle both old and new gym API
                
                # Store experience in replay buffer
                self.remember(state, action, reward, next_state, done)
                
                # Learn from past experiences
                loss = self.replay()
                losses.append(loss)
                
                state = next_state
                total_reward += reward
                step += 1
                
                # Update step count and target network if needed
                self.step_count += 1
                if self.step_count % self.target_update_freq == 0:
                    self.update_target_network()
            
            # Decay exploration rate
            self.exploration_rate = max(self.min_exploration_rate, 
                                       self.exploration_rate * self.exploration_decay)
            
            episode_rewards.append(total_reward)
            
            # Print progress every 10 episodes
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_loss = np.mean([l for l in losses if l != 0]) if losses else 0
                print(f"Episode {episode+1}/{n_episodes}, Avg Reward: {avg_reward:.2f}, "
                      f"Avg Loss: {avg_loss:.4f}, Exploration Rate: {self.exploration_rate:.4f}")
        
        self.training_history = episode_rewards
        self.is_trained = True
        
        return self
    
    def predict(self, state):
        """
        Predict the best action for a given state based on the learned Q-function.
        
        Args:
            state: The current state
            
        Returns:
            The best action according to the Q-function
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        state = self._preprocess_state(state)
        q_values = self.main_network.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def evaluate(self, env, n_episodes=100, max_steps=500, render=False):
        """
        Evaluate the trained model on the environment.
        
        Args:
            env: Gym-like environment to evaluate on
            n_episodes: Number of episodes to evaluate
            max_steps: Maximum steps per episode
            render: Whether to render the environment during evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        episode_rewards = []
        episode_steps = []
        
        for episode in range(n_episodes):
            state = env.reset()
            if isinstance(state, tuple):  # Handle gym env reset returning (state, info)
                state = state[0]
                
            total_reward = 0
            done = False
            step = 0
            
            while not done and step < max_steps:
                if render:
                    env.render()
                
                # Always choose the best action (no exploration)
                state_processed = self._preprocess_state(state)
                q_values = self.main_network.predict(state_processed, verbose=0)
                action = np.argmax(q_values[0])
                
                next_state, reward, done, *_ = env.step(action)
                
                state = next_state
                total_reward += reward
                step += 1
            
            episode_rewards.append(total_reward)
            episode_steps.append(step)
        
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_steps': np.mean(episode_steps)
        }
        
        return metrics
    
    def plot_training_history(self, window_size=100, figsize=(12, 6)):
        """
        Plot the training history showing episode rewards and their running average.
        
        Args:
            window_size: Size of the window for calculating running average
            figsize: Size of the figure (width, height)
        """
        if not self.training_history:
            raise ValueError("No training history available. Train the model first.")
        
        plt.figure(figsize=figsize)
        
        # Plot episode rewards
        plt.plot(self.training_history, alpha=0.5, label='Episode Reward')
        
        # Calculate and plot the running average
        running_avg = []
        for i in range(len(self.training_history)):
            if i < window_size:
                running_avg.append(np.mean(self.training_history[:i+1]))
            else:
                running_avg.append(np.mean(self.training_history[i-window_size+1:i+1]))
        
        plt.plot(running_avg, color='red', label=f'{window_size}-Episode Running Average')
        
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'{self.model_type} Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def visualize_q_values(self, state, figsize=(10, 6)):
        """
        Visualize the Q-values for a specific state.
        
        Args:
            state: The state to visualize Q-values for
            figsize: Size of the figure (width, height)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        state_processed = self._preprocess_state(state)
        q_values = self.main_network.predict(state_processed, verbose=0)[0]
        
        plt.figure(figsize=figsize)
        plt.bar(range(self.n_actions), q_values)
        plt.xlabel('Action')
        plt.ylabel('Q-Value')
        plt.title('Q-Values for Current State')
        plt.xticks(range(self.n_actions))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the trained DQN model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the main network
        self.main_network.save(filepath + '_main')
        
        # Save the target network
        self.target_network.save(filepath + '_target')
        
        # Save additional model data
        model_data = {
            'model_params': self.model_params,
            'training_history': self.training_history,
            'is_trained': self.is_trained,
            'step_count': self.step_count,
            'exploration_rate': self.exploration_rate
        }
        
        with open(filepath + '_data.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained DQN model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            self: The loaded model instance
        """
        # Load the main network
        self.main_network = load_model(filepath + '_main')
        
        # Load the target network
        self.target_network = load_model(filepath + '_target')
        
        # Load additional model data
        with open(filepath + '_data.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_params = model_data['model_params']
        self.training_history = model_data['training_history']
        self.is_trained = model_data['is_trained']
        self.step_count = model_data['step_count']
        self.exploration_rate = model_data['exploration_rate']
        
        # Update model parameters
        for key, value in self.model_params.items():
            setattr(self, key, value)
        
        print(f"Model loaded from {filepath}")
        return self 