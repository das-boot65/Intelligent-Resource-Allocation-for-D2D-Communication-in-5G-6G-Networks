"""
D2D Multi-Agent Reinforcement Learning System
=============================================

A comprehensive multi-agent reinforcement learning system for Device-to-Device (D2D) 
communication optimization using Ray RLlib and PPO algorithm.

Features:
- Multi-agent environment with realistic channel modeling
- PPO-based learning with customizable policies
- Comprehensive evaluation metrics
- Visualization and analysis tools
- Hyperparameter tuning capabilities

Author: Enhanced version
Date: 2025
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from tqdm import tqdm
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from plotting_module import generate_all_visualizations
except ImportError:
    logger.warning("plotting_module not found. Visualization functions may not work.")
    generate_all_visualizations = None

# For hyperparameter tuning
from ray import tune

# =========================
# CONFIGURATION CLASSES
# =========================

@dataclass
class EnvironmentConfig:
    """Configuration class for D2D environment parameters."""
    num_agents: int = 5
    max_steps: int = 100
    bandwidth: float = 50e6  # 50 MHz
    noise_power: float = 1e-10
    speed: float = 0.01
    path_loss_exponent: float = 3.5
    fading_scale: float = 1.0
    shadowing_std: float = 8.0
    channel_correlation: float = 0.7

@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    num_iterations: int = 100
    patience: int = 10
    evaluation_episodes: int = 10
    checkpoint_frequency: int = 50
    eval_frequency: int = 10

# =========================
# CONSTANTS
# =========================

SEED = 42
np.random.seed(SEED)

# =========================
# UTILITY FUNCTIONS
# =========================

def multi_policy_mapping_fn(agent_id: str, *args, **kwargs) -> str:
    """Map each agent to its own unique policy."""
    return agent_id

def ensure_directory_exists(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def calculate_fairness_index(values: List[float]) -> float:
    """Calculate Jain's fairness index."""
    if not values:
        return 0.0
    values_array = np.array(values)
    return (np.sum(values_array)**2) / (len(values_array) * np.sum(values_array**2))

# =========================
# CUSTOM D2D ENVIRONMENT
# =========================

class D2DMultiAgentEnv(MultiAgentEnv):
    """
    Multi-agent environment for D2D communication optimization.
    
    Features:
    - Realistic channel modeling with path loss, shadowing, and fading
    - Dynamic mobility patterns
    - Multi-channel and multi-mode selection
    - Comprehensive reward function
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the D2D multi-agent environment."""
        super().__init__()
        
        # Parse configuration
        if config is None:
            config = {}
        
        self.env_config = EnvironmentConfig(**{k: v for k, v in config.items() 
                                             if k in EnvironmentConfig.__annotations__})
        
        # Environment state
        self.current_step = 0
        self.prev_channel_gains = None
        
        # Agent management
        self.possible_agents = [f"agent_{i}" for i in range(self.env_config.num_agents)]
        self.agents = self.possible_agents[:]
        
        # Define observation and action spaces
        self._setup_spaces()
        
        # Initialize agent states
        self._initialize_agents()
    
    def _setup_spaces(self) -> None:
        """Set up observation and action spaces for all agents."""
        # Observation: [pos_x, pos_y, power, channel_gain, interference, 
        #               distance_to_dest, channel_one_hot(3), mode_one_hot(3), congestion]
        obs_dim = 2 + 1 + 1 + 1 + 1 + 3 + 3 + 1  # Total: 13
        
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }
        
        # Action: (continuous_power, discrete_channel_mode_combo)
        power_space = spaces.Box(low=0.1, high=1.0, shape=(1,), dtype=np.float32)
        combined_space = spaces.Discrete(9)  # 3 channels × 3 modes
        
        self.action_spaces = {
            agent: spaces.Tuple((power_space, combined_space))
            for agent in self.possible_agents
        }
    
    def _initialize_agents(self) -> None:
        """Initialize agent positions, powers, and other state variables."""
        num_agents = self.env_config.num_agents
        
        # Random initial positions with minimum separation to avoid overlap
        self.devices = self._generate_separated_positions(num_agents)
        self.powers = np.random.uniform(0.1, 1.0, size=(num_agents,))
        self.destinations = np.random.rand(num_agents, 2)
        
        # Communication parameters
        self.channels = np.zeros(num_agents, dtype=int)
        self.modes = np.zeros(num_agents, dtype=int)
        self.channel_congestion = np.zeros(3)
    
    def _generate_separated_positions(self, num_agents: int, min_distance: float = 0.1) -> np.ndarray:
        """Generate agent positions with minimum separation."""
        positions = []
        max_attempts = 100
        
        for i in range(num_agents):
            attempts = 0
            while attempts < max_attempts:
                pos = np.random.rand(2)
                if not positions:
                    positions.append(pos)
                    break
                
                distances = [np.linalg.norm(pos - existing) for existing in positions]
                if min(distances) >= min_distance:
                    positions.append(pos)
                    break
                attempts += 1
            
            if attempts == max_attempts:
                # Fallback: just use random position
                positions.append(np.random.rand(2))
        
        return np.array(positions)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.agents = self.possible_agents[:]
        
        # Reset agent states
        self._initialize_agents()
        self.prev_channel_gains = None
        
        observations = self._get_observations()
        return observations, {}
    
    def step(self, action_dict: Dict[str, Any]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """Execute one environment step."""
        # Process actions
        self._process_actions(action_dict)
        
        # Update environment state
        self._update_positions()
        
        # Calculate communication metrics
        channel_gains = self._calculate_channel_gains()
        interference = self._calculate_interference(channel_gains)
        throughput, sinr = self._calculate_throughput(channel_gains, interference)
        
        # Generate observations, rewards, and info
        observations = self._get_observations()
        rewards = self._calculate_rewards(throughput, sinr, interference)
        dones, truncated = self._check_episode_end()
        infos = self._generate_info_dict(throughput, sinr, interference)
        
        self.current_step += 1
        
        # Remove agents if episode is done
        if dones["__all__"]:
            self.agents = []
        
        return observations, rewards, dones, truncated, infos
    
    def _process_actions(self, action_dict: Dict[str, Any]) -> None:
        """Process actions from all agents."""
        for agent_id, action in action_dict.items():
            if agent_id not in self.agents:
                continue
                
            agent_idx = int(agent_id.split('_')[1])
            
            # Update power level
            self.powers[agent_idx] = np.clip(action[0][0], 0.1, 1.0)
            
            # Update channel and mode
            combined = action[1]
            old_channel = self.channels[agent_idx]
            new_channel = combined // 3
            
            # Update channel congestion tracking
            if old_channel != new_channel:
                self.channel_congestion[old_channel] = max(0, self.channel_congestion[old_channel] - 1)
                self.channel_congestion[new_channel] += 1
            
            self.channels[agent_idx] = new_channel
            self.modes[agent_idx] = combined % 3
    
    def _update_positions(self) -> None:
        """Update device positions using vectorized mobility model."""
        vectors = self.destinations - self.devices
        distances = np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Check which devices have reached their destinations
        reached = distances.flatten() < self.env_config.speed
        if np.any(reached):
            self.destinations[reached] = np.random.rand(np.sum(reached), 2)
            # Recalculate vectors after updating destinations
            vectors = self.destinations - self.devices
            distances = np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Calculate movement direction
        direction = np.divide(vectors, distances, out=np.zeros_like(vectors), where=distances > 1e-9)
        
        # Update positions
        self.devices = np.clip(self.devices + self.env_config.speed * direction, 0, 1)
    
    def _calculate_channel_gains(self) -> np.ndarray:
        """Calculate channel gains with path loss, shadowing, and fading."""
        num_agents = self.env_config.num_agents
        
        # Calculate distances between all pairs of devices
        distances = np.linalg.norm(
            self.devices[:, None] - self.devices, axis=2
        ) + 1e-6  # Add small epsilon to avoid division by zero
        
        # Path loss component
        path_loss = distances ** (-self.env_config.path_loss_exponent)
        
        # Log-normal shadowing
        shadowing_db = np.random.normal(0, self.env_config.shadowing_std, 
                                      size=(num_agents, num_agents))
        shadowing = 10 ** (shadowing_db / 10)
        
        # Rayleigh fading
        fading = np.random.exponential(scale=self.env_config.fading_scale, 
                                     size=(num_agents, num_agents))
        
        # Combined channel gains
        channel_gains = path_loss * shadowing * fading
        
        # Apply temporal correlation if available
        if self.prev_channel_gains is not None:
            correlation = self.env_config.channel_correlation
            channel_gains = (correlation * self.prev_channel_gains + 
                           (1 - correlation) * channel_gains)
        
        self.prev_channel_gains = channel_gains.copy()
        return channel_gains
    
    def _calculate_interference(self, channel_gains: np.ndarray) -> np.ndarray:
        """Calculate interference for each agent."""
        num_agents = self.env_config.num_agents
        interference = np.zeros(num_agents)
        
        for i in range(num_agents):
            i_channel = self.channels[i]
            for j in range(num_agents):
                if j != i and self.channels[j] == i_channel:
                    interference[i] += self.powers[j] * channel_gains[j, i]
        
        return interference
    
    def _calculate_throughput(self, channel_gains: np.ndarray, 
                            interference: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate throughput using Shannon capacity with mode adjustments."""
        # Signal power at each receiver
        signal = self.powers * np.diag(channel_gains)
        
        # SINR calculation
        sinr = signal / (interference + self.env_config.noise_power)
        
        # Throughput calculation with mode-specific spectral efficiency
        throughput = np.zeros_like(sinr)
        
        for i in range(self.env_config.num_agents):
            mode = self.modes[i]
            if mode == 0:  # BPSK-like
                spectral_efficiency = np.log2(1 + sinr[i])
            elif mode == 1:  # QPSK-like
                spectral_efficiency = 1.5 * np.log2(1 + 0.8 * sinr[i])
            else:  # 16QAM-like
                spectral_efficiency = 2.0 * np.log2(1 + 0.6 * sinr[i])
            
            throughput[i] = self.env_config.bandwidth * spectral_efficiency / 1e6  # Mbps
        
        return throughput, sinr
    
    def _calculate_rewards(self, throughput: np.ndarray, sinr: np.ndarray, 
                         interference: np.ndarray) -> Dict[str, float]:
        """Calculate rewards for each agent."""
        rewards = {}
        max_throughput = 10.0  # Mbps for normalization
        
        for i, agent_id in enumerate(self.agents):
            agent_idx = int(agent_id.split('_')[1])
            
            # Normalize metrics
            norm_throughput = np.clip(throughput[agent_idx] / max_throughput, 0, 1)
            
            # Energy efficiency
            energy_efficiency = throughput[agent_idx] / (self.powers[agent_idx] + 1e-9)
            norm_efficiency = np.clip(energy_efficiency / 50.0, 0, 1)
            
            # Latency (inverse of throughput)
            latency = 1000.0 / (throughput[agent_idx] + 1e-9)  # ms
            norm_latency = np.clip(1.0 - (latency / 1000.0), 0, 1)
            
            # Interference penalty
            norm_interference = np.clip(interference[agent_idx] / 1000.0, 0, 1)
            
            # QoS satisfaction
            qos_threshold = 2.0  # Mbps
            qos = 1.0 if throughput[agent_idx] >= qos_threshold else 0.0
            
            # Channel congestion penalty
            channel_idx = self.channels[agent_idx]
            congestion_penalty = self.channel_congestion[channel_idx] / self.env_config.num_agents
            
            # Composite reward function
            reward = (
                4.0 * norm_throughput +
                1.5 * norm_efficiency +
                1.0 * norm_latency +
                -1.5 * norm_interference +
                2.0 * qos +
                -0.5 * congestion_penalty
            )
            
            # Mode-specific bonuses
            mode = self.modes[agent_idx]
            if mode == 0 and self.powers[agent_idx] < 0.5:  # Efficient low-power mode
                reward += 0.5
            elif mode == 2 and throughput[agent_idx] > 5.0:  # High-throughput mode
                reward += 0.5
            
            rewards[agent_id] = reward
        
        return rewards
    
    def _check_episode_end(self) -> Tuple[Dict[str, bool], Dict[str, bool]]:
        """Check if episode should end."""
        done = self.current_step >= self.env_config.max_steps
        
        dones = {agent_id: done for agent_id in self.agents}
        dones["__all__"] = done
        
        truncated = {agent_id: done for agent_id in self.agents}
        truncated["__all__"] = done
        
        return dones, truncated
    
    def _generate_info_dict(self, throughput: np.ndarray, sinr: np.ndarray, 
                          interference: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Generate information dictionary for each agent."""
        infos = {}
        
        for i, agent_id in enumerate(self.agents):
            agent_idx = int(agent_id.split('_')[1])
            
            energy_efficiency = throughput[agent_idx] / (self.powers[agent_idx] + 1e-9)
            latency = 1000.0 / (throughput[agent_idx] + 1e-9)
            qos_threshold = 2.0
            qos = 1.0 if throughput[agent_idx] >= qos_threshold else 0.0
            
            infos[agent_id] = {
                "throughput_mbps": float(throughput[agent_idx]),
                "energy_efficiency": float(energy_efficiency),
                "latency_ms": float(latency),
                "interference": float(interference[agent_idx]),
                "qos": float(qos),
                "sinr_db": float(10 * np.log10(sinr[agent_idx] + 1e-9)),
                "channel": int(self.channels[agent_idx]),
                "mode": int(self.modes[agent_idx]),
                "power": float(self.powers[agent_idx]),
                "position_x": float(self.devices[agent_idx, 0]),
                "position_y": float(self.devices[agent_idx, 1])
            }
        
        return infos
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Generate observations for all agents."""
        channel_gains = self._calculate_channel_gains()
        interference = self._calculate_interference(channel_gains)
        norm_congestion = self.channel_congestion / self.env_config.num_agents
        
        observations = {}
        
        for i in range(self.env_config.num_agents):
            agent_id = f"agent_{i}"
            if agent_id in self.agents:
                # Calculate distance to destination
                distance_to_dest = np.linalg.norm(self.devices[i] - self.destinations[i])
                norm_distance = distance_to_dest / np.sqrt(2)  # Max distance in unit square
                
                # One-hot encodings
                channel_one_hot = np.zeros(3)
                channel_one_hot[self.channels[i]] = 1.0
                
                mode_one_hot = np.zeros(3)
                mode_one_hot[self.modes[i]] = 1.0
                
                # Combine all observation components
                observations[agent_id] = np.concatenate([
                    [self.devices[i, 0], self.devices[i, 1]],  # Position
                    [self.powers[i]],  # Power
                    [np.clip(channel_gains[i, i], 0, 1)],  # Channel gain
                    [np.clip(interference[i], 0, 1)],  # Interference
                    [norm_distance],  # Distance to destination
                    channel_one_hot,  # Channel selection
                    mode_one_hot,  # Mode selection
                    [norm_congestion[self.channels[i]]]  # Channel congestion
                ], dtype=np.float32)
        
        return observations

# =========================
# BASELINE POLICY
# =========================

class RuleBasedPolicy:
    """Rule-based policy for baseline comparison."""
    
    @staticmethod
    def get_action(observation: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Generate action based on simple rules.
        
        Args:
            observation: Agent observation vector
            
        Returns:
            Tuple of (power_action, combined_channel_mode_action)
        """
        pos_x, pos_y = observation[0], observation[1]
        current_power = observation[2]
        channel_gain = observation[3]
        interference = observation[4]
        
        # Power control based on interference
        if interference > 0.5:
            power = np.array([max(0.1, current_power * 0.8)], dtype=np.float32)
        else:
            power = np.array([min(1.0, current_power * 1.2)], dtype=np.float32)
        
        # Channel selection based on position
        if interference > 0.5:
            channel = (int(pos_x * 3) + 1) % 3
        else:
            channel = int(pos_x * 3) % 3
        
        # Mode selection based on channel quality
        if channel_gain < 0.3:
            mode = 0  # Robust mode
        elif channel_gain < 0.7:
            mode = 1  # Balanced mode
        else:
            mode = 2  # High-throughput mode
        
        combined = channel * 3 + mode
        return power, combined

# =========================
# EVALUATION FUNCTIONS
# =========================

def evaluate_policy(trainer, num_episodes: int = 10, env_config_extra: Optional[Dict] = None, 
                   baseline: bool = False) -> Dict[str, float]:
    """
    Evaluate policy performance over multiple episodes.
    
    Args:
        trainer: RLlib trainer instance
        num_episodes: Number of evaluation episodes
        env_config_extra: Additional environment configuration
        baseline: Whether to use rule-based baseline
        
    Returns:
        Dictionary of averaged performance metrics
    """
    logger.info(f"Evaluating {'baseline' if baseline else 'trained'} policy over {num_episodes} episodes")
    
    # Initialize metrics tracking
    metrics = {
        "throughput": [], "energy_efficiency": [], "latency": [],
        "interference": [], "qos": [], "sinr_db": [],
        "channel_utilization": [], "fairness": [], "spectral_efficiency": []
    }
    
    # Environment configuration
    base_config = {
        "num_agents": 5,
        "max_steps": 100,
        "bandwidth": 50e6,
        "noise_power": 1e-10,
        "speed": 0.01,
        "path_loss_exponent": 3.5,
        "fading_scale": 1.0,
        "shadowing_std": 8.0,
        "channel_correlation": 0.7
    }
    
    if env_config_extra:
        base_config.update(env_config_extra)
    
    env = D2DMultiAgentEnv(base_config)
    rule_policy = RuleBasedPolicy()
    
    episode_agent_throughputs = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=episode + 42)
        done = False
        
        episode_metrics = {
            "throughput": [], "energy_efficiency": [], "latency": [],
            "interference": [], "qos": [], "sinr_db": [],
            "channel_counts": np.zeros(3)
        }
        
        agent_throughputs = {agent_id: [] for agent_id in env.possible_agents}
        
        while not done:
            actions = {}
            for agent_id, agent_obs in obs.items():
                if baseline:
                    actions[agent_id] = rule_policy.get_action(agent_obs)
                else:
                    actions[agent_id] = trainer.compute_single_action(
                        agent_obs, policy_id=agent_id
                    )
            
            obs, rewards, dones, _, infos = env.step(actions)
            done = dones["__all__"]
            
            # Collect episode metrics
            for agent_id, info in infos.items():
                if isinstance(info, dict):
                    for key in ["throughput_mbps", "energy_efficiency", "latency_ms", 
                              "interference", "qos", "sinr_db"]:
                        if key.endswith("_mbps"):
                            episode_metrics["throughput"].append(info[key])
                        elif key.endswith("_ms"):
                            episode_metrics["latency"].append(info[key])
                        else:
                            episode_metrics[key.replace("_db", "")].append(info[key])
                    
                    episode_metrics["channel_counts"][info["channel"]] += 1
                    agent_throughputs[agent_id].append(info["throughput_mbps"])
        
        # Calculate episode-level metrics
        for key in ["throughput", "energy_efficiency", "latency", "interference", "qos", "sinr"]:
            metrics[key].append(np.mean(episode_metrics[key]))
        
        # Channel utilization (measure of load balancing)
        total_channel_usage = np.sum(episode_metrics["channel_counts"])
        if total_channel_usage > 0:
            channel_util = 1 - (np.std(episode_metrics["channel_counts"]) / 
                               np.mean(episode_metrics["channel_counts"]))
        else:
            channel_util = 0
        metrics["channel_utilization"].append(channel_util)
        
        # Spectral efficiency
        spectral_efficiency = np.mean(episode_metrics["throughput"]) / (env.env_config.bandwidth / 1e6)
        metrics["spectral_efficiency"].append(spectral_efficiency)
        
        # Store agent throughputs for fairness calculation
        episode_agent_throughputs.append({
            agent: np.mean(vals) for agent, vals in agent_throughputs.items()
        })
    
    # Calculate fairness indices
    fairness_indices = []
    for ep_throughputs in episode_agent_throughputs:
        values = list(ep_throughputs.values())
        if values:
            fairness_indices.append(calculate_fairness_index(values))
    
    metrics["fairness"] = fairness_indices
    
    # Calculate averaged metrics
    avg_metrics = {key: float(np.mean(values)) for key, values in metrics.items()}
    
    # Add derived metrics
    avg_metrics["qos_rate"] = avg_metrics["qos"] * 100.0
    avg_metrics["system_capacity"] = avg_metrics["throughput"] * base_config["num_agents"]
    
    return avg_metrics

# =========================
# PPO CONFIGURATION
# =========================

def create_ppo_config(env_config: Optional[Dict] = None) -> PPOConfig:
    """
    Create optimized PPO configuration for D2D environment.
    
    Args:
        env_config: Environment configuration dictionary
        
    Returns:
        Configured PPOConfig instance
    """
    if env_config is None:
        env_config = {}
    
    base_env_config = {
        "num_agents": 5,
        "max_steps": 100,
        "bandwidth": 50e6,
        "noise_power": 1e-10,
        "speed": 0.01,
        "path_loss_exponent": 3.5,
        "fading_scale": 1.0,
        "shadowing_std": 8.0,
        "channel_correlation": 0.7
    }
    base_env_config.update(env_config)
    
    num_agents = base_env_config["num_agents"]
    
    # Define policies for each agent
    policies = {
        f"agent_{i}": PolicySpec(
            observation_space=spaces.Box(low=0, high=1, shape=(13,), dtype=np.float32),
            action_space=spaces.Tuple((
                spaces.Box(low=0.1, high=1.0, shape=(1,), dtype=np.float32),
                spaces.Discrete(9)
            )),
            config={}
        ) for i in range(num_agents)
    }
    
    return (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False, 
            enable_env_runner_and_connector_v2=False
        )
        .environment(
            env=D2DMultiAgentEnv,
            env_config=base_env_config
        )
        .framework("torch")
        .env_runners(
            num_env_runners=4,
            num_cpus_per_env_runner=1,
            rollout_fragment_length=128
        )
        .training(
            train_batch_size=1024,
            minibatch_size=128,
            num_sgd_iter=10,
            lr=5e-4,
            lr_schedule=[[0, 5e-4], [5000000, 1e-4]],
            gamma=0.99,
            lambda_=0.95,
            kl_coeff=0.2,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            entropy_coeff_schedule=[[0, 0.01], [5000000, 0.001]],
            model={
                "fcnet_hiddens": [128, 128],
                "fcnet_activation": "relu",
                "vf_share_layers": False,
            }
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=multi_policy_mapping_fn,
            policies_to_train=list(policies.keys())
        )
        .resources(num_gpus=0)
    )

# =========================
# HYPERPARAMETER TUNING
# =========================

def tune_hyperparameters(num_samples: int = 3, max_iterations: int = 50) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning using Ray Tune.
    
    Args:
        num_samples: Number of hyperparameter combinations to try
        max_iterations: Maximum training iterations per trial
        
    Returns:
        Best configuration found
    """
    logger.info("Starting hyperparameter tuning...")
    
    analysis = tune.run(
        "PPO",
        config={
            "env": D2DMultiAgentEnv,
            "env_config": {"num_agents": 5},
            "lr": tune.grid_search([1e-4, 5e-4, 1e-3]),
            "gamma": tune.grid_search([0.95, 0.99]),
            "entropy_coeff": tune.grid_search([0.01, 0.001]),
            "train_batch_size": tune.grid_search([512, 1024]),
            "sgd_minibatch_size": tune.grid_search([64, 128]),
        },
        num_samples=num_samples,
        stop={"training_iteration": max_iterations},
        verbose=1
    )
    
    logger.info("Hyperparameter tuning completed!")
    return analysis.best_config

# =========================
# VISUALIZATION FUNCTIONS
# =========================

def visualize_agent_trajectories(env_history: List[Dict], save_path: Optional[str] = None) -> None:
    """
    Visualize agent trajectories over an episode.
    
    Args:
        env_history: List of dictionaries with agent positions at each timestep
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, 5))  # Assuming 5 agents
    
    for agent_idx in range(5):  # Assuming 5 agents
        agent_id = f"agent_{agent_idx}"
        positions = []
        
        for step in env_history:
            if agent_id in step and "position" in step[agent_id]:
                positions.append(step[agent_id]["position"])
        
        if positions:
            positions = np.array(positions)
            plt.plot(positions[:, 0], positions[:, 1], '-', 
                    color=colors[agent_idx], label=f"Agent {agent_idx}", 
                    alpha=0.7, linewidth=2)
            plt.scatter(positions[0, 0], positions[0, 1], 
                       marker='>', s=150, color=colors[agent_idx], 
                       edgecolors='black', linewidth=2)  # Start
            plt.scatter(positions[-1, 0], positions[-1, 1], 
                       marker='s', s=150, color=colors[agent_idx], 
                       edgecolors='black', linewidth=2)  # End
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("Agent Trajectories Over Episode", fontsize=16, fontweight='bold')
    plt.xlabel("X Position", fontsize=14)
    plt.ylabel("Y Position", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Trajectory plot saved to {save_path}")
    
    plt.show()

def plot_training_metrics(metrics_history: List[Tuple], save_path: Optional[str] = None) -> None:
    """
    Plot training metrics over time.
    
    Args:
        metrics_history: List of (iteration, metrics_dict) tuples
        save_path: Optional path to save the plot
    """
    if not metrics_history:
        logger.warning("No metrics history provided for plotting")
        return
    
    # Extract data
    iterations = [item[0] for item in metrics_history]
    throughput = [item[1]['throughput'] for item in metrics_history]
    efficiency = [item[1]['energy_efficiency'] for item in metrics_history]
    fairness = [item[1]['fairness'] for item in metrics_history]
    qos_rate = [item[1]['qos_rate'] for item in metrics_history]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Training Progress Metrics", fontsize=16, fontweight='bold')
    
    # Throughput
    axes[0, 0].plot(iterations, throughput, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_title("Average Throughput", fontweight='bold')
    axes[0, 0].set_xlabel("Training Iteration")
    axes[0, 0].set_ylabel("Throughput (Mbps)")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Energy Efficiency
    axes[0, 1].plot(iterations, efficiency, 'g-', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_title("Energy Efficiency", fontweight='bold')
    axes[0, 1].set_xlabel("Training Iteration")
    axes[0, 1].set_ylabel("Energy Efficiency")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Fairness
    axes[1, 0].plot(iterations, fairness, 'r-', linewidth=2, marker='^', markersize=4)
    axes[1, 0].set_title("Fairness Index", fontweight='bold')
    axes[1, 0].set_xlabel("Training Iteration")
    axes[1, 0].set_ylabel("Jain's Fairness Index")
    axes[1, 0].grid(True, alpha=0.3)
    
    # QoS Rate
    axes[1, 1].plot(iterations, qos_rate, 'm-', linewidth=2, marker='d', markersize=4)
    axes[1, 1].set_title("QoS Satisfaction Rate", fontweight='bold')
    axes[1, 1].set_xlabel("Training Iteration")
    axes[1, 1].set_ylabel("QoS Rate (%)")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training metrics plot saved to {save_path}")
    
    plt.show()

# =========================
# TESTING FUNCTIONS
# =========================

def test_environment() -> None:
    """Run comprehensive tests on the D2D environment."""
    logger.info("Running environment tests...")
    
    # Test 1: Environment initialization
    try:
        env = D2DMultiAgentEnv({"num_agents": 3})
        assert len(env.possible_agents) == 3, "Incorrect number of agents"
        logger.info("✓ Environment initialization test passed")
    except Exception as e:
        logger.error(f"✗ Environment initialization test failed: {e}")
        return
    
    # Test 2: Reset functionality
    try:
        obs, info = env.reset()
        assert len(obs) == 3, "Incorrect number of observations"
        assert all(len(ob) == 13 for ob in obs.values()), "Incorrect observation dimension"
        logger.info("✓ Environment reset test passed")
    except Exception as e:
        logger.error(f"✗ Environment reset test failed: {e}")
        return
    
    # Test 3: Step functionality
    try:
        actions = {
            f"agent_{i}": (np.array([0.5]), 4)  # Mid power, mid channel/mode
            for i in range(3)
        }
        obs, rewards, dones, truncated, infos = env.step(actions)
        
        assert len(rewards) == 3, "Incorrect number of rewards"
        assert all(isinstance(r, (int, float)) for r in rewards.values()), "Invalid reward types"
        logger.info("✓ Environment step test passed")
    except Exception as e:
        logger.error(f"✗ Environment step test failed: {e}")
        return
    
    # Test 4: Channel gain calculation
    try:
        env.devices = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
        gains = env._calculate_channel_gains()
        
        assert gains.shape == (3, 3), "Incorrect channel gains shape"
        assert np.all(gains >= 0), "Channel gains must be non-negative"
        assert np.all(np.diag(gains) > 0), "Self-channel gains must be positive"
        logger.info("✓ Channel gain calculation test passed")
    except Exception as e:
        logger.error(f"✗ Channel gain calculation test failed: {e}")
        return
    
    logger.info("All environment tests passed! ✓")

def test_rule_based_policy() -> None:
    """Test the rule-based baseline policy."""
    logger.info("Testing rule-based policy...")
    
    try:
        policy = RuleBasedPolicy()
        
        # Test observation
        test_obs = np.array([0.5, 0.5, 0.7, 0.6, 0.3, 0.4, 1, 0, 0, 0, 1, 0, 0.2])
        
        power_action, combined_action = policy.get_action(test_obs)
        
        assert 0.1 <= power_action[0] <= 1.0, "Power action out of range"
        assert 0 <= combined_action < 9, "Combined action out of range"
        
        logger.info("✓ Rule-based policy test passed")
    except Exception as e:
        logger.error(f"✗ Rule-based policy test failed: {e}")

# =========================
# UTILITY FUNCTIONS
# =========================

def find_latest_progress_csv(base_dir: str = "ray_results") -> str:
    """
    Find the latest progress.csv file from Ray results.
    
    Args:
        base_dir: Base directory to search for experiments
        
    Returns:
        Path to the latest progress.csv file
        
    Raises:
        FileNotFoundError: If no progress.csv files found
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Ray results directory not found: {base_dir}")
    
    experiment_folders = [
        os.path.join(base_dir, name) 
        for name in os.listdir(base_dir) 
        if os.path.isdir(os.path.join(base_dir, name))
    ]
    
    if not experiment_folders:
        raise FileNotFoundError("No experiment folders found in ray_results")
    
    # Find the latest experiment folder
    latest_experiment = max(experiment_folders, key=os.path.getctime)
    
    # Look for progress.csv
    progress_path = os.path.join(latest_experiment, "progress.csv")
    if not os.path.exists(progress_path):
        raise FileNotFoundError(f"No progress.csv found in: {progress_path}")
    
    return progress_path

def save_results(results: Dict[str, Any], save_dir: str, experiment_id: str) -> None:
    """
    Save experiment results to files.
    
    Args:
        results: Dictionary of results to save
        save_dir: Directory to save results
        experiment_id: Unique experiment identifier
    """
    results_path = os.path.join(save_dir, experiment_id)
    ensure_directory_exists(results_path)
    
    # Save metrics as CSV
    if 'metrics_comparison' in results:
        metrics_df = results['metrics_comparison']
        csv_path = os.path.join(results_path, "metrics_comparison.csv")
        metrics_df.to_csv(csv_path, index=False)
        logger.info(f"Metrics comparison saved to {csv_path}")
    
    # Save training history
    if 'training_history' in results:
        history_df = pd.DataFrame(results['training_history'])
        history_path = os.path.join(results_path, "training_history.csv")
        history_df.to_csv(history_path, index=False)
        logger.info(f"Training history saved to {history_path}")

# =========================
# MAIN TRAINING FUNCTION
# =========================

def train_and_evaluate(config: Optional[TrainingConfig] = None, 
                      save_dir: str = "d2d_results") -> Tuple[Any, List, List]:
    """
    Main training and evaluation function.
    
    Args:
        config: Training configuration
        save_dir: Directory to save results
        
    Returns:
        Tuple of (trainer, metrics_history, training_rewards)
    """
    if config is None:
        config = TrainingConfig()
    
    # Setup experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"d2d_multiagent_{timestamp}"
    ensure_directory_exists(save_dir)
    
    results_path = os.path.join(save_dir, experiment_id)
    ensure_directory_exists(results_path)
    
    logger.info(f"Starting D2D Multi-Agent RL Training - Experiment ID: {experiment_id}")
    logger.info(f"Results will be saved to: {results_path}")
    
    # Initialize Ray (if not already initialized)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Create trainer
    ppo_config = create_ppo_config()
    trainer = ppo_config.build()
    
    # Training tracking
    metrics_history = []
    training_rewards = []
    
    # Early stopping parameters
    best_metric = -np.inf
    no_improve_count = 0
    
    logger.info("="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60)
    
    start_time = time.time()
    
    try:
        for iteration in tqdm(range(config.num_iterations), desc="Training Progress"):
            # Train one iteration
            result = trainer.train()
            iter_reward = result.get("episode_reward_mean", 0)
            training_rewards.append(iter_reward)
            
            # Log progress
            if iteration % config.eval_frequency == 0 or iteration == config.num_iterations - 1:
                tqdm.write(f"Iteration {iteration}: episode_reward_mean={iter_reward:.3f}")
                
                # Evaluate policy
                eval_metrics = evaluate_policy(trainer, num_episodes=config.evaluation_episodes)
                metrics_history.append((iteration, eval_metrics))
                
                # Log evaluation results
                tqdm.write(
                    f"[Evaluation @ {iteration}] "
                    f"Throughput: {eval_metrics['throughput']:.2f} Mbps | "
                    f"Efficiency: {eval_metrics['energy_efficiency']:.2f} | "
                    f"Latency: {eval_metrics['latency']:.1f} ms | "
                    f"SINR: {eval_metrics['sinr_db']:.1f} dB | "
                    f"QoS: {eval_metrics['qos_rate']:.1f}% | "
                    f"Fairness: {eval_metrics['fairness']:.3f}"
                )
                
                # Early stopping check
                current_metric = eval_metrics["throughput"]
                if current_metric > best_metric:
                    best_metric = current_metric
                    no_improve_count = 0
                    # Save best model
                    best_checkpoint_path = os.path.join(results_path, "best_model")
                    trainer.save(best_checkpoint_path)
                    logger.info(f"New best model saved: {current_metric:.3f} Mbps")
                else:
                    no_improve_count += 1
                
                if no_improve_count >= config.patience:
                    tqdm.write(f"Early stopping: No improvement for {config.patience} evaluations")
                    break
            
            # Periodic checkpoints
            if iteration % config.checkpoint_frequency == 0 and iteration > 0:
                checkpoint_path = os.path.join(results_path, f"checkpoint_{iteration}")
                trainer.save(checkpoint_path)
                logger.info(f"Checkpoint saved at iteration {iteration}")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.1f} seconds")
    
    # Final evaluation
    logger.info("="*60)
    logger.info("FINAL EVALUATION")
    logger.info("="*60)
    
    # Baseline evaluation
    logger.info("Evaluating rule-based baseline...")
    baseline_metrics = evaluate_policy(trainer, num_episodes=20, baseline=True)
    
    # Trained policy evaluation
    logger.info("Evaluating trained policy...")
    final_metrics = evaluate_policy(trainer, num_episodes=20)
    
    # Generalization tests
    logger.info("Testing generalization to different conditions...")
    
    mobility_metrics = evaluate_policy(
        trainer, num_episodes=20, 
        env_config_extra={"speed": 0.02}
    )
    
    channel_metrics = evaluate_policy(
        trainer, num_episodes=20,
        env_config_extra={"path_loss_exponent": 4.0, "shadowing_std": 10.0}
    )
    
    # Create results comparison
    metrics_comparison = pd.DataFrame({
        'Metric': [
            'System Capacity (Mbps)',
            'Avg Throughput (Mbps)',
            'Energy Efficiency',
            'Avg Latency (ms)',
            'Avg SINR (dB)',
            'QoS Satisfaction (%)',
            'Channel Utilization',
            'Fairness Index',
            'Spectral Efficiency (bps/Hz)'
        ],
        'Baseline': [
            baseline_metrics['system_capacity'],
            baseline_metrics['throughput'],
            baseline_metrics['energy_efficiency'],
            baseline_metrics['latency'],
            baseline_metrics['sinr_db'],
            baseline_metrics['qos_rate'],
            baseline_metrics['channel_utilization'],
            baseline_metrics['fairness'],
            baseline_metrics['spectral_efficiency']
        ],
        'Trained Policy': [
            final_metrics['system_capacity'],
            final_metrics['throughput'],
            final_metrics['energy_efficiency'],
            final_metrics['latency'],
            final_metrics['sinr_db'],
            final_metrics['qos_rate'],
            final_metrics['channel_utilization'],
            final_metrics['fairness'],
            final_metrics['spectral_efficiency']
        ],
        'Improvement (%)': [
            ((final_metrics['system_capacity'] - baseline_metrics['system_capacity']) / 
             baseline_metrics['system_capacity'] * 100),
            ((final_metrics['throughput'] - baseline_metrics['throughput']) / 
             baseline_metrics['throughput'] * 100),
            ((final_metrics['energy_efficiency'] - baseline_metrics['energy_efficiency']) / 
             baseline_metrics['energy_efficiency'] * 100),
            ((baseline_metrics['latency'] - final_metrics['latency']) / 
             baseline_metrics['latency'] * 100),  # Lower is better for latency
            ((final_metrics['sinr_db'] - baseline_metrics['sinr_db']) / 
             abs(baseline_metrics['sinr_db']) * 100),
            ((final_metrics['qos_rate'] - baseline_metrics['qos_rate']) / 
             baseline_metrics['qos_rate'] * 100),
            ((final_metrics['channel_utilization'] - baseline_metrics['channel_utilization']) / 
             baseline_metrics['channel_utilization'] * 100),
            ((final_metrics['fairness'] - baseline_metrics['fairness']) / 
             baseline_metrics['fairness'] * 100),
            ((final_metrics['spectral_efficiency'] - baseline_metrics['spectral_efficiency']) / 
             baseline_metrics['spectral_efficiency'] * 100)
        ]
    })
    
    # Display results
    logger.info("\n" + "="*80)
    logger.info("FINAL RESULTS COMPARISON")
    logger.info("="*80)
    print("\n" + metrics_comparison.to_string(index=False, float_format='%.2f'))
    
    # Save results
    results_dict = {
        'metrics_comparison': metrics_comparison,
        'training_history': [(i, m) for i, m in metrics_history],
        'baseline_metrics': baseline_metrics,
        'final_metrics': final_metrics,
        'mobility_metrics': mobility_metrics,
        'channel_metrics': channel_metrics
    }
    
    save_results(results_dict, save_dir, experiment_id)
    
    # Generate visualizations if available
    if generate_all_visualizations:
        try:
            log_path = find_latest_progress_csv()
            generate_all_visualizations(
                metrics_history, training_rewards, 
                baseline_metrics, final_metrics, log_path
            )
            logger.info("All visualizations generated successfully")
        except Exception as e:
            logger.warning(f"Could not generate visualizations: {e}")
    
    # Plot training metrics
    plot_training_metrics(metrics_history, 
                         os.path.join(results_path, "training_metrics.png"))
    
    return trainer, metrics_history, training_rewards

# =========================
# MAIN EXECUTION
# =========================

def main():
    """Main execution function."""
    logger.info("D2D Multi-Agent RL System Starting...")
    
    # Run tests
    logger.info("Running system tests...")
    test_environment()
    test_rule_based_policy()
    
    # Train and evaluate
    config = TrainingConfig(
        num_iterations=100,
        patience=15,
        evaluation_episodes=10,
        checkpoint_frequency=25,
        eval_frequency=10
    )
    
    try:
        trainer, metrics_history, training_rewards = train_and_evaluate(
            config=config, 
            save_dir="d2d_results"
        )
        
        logger.info("="*60)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        return trainer, metrics_history, training_rewards
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise
    finally:
        # Cleanup Ray
        if ray.is_initialized():
            ray.shutdown()

if __name__ == "__main__":
    main()