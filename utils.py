import numpy as np
import os
import random
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
import gym
import torch.nn as nn


TensorBatch = List[torch.Tensor]


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

        self.state_mean = None
        self.state_std = None
        self.action_mean = None
        self.action_std = None

        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )

        self.state_mean = np.mean(data["observations"], axis=0)
        self.state_std = np.std(data["observations"], axis=0)

        self.action_mean = np.mean(data["actions"], axis=0)
        self.action_std = np.std(data["actions"], axis=0)

        self.next_state_mean = np.mean(data["next_observations"], axis=0)
        self.next_state_std = np.std(data["next_observations"], axis=0)

        self._states[:n_transitions] = self._to_tensor(self.normalize_state(data["observations"]))
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(self.normalize_next_state(data["next_observations"]))
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print("Load dataset to replay buffer successfully! " + f"Dataset size: {n_transitions}")

        return self.state_mean, self.state_std

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, int(self._size * 0.2), size=batch_size)
        # indices = np.random.randint(0, 50000, size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._dones[self._pointer] = self._to_tensor(done)

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)

    def normalize_state(self, state):
        return (state - self.state_mean) / (self.state_std + 0.000001)

    def unnormalize_state(self, state):
        return state * (self.state_std + 0.000001) + self.state_mean

    def normalize_next_state(self, next_state):
        return (next_state - self.next_state_mean) / (self.next_state_std + 0.000001)

    def unnormalize_next_state(self, next_state):
        return next_state * (self.next_state_std + 0.000001) + self.next_state_mean

    def normalize_action(self, action):
        return (action - self.action_mean)/(self.action_std+0.000001)

    def unnormalize_action(self, action):
        return action * (self.action_std+0.000001) + self.action_mean


def set_seed(seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False):
    """
    Set seed for reproducing results.
    :param seed:
    :param env:
    :param deterministic_torch:
    :return:
    """

    env.seed(seed)
    env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(deterministic_torch)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computer the mean and std of the input vector.
    :param states:
    :param eps:
    :return:
    """
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def wrap_env(env: gym.Env, state_mean, state_std):
    """
    Return a wrapped env with the normalized state.
    :param env:
    :param state_mean:
    :param state_std:
    :return:
    """
    def normalize_state(state):
        return (state - state_mean) / state_std

    env = gym.wrappers.TransformObservation(env, normalize_state)
    return env


def is_goal_reached(reward: float, info: Dict) -> bool:
    """
    Return the "goal_achieved" info in those goal envs.
    :param reward:
    :param info:
    :return:
    """
    if "goal_achieved" in info:
        return info["goal_achieved"]
    return reward > 0  # Assuming that reaching target is a positive reward


def return_reward_range(dataset: Dict, max_episode_steps: int) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset: Dict, env_name: str, max_episode_steps: int = 1000) -> Dict:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
        return {
            "max_ret": max_ret,
            "min_ret": min_ret,
            "max_episode_steps": max_episode_steps,
        }
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0
    return {}


def modify_reward_online(reward: float, env_name: str, **kwargs) -> float:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        reward /= kwargs["max_ret"] - kwargs["min_ret"]
        reward *= kwargs["max_episode_steps"]
    elif "antmaze" in env_name:
        reward -= 1.0
    return reward


def eval_policy(policy, eval_env, seed, eval_episodes=10):
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            # eval_env.render()
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    return avg_reward


def return_reward_range(dataset: Dict, max_episode_steps: int) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset: Dict, env_name: str, max_episode_steps: int = 1000) -> Dict:
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
        return {
            "max_ret": max_ret,
            "min_ret": min_ret,
            "max_episode_steps": max_episode_steps,
        }
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0
    return {}


def weights_init(m: nn.Module, init_w: float = 3e-3):
    if isinstance(m, nn.Linear):
        m.weight.data.uniform_(-init_w, init_w)
        m.bias.data.uniform_(-init_w, init_w)
