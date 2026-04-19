import json
import sys
import time
import socket
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import MaskablePPO

N_ACTIONS_PER_CELL = 30


def log(event: str, **kwargs):
    print(json.dumps({"ts": time.time(), "event": event, **kwargs}), flush=True)


class LoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_reward = 0.0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]
        self.current_reward += reward

        log(
            "step",
            timestep=self.num_timesteps,
            reward=float(reward),
            done=bool(done),
        )

        if done:
            self.episode_rewards.append(self.current_reward)
            log(
                "episode_end",
                timestep=self.num_timesteps,
                episode_return=self.current_reward,
                episode=len(self.episode_rewards),
            )
            self.current_reward = 0.0

        return True


class GameFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]
        h = observation_space.shape[1]
        w = observation_space.shape[2]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * h * w, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.float() / 255.0
        return self.cnn(x)


class MacroRTSEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, host: str = "127.0.0.1", port: int = 9000):
        super().__init__()
        self.host = host
        self.port = port
        self.sock = None
        self.width = None
        self.height = None
        self.cells = None
        self._connect()

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(4, self.height, self.width),
            dtype=np.uint8,
        )

        self.action_space = spaces.MultiDiscrete([N_ACTIONS_PER_CELL] * self.cells)
        self._last_mask = np.ones((self.cells, N_ACTIONS_PER_CELL), dtype=bool)

    def _connect(self) -> None:
        if self.sock is not None:
            return
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        self._fetch_config()

    def _fetch_config(self) -> None:
        self.sock.sendall(bytes([4]))
        data = self._recv_exact(2)
        self.width = data[0]
        self.height = data[1]
        self.cells = self.width * self.height
        log("config_fetched", width=self.width, height=self.height, cells=self.cells)

    def _disconnect(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def _recv_exact(self, num_bytes: int) -> bytes:
        data = bytearray()
        while len(data) < num_bytes:
            chunk = self.sock.recv(num_bytes - len(data))
            if not chunk:
                raise ConnectionError("Server closed connection")
            data.extend(chunk)
        return bytes(data)

    def _recv_obs(self):
        obs_size = self.cells * 4
        mask_size = self.cells * N_ACTIONS_PER_CELL
        total = obs_size + mask_size + 1

        data = self._recv_exact(total)

        obs = np.frombuffer(data[:obs_size], dtype=np.uint8).copy()
        obs = obs.reshape(self.height, self.width, 4)
        obs = obs.transpose(2, 0, 1)

        mask = np.frombuffer(
            data[obs_size:obs_size + mask_size],
            dtype=np.uint8,
        ).reshape(self.cells, N_ACTIONS_PER_CELL).astype(bool)

        done = bool(data[-1])
        return obs, mask, done

    def _encode_actions(self, action: np.ndarray) -> np.ndarray:
        encoded = np.zeros(self.cells, dtype=np.uint8)
        for i, a in enumerate(action):
            a = int(a)
            act = a // 5
            direction = a % 5
            encoded[i] = (act << 4) | direction
        return encoded

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.sock is None:
            self._connect()
        self.sock.sendall(bytes([0]))
        obs, mask, _done = self._recv_obs()
        self._last_mask = mask
        log("episode_reset")
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.int64)
        if action.shape != (self.cells,):
            raise ValueError(f"Expected action shape {(self.cells,)}, got {action.shape}")

        encoded = self._encode_actions(action)
        self.sock.sendall(bytes([1]))
        self.sock.sendall(encoded.tobytes())

        obs, mask, done = self._recv_obs()
        self._last_mask = mask

        reward = self._reward(obs)
        return obs, reward, done, False, {}

    def _reward(self, obs: np.ndarray) -> float:
        players = obs[1, :, :]
        friendly = np.sum(players == 1)
        enemy = np.sum(players == 2)
        log("reward", friendly=int(friendly), enemy=int(enemy), reward=float(friendly - enemy) * 0.01)
        return float(friendly - enemy) * 0.01

    def action_masks(self) -> np.ndarray:
        return self._last_mask.reshape(-1)

    def close(self):
        log("env_close")
        self._disconnect()


def main():
    log("training_start")
    env = MacroRTSEnv()

    policy_kwargs = dict(
        features_extractor_class=GameFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False,
    )

    model = MaskablePPO(
        policy="CnnPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=2.5e-4,
        n_steps=512,
        batch_size=64,
        verbose=0,
        tensorboard_log="./tb_logs/",
    )

    model.learn(total_timesteps=100_000, callback=LoggingCallback())
    log("training_complete", total_timesteps=100_000)
    model.save("macrorts_ppo")
    log("model_saved", path="macrorts_ppo")
    env.close()


if __name__ == "__main__":
    main()
