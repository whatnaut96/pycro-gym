import numpy as np
import torch
import torch.nn as nn
from pycro_rts import microrts_ai
from pycro_rts.vec_env import MicroRTSGridModeVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnvWrapper
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnvWrapper
import gymnasium as gym




class PycroRTSExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(29, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*8*8, features_dim),
            nn.ReLU(),
        )
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.float().permute(0, 3, 1, 2)
        return self.cnn(x)

class PycroRTSVecEnvWrapper(VecEnvWrapper):
    def reset(self):
        return self.venv.reset()
    def step_async(self, actions):
        self.venv.step_async(actions)
    def step_wait(self):
        return self.venv.step_wait()
    def action_masks(self) -> np.ndarray:
        return self.venv.get_action_mask().reshape(self.num_envs, -1)
    def has_attr(self, attr_name: str) -> bool:
        return hasattr(self, attr_name) or hasattr(self.venv, attr_name)
    def get_attr(self, attr_name: str, indices=None):
        return [getattr(self.venv, attr_name)]* self.num_envs
    def set_attr(self, attr_name: str, value, indicies:None):
        setattr(self.venv, attr_name, value)
    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        method = getattr(self, method_name, None) or getattr(self.venv, method_name)
        return [method(*method_args, **method_kwargs)]




def main():
    raw_env = MicroRTSGridModeVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=1,
        max_steps=2000,
        render_theme=2,
        ai2s=[microrts_ai.randomBiasedAI],
        map_paths=["8x8/basesWorkers8x8.xml"],
        reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
    )
    env = PycroRTSVecEnvWrapper(raw_env)

    policy_kwargs = dict(
        features_extractor_class=PycroRTSExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path="./checkpoints/",
        name_prefix="ppo_microrts",
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=64,
        verbose=1,
        tensorboard_log="./tb_logs/",
    )

    model.learn(
        total_timesteps=1_000_000,
        callback=checkpoint_callback,
    )
    model.save("checkpoints/ppo_microrts_final")
    env.close()
if __name__ == "__main__":
    main()
