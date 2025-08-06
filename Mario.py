import os
import gymnasium as gym
import numpy as np
import signal
import sys
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
from stable_baselines3 import DQN
import csv
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def make_mario_env():
    """
    Crea y envuelve el entorno de Super Mario Bros con las configuraciones que ya funcionan.
    """
    # 🚨 CAMBIO APLICADO AQUÍ: Se especifica una versión funcional
    env = gym.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = GrayscaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, 4)
    return env

def main():
    env = make_mario_env()
    env = Monitor(env) # Se agrega el monitor como último wrapper
    
    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=10000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        policy_kwargs={"normalize_images": False},
        verbose=1,
        tensorboard_log="./mario_dqn_tensorboard/"
    )

    total_timesteps = 10000000
    print("Iniciando el entrenamiento. Esto puede tomar varias horas.")
    model.learn(
        total_timesteps=total_timesteps
    )

    model.save("dqn_mario_final_model")
    print("Entrenamiento finalizado. Modelo guardado.")

if __name__ == "__main__":
    main()