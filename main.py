# === archivo: main.py ===

import gym
import numpy as np
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, RecordVideo
from dqn_agent import DQNAgent

# === Crear entorno con grabación de video ===
# Creamos el entorno de Super Mario Bros con soporte para renderizado en RGB (necesario para grabación)
env = gym.make("SuperMarioBros-v3", render_mode="rgb_array", apply_api_compatibility=True)

# Limitamos las acciones del agente a un conjunto simple (correr, saltar, avanzar)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# 📽️ Grabar video de cada episodio en la carpeta "videos"
env = RecordVideo(env, video_folder="videos", episode_trigger=lambda e: True)

# === Preprocesamiento visual ===
# Convertimos a escala de grises para reducir información redundante
env = GrayScaleObservation(env, keep_dim=True)

# Redimensionamos los frames a 84x84 para simplificar el input a la red neuronal
env = ResizeObservation(env, (84, 84))

# Apilamos 4 frames consecutivos para que el agente pueda percibir el movimiento
env = FrameStack(env, 4)

# === Inicializar agente DQN ===
# Creamos una instancia del agente, indicándole cuántas acciones puede tomar
agent = DQNAgent(action_size=env.action_space.n)

# Número de episodios a entrenar
num_episodes = 10

# Entrenamiento por episodios
for episode in range(num_episodes):
    # Reiniciamos el entorno al inicio de cada episodio
    state, _ = env.reset()
    total_reward = 0
    done = False

    # Mientras no se termine el episodio
    while not done:
        # Seleccionar una acción según la política actual del agente
        action = agent.select_action(state)

        # Ejecutamos la acción en el entorno y obtenemos el resultado
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Guardamos la experiencia y actualizamos la red si es posible
        agent.step(state, action, reward, next_state, done)

        # Avanzamos al siguiente estado
        state = next_state

        # Acumulamos la recompensa del episodio
        total_reward += reward

    # Mostramos el resultado del episodio actual
    print(f"🎮 Episodio {episode+1} - Recompensa total: {total_reward}")

    # Guardamos un checkpoint cada 5 episodios
    if episode % 5 == 0:
        agent.save(f"checkpoint_ep{episode+1}.pth")

# Cerramos el entorno al finalizar
env.close()