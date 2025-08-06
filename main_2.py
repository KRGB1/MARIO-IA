import os
import gym
import numpy as np
from nes_py.wrappers import JoypadSpace
import signal
import sys
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack, RecordVideo
from dqn_agent import DQNAgent
import csv

# Si estás usando la COMPLEX_MOVEMENT personalizada, asegúrate de que esté aquí
# y que la importación de gym_super_mario_bros.actions.COMPLEX_MOVEMENT esté comentada.

agent_instance = None
env_instance = None

def signal_handler(sig, frame):
    print("\n🚨 Señal de interrupción detectada (Ctrl+C). Guardando memoria del agente...")
    if agent_instance:
        agent_instance.save_memory()
        print("💾 Memoria guardada exitosamente.")
    else:
        print("No se pudo guardar la memoria: el agente no está inicializado.")
    
    if env_instance:
        print("Cerrando entorno...")
        env_instance.close()
        print("Entorno cerrado.")
    
    sys.exit(0)

def main():
    global agent_instance
    global env_instance

    signal.signal(signal.SIGINT, signal_handler)

    # ### MODIFICADO: Cambiar de nuevo a render_mode="human" para ver la ventana
    env = gym.make("SuperMarioBros-v3", render_mode="human", apply_api_compatibility=True)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    # Comentado para evitar errores
    # env = RecordVideo(env, video_folder="./videos_colab", episode_trigger=lambda e: e % 50 == 0)
    env_instance = env
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, (84, 84))
    env = FrameStack(env, 4)

    print(f"El tamaño de la lista de acciones COMPLEX_MOVEMENT es: {len(COMPLEX_MOVEMENT)}")
    print(f"La lista de acciones es: {COMPLEX_MOVEMENT}")

    agent = DQNAgent(action_size=len(COMPLEX_MOVEMENT))
    agent_instance = agent
    agent.load_memory()

    num_episodes = 100

    with open("episodios_mario.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:
            writer.writerow(["episodio", "recompensa_total", "pasos", "terminado_por_muerte", "recompensa_final_ajustada", "nivel_completado"])

        try:
            for episode in range(num_episodes):
                state, _ = env.reset()
                total_reward = 0
                done = False
                steps = 0
                terminado_por_muerte = False
                nivel_completado = False
                previous_x_pos = None
                inactivity_counter = 0

                INACTIVITY_THRESHOLD = 30
                INACTIVITY_PENALTY_PER_STEP = -0.05
                RETROCESO_PENALTY = -3.0
                DEATH_PENALTY = -100.0
                STEP_REWARD = 0.05

                while not done:
                    env.render()
                    action = agent.select_action(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    current_x_pos = info.get('x_pos', previous_x_pos if previous_x_pos is not None else 0)
                    
                    if previous_x_pos is None:
                        previous_x_pos = current_x_pos

                    if current_x_pos <= previous_x_pos:
                        inactivity_counter += 1
                        if inactivity_counter >= INACTIVITY_THRESHOLD:
                            reward += INACTIVITY_PENALTY_PER_STEP
                    else:
                        inactivity_counter = 0
                    
                    if current_x_pos < previous_x_pos:
                        reward += RETROCESO_PENALTY

                    previous_x_pos = current_x_pos

                    if terminated:
                        terminado_por_muerte = True
                        reward += DEATH_PENALTY
                    
                    # Detectar si el nivel fue completado (no por muerte)
                    if truncated and not terminado_por_muerte:
                        nivel_completado = True
                    
                    reward += STEP_REWARD

                    agent.step(state, action, reward, next_state, done)
                    state = next_state
                    total_reward += reward
                    steps += 1

                print(f"🎮 Episodio {episode+1} - Recompensa total: {total_reward}")
                print(f"Nivel completado: {nivel_completado}")
                writer.writerow([episode+1, total_reward, steps, terminado_por_muerte, total_reward, nivel_completado])
                csvfile.flush()
                os.fsync(csvfile.fileno())

                if (episode + 1) % 5 == 0:
                    agent.save_memory()
                    print(f"💾 Memoria guardada periódicamente después del episodio {episode+1}")

        finally:
            if env_instance:
                env_instance.close()
                print("Entorno cerrado al finalizar la ejecución.")
            agent.save_memory()

if __name__ == "__main__":
    main()