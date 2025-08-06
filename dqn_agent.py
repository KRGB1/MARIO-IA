import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import pickle

print("Versión de PyTorch:", torch.__version__)
print("¿CUDA disponible?:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Versión de CUDA:", torch.version.cuda)
    print("Nombre de GPU:", torch.cuda.get_device_name(0))

# Red neuronal convolucional que estima los valores Q
class DQNNetwork(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        print(f"DQNNetwork se está inicializando con action_size: {action_size}")
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),   # entrada: 4 frames apilados (canales=4)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, x):
        return self.net(x)

# Clase que representa al agente DQN
class DQNAgent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQNNetwork(action_size).to(self.device)
        self.target_net = DQNNetwork(action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()

        self.memory = deque(maxlen=100000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.002
        self.epsilon_decay = 0.998
        self.step_counter = 0
        self.update_target_every = 1000

    def select_action(self, state):
        if not isinstance(state, np.ndarray):
            state = np.array(state)

        # Quitar canal extra de tamaño 1 si existe
        if state.ndim == 4 and state.shape[-1] == 1:
            state = np.squeeze(state, axis=-1)

        # Ahora analizar los casos posibles:
        if state.ndim == 3:
            # Puede ser (alto, ancho, canales) o (canales, alto, ancho)
            if state.shape[2] == 4:
                # (alto, ancho, canales) => permutar a (canales, alto, ancho)
                state = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
            elif state.shape[0] == 4:
                # (canales, alto, ancho) => solo añadir batch
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                raise ValueError(f"Estado 3D con canales inesperados: {state.shape}")
        elif state.ndim == 4:
            # Se asume (batch, alto, ancho, canales)
            if state.shape[-1] == 4:
                # Permutar a (batch, canales, alto, ancho)
                state = torch.tensor(state, dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)
            elif state.shape[1] == 4:
                # Permutar a (batch, canales, alto, ancho)
                state = torch.tensor(state, dtype=torch.float32).permute(0, 1, 2, 3).to(self.device)
                # Nota: Si realmente tiene canales en la segunda dimensión, ajustar según sea necesario
            else:
                raise ValueError(f"Estado 4D con canales inesperados: {state.shape}")
        else:
            raise ValueError(f"Shape inesperado del estado: {state.shape}")

        if random.random() < self.epsilon:
            action = random.randint(0, self.action_size - 1)
            print(f"--- Debugging `select_action` --- Random action chosen: {action}")
            return action

        with torch.no_grad():
            action = self.policy_net(state).argmax().item()
            print(f"--- Debugging `select_action` --- Policy action chosen: {action}")
            return action

    def step(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

        if len(self.memory) >= self.batch_size:
            self.learn()

        self.step_counter += 1
        if self.step_counter % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def learn(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)

        # Quitar canal extra de tamaño 1 si existe
        if states.ndim == 5 and states.shape[-1] == 1:
            states = np.squeeze(states, axis=-1)
        if next_states.ndim == 5 and next_states.shape[-1] == 1:
            next_states = np.squeeze(next_states, axis=-1)

        # Si el canal está en última dimensión (batch, alto, ancho, canales), permutar a (batch, canales, alto, ancho)
        if states.shape[-1] == 4:
            states = np.transpose(states, (0, 3, 1, 2))
            next_states = np.transpose(next_states, (0, 3, 1, 2))

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # --- DEBUGGING PRINTS FOR `learn` METHOD ---
        print(f"--- Debugging `learn` method ---")
        print(f"  Shape of `states`: {states.shape}, dtype: {states.dtype}")
        print(f"  Shape of `next_states`: {next_states.shape}, dtype: {next_states.dtype}")
        print(f"  Shape of `actions`: {actions.shape}, dtype: {actions.dtype}")
        print(f"-----------------------------------")

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.criterion(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename)

    def load(self, filename):
        self.policy_net.load_state_dict(torch.load(filename))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_memory(self, filename="memory.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.memory, f)

    def load_memory(self, filename="memory.pkl"):
        try:
            with open(filename, "rb") as f:
                self.memory = pickle.load(f)
        except FileNotFoundError:
            pass