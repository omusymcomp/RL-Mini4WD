import socket
import threading
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import random
from collections import deque
import matplotlib.pyplot as plt
import pyautogui

# Neural Network definition using TensorFlow/Keras
def build_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(9,)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

# DQN Agent
class DQNAgent:
    def __init__(self):
        self.model = build_model()
        self.target_model = build_model()
        self.update_target_model()
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.update_target_every = 5
        self.steps = 0

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice([0, 1])
        state = np.array(state).reshape((1, -1))
        act_values = self.model.predict(state)
        return int(act_values[0][0] > 0.5)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                next_state = np.array(next_state).reshape((1, -1))
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            state = np.array(state).reshape((1, -1))
            target_f = self.model.predict(state)
            target_f[0][0] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.update_target_model()

# Server and reinforcement learning integration
def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('127.0.0.1', 1234))
    server_socket.listen(1)
    print("サーバーが127.0.0.1:1234で待機中...")

    agent = DQNAgent()
    rewards = []
    episode_reward = 0
    last_section_time = 0
    last_section_name = None

    def wait_for_connection():
        nonlocal episode_reward, last_section_time, last_section_name

        for episode in range(500):  # Run for 500 episodes
            client_socket, addr = server_socket.accept()
            print(f"接続されました: {addr}")

            buffer = ""
            while True:
                try:
                    data = client_socket.recv(1024)
                    if not data:
                        continue

                    buffer += data.decode('utf-8')
                    if '\n' in buffer:
                        lines = buffer.split('\n')
                        for line in lines[:-1]:
                            data_list = line.strip().split(',')
                            if len(data_list) == 23:
                                for i in range(len(data_list)):
                                    if i in list(range(10)) + [12, 19, 20]:
                                        try:
                                            data_list[i] = float(data_list[i])
                                        except ValueError:
                                            data_list[i] = 0.0
                                    else:
                                        data_list[i] = str(data_list[i])

                                state = data_list[1:10]
                                action = agent.act(state)

                                # Press or release 'w' key based on the action
                                if action == 0:
                                    pyautogui.keyDown('w')
                                else:
                                    pyautogui.keyUp('w')

                                if data_list[18] != last_section_name:
                                    if last_section_name is not None:
                                        reward = 1 / (data_list[0] - last_section_time)
                                        episode_reward += reward
                                        agent.remember(state, action, reward, state, False)
                                    last_section_time = data_list[0]
                                    last_section_name = data_list[18]

                                if data_list[12] <= 500 and data_list[0] - last_section_time >= 3:
                                    done = True
                                elif data_list[19] == 3:
                                    done = True
                                else:
                                    done = False

                                if done:
                                    rewards.append(episode_reward)
                                    episode_reward = 0
                                    last_section_time = 0
                                    last_section_name = None

                                    # Press 'Enter' key to reset the game
                                    pyautogui.press('enter')
                                    print(f"エピソード {episode + 1} 終了。")
                                    break

                                agent.replay()
                            else:
                                print("受信データの長さが23ではありません。")
                        buffer = lines[-1]
                except ConnectionResetError:
                    break

            client_socket.close()
            print("接続が切断されました。再び接続待機中...")

        plot_rewards()

    def check_for_exit():
        input("終了するにはEnterキーを押してください...\n")
        server_socket.close()
        print("サーバーを終了します。")
        sys.exit()

    connection_thread = threading.Thread(target=wait_for_connection)
    exit_thread = threading.Thread(target=check_for_exit)
    connection_thread.start()
    exit_thread.start()

    def plot_rewards():
        df = pd.DataFrame(rewards, columns=['Reward'])
        df.plot()
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('DQN Training Rewards')
        plt.show()

if __name__ == "__main__":
    start_server()
