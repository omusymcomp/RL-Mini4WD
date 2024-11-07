import socket
import numpy as np
import pyautogui
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random
import time

# DQNエージェントの定義
class DQNAgent:
    def __init__(self):
        self.state_size = 9
        self.action_size = 2
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 環境から情報を取得する関数
def get_environment_info(conn):
    float_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 19, 20}
    
    while True:
        data = conn.recv(1024).decode('utf-8')
        if len(data) > 21 and data.strip():  # データが空でないことを確認
            result = []
            for i, x in enumerate(data.split(',')):
                if i in float_indices:
                    try:
                        result.append(float(x))
                    except ValueError:
                        result.append(0.0)
                else:
                    result.append(x)
            return result



# データを無視する関数
def ignore_initial_data(conn, num_ignores=5):
    for _ in range(num_ignores):
        get_environment_info(conn)

# メインの強化学習ループ
if __name__ == "__main__":
    agent = DQNAgent()
    episodes = 100      # 学習回数
    batch_size = 32
    rewards = []

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('127.0.0.1', 1234))
    server_socket.listen(1)

    for e in range(episodes):
        print(f"Waiting for connection for episode {e+1}...")
        conn, addr = server_socket.accept()
        print(f"Connected by {addr}")

        # 初期データを無視
        ignore_initial_data(conn)

        state = np.array(get_environment_info(conn)[1:10]).reshape(1, -1)
        total_reward = 0
        for time_step in range(5000):  # 最大ステップ数を設定
            action = agent.act(state)
            if action == 1:
                pyautogui.keyDown('w')
            else:
                pyautogui.keyUp('w')
            
            next_state_info = get_environment_info(conn)
            next_state = np.array(next_state_info[1:10]).reshape(1, -1)
            reward = 0
            if next_state_info[18] != state[0][8]:  # セクション名が変わったら
                reward = 1 / (next_state_info[0] - state[0][0])
            total_reward += reward
            done = False
            if next_state_info[12] <= 500 and time_step >= 3:
                done = True
            if next_state_info[19] == 3:
                done = True
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                pyautogui.keyDown('r')
                pyautogui.keyUp('r')
                conn.close()
                break
        rewards.append(total_reward)
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        print(f"Episode {e+1}/{episodes} - Reward: {total_reward}")

    server_socket.close()

    # 結果をグラフで表示
    df = pd.DataFrame(rewards, columns=['Total Reward'])
    df.plot(title='Total Rewards per Episode')