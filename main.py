import socket
import numpy as np
import vgamepad as vg
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random
from datetime import datetime

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
def get_environment_info(conn, buffer):
    float_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 19, 20}
    
    while True:
        data = conn.recv(1024).decode('utf-8')
        buffer += data
        if '\n' in buffer:  # 改行を区切り文字として使用
            lines = buffer.split('\n')
            buffer = lines[-1]  # 最後の部分を次のバッファに残す
            for line in lines[:-1]:
                if len(line) > 21 and line.strip():  # データが空でないことを確認
                    result = []
                    for i, x in enumerate(line.split(',')):
                        if i in float_indices:
                            try:
                                result.append(float(x))
                            except ValueError:
                                result.append(0.0)
                        else:
                            result.append(x)
                    return result, buffer



# データを無視する関数
def ignore_initial_data(conn, buffer, num_ignores=5):
    for _ in range(num_ignores):
        _, buffer = get_environment_info(conn, buffer)
    return buffer

# シム内環境をリセット
def reset_env():
        pyautogui.keyDown('r')
        pyautogui.keyUp('r')

# 変数の初期化
initial_inGameSec = None  # 初期のinGameSecの値を記録する変数

# メインの強化学習ループ
if __name__ == "__main__":

    # 稼働時刻を記録
    start_time = datetime.now()

    # 仮想コントローラ接続
    gamepad = vg.VX360Gamepad()

    # 学習条件
    pyautogui.FAILSAFE = False
    agent = DQNAgent()

    from valuables import EPISODES,BATCH_SIZE,DIR
    episodes = EPISODES      # 学習回数
    batch_size = BATCH_SIZE

    # ゲーム内終了条件
    under_limit = 500
    keep_time = 2

    #データ保存用
    rewards = []        # 最終スコア(報酬)
    times_finished = []  # エピソード終了時タイム
    low_speed_n = 0     # 停止によるエピソード終了回数
    time_over_n = 0     # 制限時間によるエピソード終了回数


    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('127.0.0.1', 1234))
    server_socket.listen(1)

    for e in range(episodes):
        print(f"Waiting for connection for episode {e+1}...")
        conn, addr = server_socket.accept()
        print(f"Connected by {addr}")

        buffer = ""  # バッファの初期化

        # 初期データを無視
        buffer = ignore_initial_data(conn, buffer)

        env_info, buffer = get_environment_info(conn, buffer)
        state = np.array(env_info[1:10]).reshape(1, -1)
        total_reward = 0
        time_passed = 0

        # 変数の初期化
        inGameSec = env_info[0]  # シミュレータ内経過時間
        initial_inGameSec = None  # 初期のinGameSecの値を記録する変数

        while inGameSec <= 60:  # 最大ゲーム内時間
            action = agent.act(state)
            if action == 1:
                # Aボタンを押す
                gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
                gamepad.update()
            else:
                gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
                gamepad.update()
            
            next_env_info, buffer = get_environment_info(conn, buffer)
            next_state = np.array(next_env_info[1:10]).reshape(1, -1)
            reward = 0
            done = False

            if next_env_info[20] != env_info[20]:  # セクション名が変わったら
                reward = 10 / (next_env_info[0] - time_passed)
                total_reward += reward
                print(f"{next_env_info[20]:.0f}S通過:{reward:.3f}, 時間:{(next_env_info[0] - time_passed):.3f}")
                time_passed = next_env_info[0]
            
            # next_state_info[12] がunder_limit以下になった時点の inGameSec を記録w

            if next_env_info[12] >= under_limit:
                initial_inGameSec = inGameSec

            # inGameSec が3増えたかどうかをチェック
            if initial_inGameSec is not None and inGameSec - initial_inGameSec >= keep_time:
                done = True
                print("一定時間停止していました")


            if next_env_info[19] == 1:
                done = True
                print("完走しました")

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            env_info = next_env_info
            inGameSec = next_env_info[0]  # シミュレータ内経過時間を更新
            if done:
                break
        conn.close()
            
        rewards.append(total_reward)
        times_finished.append(inGameSec)
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        print(f"Episode {e+1}/{episodes} - Reward: {total_reward}")
        reset_env()

    server_socket.close()

    end_time = datetime.now()

   # 結果をグラフで表示
    fig, ax1 = plt.subplots()

    color = "tab:blue"
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.plot(rewards, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Finish Time", color=color)
    ax2.plot(times_finished, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Progress of Learning")  # 修正箇所

    plt.savefig(DIR+f"{start_time.strftime('%Y%m%d_%H%M%S')}.png")

    df = pd.DataFrame({"Total Reward": rewards, "Time Finished": times_finished})
    df.to_csv(DIR+f"{start_time.strftime('%Y%m%d_%H%M%S')}.csv", index=False)

    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"プログラムの稼働時間: {int(hours)}時間 {int(minutes)}分 {seconds:.2f}秒")

