import socket
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque
import random
from datetime import datetime
import csv
import os

# 同一ディレクトリの別ファイルからロード
import importlib
file_name = os.path.basename(__file__)
module_suffix = os.path.splitext(file_name)[0][-1]
module_name = f"valuables{module_suffix}"
val = importlib.import_module(module_name)

# フォルダの存在を確認し、存在しない場合に作成
if not os.path.exists(val.DIR):
    os.makedirs(val.DIR)

# DQNエージェントの定義
class DQNAgent:
    # 初期設定
    def __init__(self):
        self.state_size = 10
        self.action_size = 2
        self.memory = deque(maxlen=val.MEMORY_LEN)
        self.gamma = val.GAMMA
        self.epsilon = val.EPSILON
        self.epsilon_min = val.EPSILON_MIN
        self.epsilon_decay = val.EPSILON_DECAY
        self.learning_rate = val.LEARNING_RATE
        self.model = self._build_model()
        self.qv = [0,0]

    # NNモデル定義
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
        act_values = self.model.predict(state.reshape(1, -1))
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
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
    
    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model.load_model(filepath)


class EnvironmentInfo:
    def __init__(self, env_info=None):
        if env_info is None:
            env_info = [0] * 26  # デフォルト値として0のリストを使用
        self.Time = self._to_float(env_info[0])
        self.Acceleration_x = self._to_float(env_info[1])
        self.Acceleration_y = self._to_float(env_info[2])
        self.Acceleration_z = self._to_float(env_info[3])
        self.AngularVelocity_roll = self._to_float(env_info[4])
        self.AngularVelocity_pitch = self._to_float(env_info[5])
        self.AngularVelocity_yaw = self._to_float(env_info[6])
        self.MagneticForce_x = self._to_float(env_info[7])
        self.MagneticForce_y = self._to_float(env_info[8])
        self.MagneticForce_z = self._to_float(env_info[9])
        self.GroundReflection1 = self._to_float(env_info[10])
        self.GroundReflection2 = self._to_float(env_info[11])
        self.CurrentVelocity = self._to_float(env_info[12])
        self.TargetVelocity = self._to_float(env_info[13])
        self.RevolutionCounter1 = self._to_float(env_info[14])
        self.RevolutionCounter2 = self._to_float(env_info[15])
        self.BatteryVoltage = self._to_float(env_info[16])
        self.MotorCurrent = self._to_float(env_info[17])
        self.GeoMapping = str(env_info[18]) if env_info[18] else ""
        self.Lap = self._to_float(env_info[19])
        self.Section = self._to_float(env_info[20])
        self.CPUTemp = self._to_float(env_info[21])
        self.MotorTemp = self._to_float(env_info[22])
        self.WorldLocation_x = self._to_float(env_info[23])
        self.WorldLocation_y = self._to_float(env_info[24])
        self.WorldLocation_z = self._to_float(env_info[25])

    def _to_float(self, value):
        try:
            return float(value)
        except ValueError:
            return 0.0

def get_environment_info(conn, buffer):
    while True:
        data = conn.recv(1024).decode('utf-8')
        buffer += data
        if '\n' in buffer:  # 改行を区切り文字として使用
            lines = buffer.split('\n')
            buffer = lines[-1]  # 最後の部分を次のバッファに残す
            for line in lines[:-1]:
                if len(line) > 21 and line.strip():  # データが空でないことを確認
                    result = line.split(',')
                    return result, buffer



# データを無視する関数
def ignore_initial_data(conn, buffer, num_ignores=5):
    for _ in range(num_ignores):
        _, buffer = get_environment_info(conn, buffer)
    return buffer

# シム内環境をリセット
def reset_env(conn):
    conn.sendall("true".encode("utf-8"))

def change_throttle(conn, duty):
    conn.sendall(f"{duty}".encode("utf-8"))

# メインの強化学習ループ
if __name__ == "__main__":


    # 稼働時刻を記録
    start_time = datetime.now()

    # 学習条件
    agent = DQNAgent()
    episodes = val.EPISODES      # 学習回数
    batch_size = val.BATCH_SIZE

    # 前回の一時保存データを削除
    if os.path.exists(val.DIR+f"temp{module_suffix}.csv"):
        os.remove(val.DIR+f"temp{module_suffix}.csv")

    #データ保存用
    rewards = []        # 最終スコア(報酬)
    times_finished = []  # エピソード終了時タイム
    low_speed_n = 0     # 停止によるエピソード終了回数
    time_over_n = 0     # 制限時間によるエピソード終了回数
    EnvNow = EnvironmentInfo()
    EnvNext = EnvironmentInfo()

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((val.IP, val.PORT))
    server_socket.listen(1)

    for e in range(episodes):
        print(f"Waiting for connection({val.IP}:{val.PORT}) for episode {e+1}...")
        conn, addr = server_socket.accept()
        print(f"Connected by {addr}")



        buffer = ""  # バッファの初期化

        # 初期データを無視
        buffer = ignore_initial_data(conn, buffer)

        env_info, buffer = get_environment_info(conn, buffer)
        EnvNow = EnvironmentInfo(env_info)
        state = np.array([EnvNow.Time, EnvNow.Acceleration_x, EnvNow.Acceleration_y, EnvNow.Acceleration_z, 
                        EnvNow.AngularVelocity_roll, EnvNow.AngularVelocity_pitch, EnvNow.AngularVelocity_yaw, 
                        EnvNow.MagneticForce_x, EnvNow.MagneticForce_y, EnvNow.MagneticForce_z]).reshape(1, -1)

        total_reward = 0
        time_passed = 0

        # 変数の初期化
        initial_inGameSec = None  # 初期のinGameSecの値を記録する変数
        sum_speed = 0

        count = 0
        while True: # エピソード進行中の処理
            count += 1
            reward = 0  # 報酬初期化


            action = agent.act(state)   # 行動の意思決定
            if action == 1:             # 行動実行
                change_throttle(conn, 1)  
                reward += 1
            elif action == 0:           
                change_throttle(conn, 0)  


            # 環境更新
            EnvNext, buffer = get_environment_info(conn, buffer)
            EnvNext = EnvironmentInfo(EnvNext)


            # 評価関数用
            sum_speed += EnvNext.CurrentVelocity
            reward += EnvNext.CurrentVelocity #現在速度を報酬に加算

            done = 0    # 0:未完走, 1:完走, それ以外:なんらかで強制終了

            if EnvNext.Lap == val.LAP_NUM:
                done = 1
                print("完走しました")

            # 3秒以上速度がunder_limitを下回ると強制終了
            if EnvNext.CurrentVelocity >= val.UNDER_SPEED_LIMIT:    # 速度がunder_limit以下になった時点の inGameSec を記録
                initial_inGameSec = EnvNext.Time            
            if initial_inGameSec is not None and EnvNext.Time - initial_inGameSec >= val.STOP_TIME_LIMIT:    # inGameSec が3増えたかどうかをチェック
                done = 2
                reward -= val.C_W
                print("一定時間停止していました")

            # コースアウト(座標が一定より下回る)と強制終了
            if EnvNext.WorldLocation_z <= -10 or EnvNext.Lap < 0:
                done = 3
                reward -= val.C_W
                print("コースアウトしました")

            # シム内時刻が一定を超過した場合に強制終了
            if 30 <= EnvNext.Time:
                done = 4
                reward -= val.C_W
                print("時間がかかり過ぎました")


            total_reward += reward
            next_state = np.array([EnvNext.Time, EnvNext.Acceleration_x, EnvNext.Acceleration_y, EnvNext.Acceleration_z, 
                        EnvNext.AngularVelocity_roll, EnvNext.AngularVelocity_pitch, EnvNext.AngularVelocity_yaw, 
                        EnvNext.MagneticForce_x, EnvNext.MagneticForce_y, EnvNext.MagneticForce_z]).reshape(1, -1)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done != 0:
                break

        change_throttle(conn, 0)      # EP終了時にスロットルを0に

        # 評価関数
        evaluation = sum_speed / count
        # 評価関数が向上したらモデルを保存
        if evaluation > agent.qv[0]:
            agent.qv = [evaluation, e]
            agent.save(val.DIR + f"best_model{module_suffix}.h5")


        # 途中経過記録    
        rewards.append(total_reward)
        times_finished.append(EnvNext.Time)
        with open(val.DIR+f"temp{module_suffix}.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            spend_time = datetime.now() - start_time
            hours, remainder = divmod(spend_time.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            writer.writerow([total_reward, evaluation, count, f"{EnvNext.Time:.2f}", done, f"{hours:.0f}h {minutes:.0f}min {seconds:.0f}s"])

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        print(f"Episode {e+1}/{episodes} - Reward: {total_reward}")

        # 環境リセット
        reset_env(conn)


    server_socket.close()

    end_time = datetime.now()

    # 結果をグラフで表示
    # 途中経過データを確定
    data = pd.read_csv(val.DIR+f"temp{module_suffix}.csv", header=None)
    data.columns = ["Total Reward", "Evaluation", "Total Steps", "Finish Time", "Finish Condition", "Time Spent"]
    data.to_csv(val.DIR+f"{start_time.strftime('%Y%m%d_%H%M%S')}.csv", index=False)

    # グラフを描画
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Episode')
    ax1.set_ylabel(data.columns[0], color='blue')
    ax1.plot(data[data.columns[0]], label=data.columns[0], color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel(data.columns[1], color='green')
    ax2.plot(data[data.columns[1]], label=data.columns[1], color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel(data.columns[3], color='red')
    ax3.plot(data[data.columns[3]], label=data.columns[3], color='red')
    ax3.tick_params(axis='y', labelcolor='red')

    ax4 = ax1.twinx()
    ax4.spines['right'].set_position(('outward', 120))
    ax4.set_ylabel(data.columns[4], color='purple')
    ax4.plot(data[data.columns[4]], label=data.columns[4], color='purple')
    ax4.tick_params(axis='y', labelcolor='purple')
    ax4.set_ylim(0, 10)

    # グラフのタイトルと凡例を設定
    fig.suptitle("Training Progress")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # グラフを保存
    plt.savefig(val.DIR + f"{start_time.strftime('%Y%m%d_%H%M%S')}.png")

    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"プログラムの稼働時間: {int(hours)}時間 {int(minutes)}分 {seconds:.2f}秒")

