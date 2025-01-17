# 学習実験用変数

EPISODES = 500         # エピソード数
BATCH_SIZE = 256        # バッチサイズ

MEMORY_LEN = 3600       # 経験保存ステップ数
GAMMA = 0.95            # 割引率
EPSILON = 1.0           # 初期の探索率
EPSILON_MIN = 0.01      # 最小探索率
EPSILON_DECAY = 0.997   # エピソードごとの探索率の減衰率
LEARNING_RATE = 0.002    # 学習率

C_W = 500000                 # 強制終了時のペナルティ
C_R = 0                 # ゴール時の報酬

# 終了条件
UNDER_SPEED_LIMIT = 50
STOP_TIME_LIMIT = 3
LAP_NUM = 3

# 保存先
DIR = "./results/"

# IPソケット通信接続先
IP = '127.0.0.1'
PORT = 1234
