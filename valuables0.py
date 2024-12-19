# 学習実験用変数

EPISODES = 3         # エピソード数
BATCH_SIZE = 256        # バッチサイズ

MEMORY_LEN = 3600       # 経験保存ステップ数
GAMMA = 0.95            # 割引率
EPSILON = 1.0           # 初期の探索率
EPSILON_MIN = 0.01      # 最小探索率
EPSILON_DECAY = 0.997   # エピソードごとの探索率の減衰率
LERNING_RATE = 0.002    # 学習率

# 保存先
DIR = "./results/"

# IPソケット通信接続先
IP = '127.0.0.1'
PORT = 1234
