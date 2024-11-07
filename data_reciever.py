import socket

def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def start_server():
    # サーバーの設定
    host = '127.0.0.1'
    port = 1234

    # ソケットを作成
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"サーバーが{host}:{port}で待機中...")

    while True:
        # 接続を受け入れる
        client_socket, addr = server_socket.accept()
        print(f"接続されました: {addr}")

        while True:
            # データを受信
            data = client_socket.recv(1024).decode('utf-8')
            if not data:
                break

            # データを行ごとに分割
            lines = data.split('\n')
            for line in lines:
                if line:
                    # カンマで分割
                    parts = line.split(',')
                    if len(parts) == 23:  # カンマの数が22個なので、分割後の要素数は23個
                        # 指定されたインデックスの値をfloat型に変換
                        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 19, 20]:
                            if is_numeric(parts[i]):
                                parts[i] = float(parts[i])
                            else:
                                parts[i] = 0.0
                        # 残りの値をstring型に変換（実際には既にstring型なので再代入は不要）
                        for i in range(len(parts)):
                            if i not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 19, 20]:
                                parts[i] = str(parts[i])
                        
                        # 変数に格納
                        seconds = parts[0]
                        env = parts[1:10]
                        vel = parts[12]
                        lap = parts[19]
                        section = parts[20]

                    else:
                        print(f"破棄されました (カンマの数: {len(parts) - 1})")

        client_socket.close()
        print(f"接続が終了しました: {addr}")

if __name__ == "__main__":
    start_server()
