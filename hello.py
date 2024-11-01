import socket

def start_server():
    # ソケットを作成
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('127.0.0.1', 1234))
    server_socket.listen(1)
    print("サーバーが127.0.0.1:1234で待機中...")

    while True:
        # 接続を待機
        client_socket, addr = server_socket.accept()
        print(f"接続されました: {addr}")

        while True:
            # データを受信
            data = client_socket.recv(1024)
            if not data:
                break
            print(f"受信: {data.decode('utf-8')}")

        client_socket.close()
        print("接続が切断されました。再び接続待機中...")

    server_socket.close()
    print("サーバーを終了します。")

if __name__ == "__main__":
    start_server()
