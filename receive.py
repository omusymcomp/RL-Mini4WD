import socket
import threading
import sys

def start_server():
    # ソケットを作成
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('127.0.0.1', 1234))
    server_socket.listen(1)
    print("サーバーが127.0.0.1:1234で待機中...")

    def wait_for_connection():
        while True:
            # 接続を待機
            client_socket, addr = server_socket.accept()
            print(f"接続されました: {addr}")

            while True:
                # データを受信
                data = client_socket.recv(1024)
                if not data:
                    break
                data_list = data.decode('utf-8').split(',')
                if len(data_list) == 23:
                    print(data_list)
                else:
                    print("受信データの長さが22ではありません。")

            client_socket.close()
            print("接続が切断されました。再び接続待機中...")

    def check_for_exit():
        input("終了するにはEnterキーを押してください...\n")
        server_socket.close()
        print("サーバーを終了します。")
        sys.exit()

    # スレッドを作成して実行
    connection_thread = threading.Thread(target=wait_for_connection)
    exit_thread = threading.Thread(target=check_for_exit)
    connection_thread.start()
    exit_thread.start()

if __name__ == "__main__":
    start_server()
