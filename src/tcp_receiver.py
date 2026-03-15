import argparse
import json
import socket
from typing import Iterable


def iter_lines(conn: socket.socket, bufsize: int = 4096) -> Iterable[str]:
    buffer = ""
    while True:
        chunk = conn.recv(bufsize)
        if not chunk:
            break
        buffer += chunk.decode("utf-8", errors="replace")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if line:
                yield line


def main() -> None:
    parser = argparse.ArgumentParser(description="Penerima pesan TCP LAN (newline-delimited JSON)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host bind server")
    parser.add_argument("--port", type=int, default=5000, help="Port bind server")
    args = parser.parse_args()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)

    print(f"Menunggu koneksi di {args.host}:{args.port} ...")

    try:
        while True:
            conn, addr = server.accept()
            print(f"Tersambung dari {addr[0]}:{addr[1]}")
            try:
                for line in iter_lines(conn):
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        print(f"Pesan tidak valid: {line}")
                        continue
                    print(payload)
            finally:
                conn.close()
                print("Koneksi ditutup.")
    finally:
        server.close()


if __name__ == "__main__":
    main()
