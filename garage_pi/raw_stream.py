"""TCP re-broadcast of the raw radar data-port bytes (for the laptop app over Wi-Fi).

Each client receives the byte stream exactly as the radar sends it (TI packets with magic words),
so the laptop parser syncs by itself.  Slow clients are dropped rather than slowing the service.
The radar configuration needed to interpret the heatmap is at http://<pi>:<web port>/api/radar_config.
"""
from __future__ import annotations

import logging
import queue
import socket
import threading

log = logging.getLogger("garage_pi.raw")


class RawStreamServer:
    def __init__(self, port: int, host: str = "0.0.0.0", max_clients: int = 4):
        self.port, self.host, self.max_clients = port, host, max_clients
        self._clients: list[tuple[socket.socket, queue.Queue]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self.port = self._sock.getsockname()[1]
        self._sock.listen(2)
        self._sock.settimeout(0.5)
        threading.Thread(target=self._accept, name="raw-stream", daemon=True).start()

    @property
    def clients(self) -> int:
        with self._lock:
            return len(self._clients)

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                conn, addr = self._sock.accept()
            except (socket.timeout, OSError):
                continue
            with self._lock:
                if len(self._clients) >= self.max_clients:
                    conn.close()
                    continue
                q: queue.Queue = queue.Queue(maxsize=400)  # ~10 s of radar data
                self._clients.append((conn, q))
            log.info("raw stream client %s connected", addr)
            threading.Thread(target=self._send, args=(conn, q, addr), daemon=True).start()

    def _send(self, conn: socket.socket, q: queue.Queue, addr) -> None:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while not self._stop.is_set():
                try:
                    data = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if data is None:
                    break
                conn.sendall(data)
        except OSError:
            pass
        finally:
            self._drop(conn)
            log.info("raw stream client %s disconnected", addr)

    def _drop(self, conn) -> None:
        with self._lock:
            self._clients = [(c, q) for c, q in self._clients if c is not conn]
        try:
            conn.close()
        except OSError:
            pass

    def broadcast(self, data: bytes) -> None:
        with self._lock:
            clients = list(self._clients)
        for conn, q in clients:
            try:
                q.put_nowait(data)
            except queue.Full:  # client too slow: disconnect it
                q.queue.clear()
                q.put_nowait(None)

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            for c, q in self._clients:
                q.put_nowait(None)
        try:
            self._sock.close()
        except OSError:
            pass
