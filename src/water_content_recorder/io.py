from __future__ import annotations

import json
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from water_content_recorder.signal_proccesing import SignalState

DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8000
DEFAULT_DECIMATE = 10


class SignalRecorder:
    """SignalStateを一定数バッファし、まとめてparquetに保存する。"""

    def __init__(self, buffer_size: int, output_dir: str | Path = "data"):
        self.buffer_size = buffer_size
        self.output_dir = Path(output_dir)
        self._buffer: list[tuple[datetime, SignalState]] = []
        self._latest: SignalState | None = None

    @property
    def latest(self) -> SignalState | None:
        """直近に追加されたSignalStateを返す。まだ何も追加されていなければNone。"""
        return self._latest

    def add(self, state: SignalState) -> None:
        """SignalStateをバッファに追加し、規定数に達したらparquetへ保存する。"""
        self._latest = state
        self._buffer.append((datetime.now(UTC), state))
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """バッファに溜まっているデータをparquetファイルへまとめて保存する。"""
        if not self._buffer:
            return

        df = self._to_dataframe(self._buffer)
        timestamp, _ = self._buffer[-1]
        day_dir = self.output_dir / f"{timestamp:%Y%m%d}"
        day_dir.mkdir(parents=True, exist_ok=True)
        path = day_dir / f"records_{timestamp:%Y%m%dT%H%M%S}.parquet"
        df.write_parquet(path)
        self._buffer.clear()

    def latest_summary(self, decimate: int = DEFAULT_DECIMATE) -> dict | None:
        """最新のSignalStateを間引いてJSONで送りやすい辞書に変換する(フロントエンド配信用)。"""
        state = self._latest
        if state is None:
            return None
        return {
            "frequencies": state.frequencies[::decimate].tolist(),
            "s11_real": state.s11.real[::decimate].tolist(),
            "s11_imag": state.s11.imag[::decimate].tolist(),
            "d_axis": state.d_axis[::decimate].tolist(),
            "step_response": state.step_response[::decimate].tolist(),
            "impulse_response": state.impulse_response[::decimate].tolist(),
            "peak_distance": state.peak_distance,
            "water_content": state.water_content,
        }

    @staticmethod
    def _to_dataframe(records: list[tuple[datetime, SignalState]]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "timestamp": [timestamp for timestamp, _ in records],
                "frequencies": [state.frequencies.tolist() for _, state in records],
                "s11_real": [state.s11.real.tolist() for _, state in records],
                "s11_imag": [state.s11.imag.tolist() for _, state in records],
                "t_axis": [state.t_axis.tolist() for _, state in records],
                "d_axis": [state.d_axis.tolist() for _, state in records],
                "step_response": [state.step_response.tolist() for _, state in records],
                "impulse_response": [state.impulse_response.tolist() for _, state in records],
                "peak_distance": [state.peak_distance for _, state in records],
                "water_content": [state.water_content for _, state in records],
            },
        )


def serve_latest_http(
    recorder: SignalRecorder,
    host: str = DEFAULT_HTTP_HOST,
    port: int = DEFAULT_HTTP_PORT,
    decimate: int = DEFAULT_DECIMATE,
) -> ThreadingHTTPServer:
    """recorder.latestを間引いたJSONとしてHTTPで公開する(GET /latest)。フロントエンドWebからのポーリングを想定。

    バックグラウンドスレッドでサーバーを起動し、呼び出し元はブロックしない。
    終了時は返り値の server.shutdown() を呼ぶこと。
    """

    class LatestHandler(BaseHTTPRequestHandler):
        """GET /latest に対して最新データをJSONで返すハンドラ。"""

        def do_GET(self) -> None:
            if self.path != "/latest":
                self.send_error(404)
                return
            payload = recorder.latest_summary(decimate)
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass  # 標準出力を汚さないため無効化

    server = ThreadingHTTPServer((host, port), LatestHandler)
    Thread(target=server.serve_forever, daemon=True).start()
    return server
