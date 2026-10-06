from __future__ import annotations

import json
from collections import deque
from datetime import UTC, datetime, timedelta
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlsplit

import polars as pl
from shared_python.log import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from water_content_recorder.signal_proccesing import SignalState

logger = get_logger(__name__)

DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 5290
DEFAULT_DECIMATE = 10
HISTORY_WINDOW = timedelta(hours=1)
HISTORY_MAX_POINTS = 300
PREVIEW_MAX_POINTS = 2000
# 概形プレビューで、直前のレコードからこの秒数より間隔が空いていたら測定が行われていない
# 空白期間とみなし、グラフ上では線をつなげず途切れさせる(通常の測定間隔より十分大きい値)
PREVIEW_GAP_THRESHOLD_SECONDS = 60
DEFAULT_STATIC_DIR = Path(__file__).resolve().parents[2] / "frontend" / "build"


class SignalRecorder:
    """SignalStateを一定数バッファし、まとめてparquetに保存する。"""

    def __init__(self, buffer_size: int, output_dir: str | Path = "data"):
        self.buffer_size = buffer_size
        self.output_dir = Path(output_dir)
        self._buffer: list[tuple[datetime, SignalState]] = []
        self._latest: SignalState | None = None
        self._history: deque[tuple[datetime, float]] = deque()
        self._session_start = datetime.now().astimezone()

    @property
    def latest(self) -> SignalState | None:
        """直近に追加されたSignalStateを返す。まだ何も追加されていなければNone。"""
        return self._latest

    @property
    def session_dir(self) -> Path:
        """このセッションのparquet保存先ディレクトリ。

        分単位までをフォルダ名に含めることで、同じ日に測定をやり直した場合でも
        前回のセッションと混ざらず新しいフォルダに保存される。
        """
        return self.output_dir / f"{self._session_start:%Y%m%d%H%M}"

    def add(self, state: SignalState) -> None:
        """SignalStateをバッファに追加し、規定数に達したらparquetへ保存する。"""
        self._latest = state
        now = datetime.now().astimezone()
        self._buffer.append((now, state))
        self._history.append((now, state.water_content))
        cutoff = now - HISTORY_WINDOW
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()
        if len(self._buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """バッファに溜まっているデータをparquetファイルへまとめて保存する。"""
        if not self._buffer:
            return

        df = self._to_dataframe(self._buffer)
        timestamp, _ = self._buffer[-1]
        session_dir = self.session_dir
        session_dir.mkdir(parents=True, exist_ok=True)
        path = session_dir / f"records_{timestamp:%Y%m%dT%H%M%S}.parquet"
        try:
            df.write_parquet(path)
        except OSError:
            logger.exception(f"parquetの書き込みに失敗しました: {path}")
            return  # バッファは保持し、次回のflushで再試行する
        self._buffer.clear()

    def export(self, output_root: str | Path | None = None) -> Path:
        """測定を継続したまま、現時点までの記録をpostprocessしてグラフ等を出力する。

        出力先は `<output_root>/<session_dirの名前>_output`
        (例: `data/202609080912_output`)。output_rootを省略した場合はself.output_dir。

        Returns
        -------
        - Path: 出力先ディレクトリ

        """
        self.flush()
        from postprocess import postprocess  # noqa: PLC0415 (重いためexport時のみ読み込む)

        root = Path(output_root) if output_root is not None else self.output_dir
        export_dir = root / f"{self.session_dir.name}_output"
        postprocess(self.session_dir, export_dir)
        return export_dir

    def list_data_folders(self) -> list[str]:
        """self.output_dir直下でparquetファイルを直接持つディレクトリ名一覧を新しい順に返す(postprocess対象選択用)。"""
        if not self.output_dir.exists():
            return []
        return sorted(
            (p.name for p in self.output_dir.iterdir() if p.is_dir() and next(p.glob("*.parquet"), None)),
            reverse=True,
        )

    def latest_summary(self, decimate: int = DEFAULT_DECIMATE) -> dict | None:
        """最新のSignalStateを間引いてJSONで送りやすい辞書に変換する(フロントエンド配信用)。"""
        state = self._latest
        if state is None:
            return None
        return {
            "frequencies": state.frequencies[::decimate].tolist(),
            "s11_real": state.s11.real[::decimate].tolist(),
            "s11_imag": state.s11.imag[::decimate].tolist(),
            "t_axis": state.t_axis[::decimate].tolist(),
            "step_response": state.step_response[::decimate].tolist(),
            "impulse_response": state.impulse_response[::decimate].tolist(),
            "peak_positions": state.peak_positions,
            "peak_distance": state.peak_distance,
            "water_content": state.water_content,
        }

    def history_summary(self, max_points: int = HISTORY_MAX_POINTS) -> dict:
        """直近1時間のwater_content推移を間引いてJSONで送りやすい辞書に変換する(フロントエンド配信用)。"""
        entries = list(self._history)
        if len(entries) > max_points:
            step = len(entries) // max_points
            entries = entries[::step]
        return {
            "timestamps": [timestamp.isoformat() for timestamp, _ in entries],
            "water_content": [water_content for _, water_content in entries],
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
                "peak_positions": [state.peak_positions for _, state in records],
                "peak_distance": [state.peak_distance for _, state in records],
                "water_content": [state.water_content for _, state in records],
            },
        )


def _ms_to_datetime(ms: str | float | None) -> datetime | None:
    """フロントエンドから受け取ったUNIX時刻(ms)をtz付きdatetimeに変換する。未指定(None/空文字)ならNone。"""
    if ms is None or ms == "":
        return None
    return datetime.fromtimestamp(float(ms) / 1000, tz=UTC)


def serve_latest_http(
    recorder: SignalRecorder,
    host: str = DEFAULT_HTTP_HOST,
    port: int = DEFAULT_HTTP_PORT,
    decimate: int = DEFAULT_DECIMATE,
    static_dir: str | Path = DEFAULT_STATIC_DIR,
    on_calibrate: Callable[[float], float] | None = None,
) -> ThreadingHTTPServer:
    """recorder.latestを間引いたJSONとしてHTTPで公開する(GET /latest, /history)。

    それ以外のパスはstatic_dir(フロントエンドの `pnpm build` 成果物)を配信するので、
    ブラウザで http://host:port/ を開くだけで監視画面が見られる。

    on_calibrate: POST /calibrate で呼ばれるコールバック。引数は別方式で測定した
    reference_water_content(%)、戻り値は新しいcalibration_coefficient。
    Noneの場合 /calibrate は501を返す(SignalProcessing/.envの扱いはmain.py側の責務のため)。

    バックグラウンドスレッドでサーバーを起動し、呼び出し元はブロックしない。
    終了時は返り値の server.shutdown() を呼ぶこと。
    """

    class Handler(SimpleHTTPRequestHandler):
        """GET /latest, /history, /data-folders(/preview) と POST /export, /calibrate, /postprocess は
        JSON API、それ以外は静的ビルドを返す。
        """

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            if parsed.path == "/latest":
                payload = recorder.latest_summary(decimate)
            elif parsed.path == "/history":
                payload = recorder.history_summary()
            elif parsed.path == "/data-folders":
                payload = {"folders": recorder.list_data_folders()}
            elif parsed.path == "/data-folders/preview":
                self._handle_data_folders_preview(parse_qs(parsed.query))
                return
            else:
                super().do_GET()
                return
            self._send_json(payload)

        def do_POST(self) -> None:
            if self.path == "/export":
                self._handle_export()
            elif self.path == "/calibrate":
                self._handle_calibrate()
            elif self.path == "/postprocess":
                self._handle_postprocess()
            else:
                self.send_error(404)

        def _handle_export(self) -> None:
            try:
                export_dir = recorder.export()
            except Exception:  # postprocess由来の様々な例外をJSONエラーとして返す
                logger.exception("エクスポートに失敗しました")
                self._send_json({"error": "エクスポートに失敗しました"}, status=500)
                return
            self._send_json({"output_dir": str(export_dir)})

        def _handle_data_folders_preview(self, query: dict[str, list[str]]) -> None:
            dirs = [d for d in query.get("dirs", [""])[0].split(",") if d]
            if not dirs:
                self._send_json({"error": "dirsが指定されていません"}, status=400)
                return
            parquet_files = []
            for name in dirs:
                path = recorder.output_dir / name
                if not path.exists():
                    self._send_json({"error": f"{path} が見つかりません"}, status=400)
                    return
                parquet_files.extend(sorted(path.glob("*.parquet")))
            if not parquet_files:
                self._send_json({"error": "parquetファイルが見つかりません"}, status=400)
                return
            from postprocess import _evenly_spaced_indices  # noqa: PLC0415 (重いためこの時だけ読み込む)

            lazy = pl.scan_parquet(parquet_files).select(["timestamp", "water_content"])
            # 範囲はUNIX時刻(ms)で受け取る。範囲内だけで間引くので、狭い範囲でも点が粗くならない
            try:
                start, end = (_ms_to_datetime(query.get(k, [""])[0]) for k in ("start_ms", "end_ms"))
            except ValueError:
                self._send_json({"error": "start_ms/end_msが不正です"}, status=400)
                return
            if start is not None:
                lazy = lazy.filter(pl.col("timestamp") >= start)
            if end is not None:
                lazy = lazy.filter(pl.col("timestamp") <= end)
            df = lazy.sort("timestamp").collect()
            n = df.height
            if n == 0:
                self._send_json({"times_ms": [], "water_content": []})
                return

            # 直前の行との間隔がしきい値を超える行(=測定が行われていない空白期間の直後の最初の行)を検出する
            gap_seconds = df["timestamp"].diff().dt.total_seconds().fill_null(0.0)
            gap_starts = {i for i in range(1, n) if gap_seconds[i] > PREVIEW_GAP_THRESHOLD_SECONDS}

            # 間引き後の点に加え、空白期間の前後の点は必ず残す(間引きで消えると空白の境界がぼやけるため)
            indices = set(_evenly_spaced_indices(n, PREVIEW_MAX_POINTS))
            for i in gap_starts:
                indices.update((i - 1, i))

            times_ms = df["timestamp"].dt.epoch("ms")
            timestamps: list[int] = []
            water_content: list[float | None] = []
            prev_idx = None
            for idx in sorted(indices):
                if prev_idx is not None and idx in gap_starts and idx - 1 == prev_idx:
                    # 採用した2点の間に空白期間があるため、null点を挟んでグラフ上で線をつなげないようにする
                    timestamps.append(times_ms[idx])
                    water_content.append(None)
                timestamps.append(times_ms[idx])
                water_content.append(df["water_content"][idx])
                prev_idx = idx

            self._send_json({"times_ms": timestamps, "water_content": water_content})

        def _handle_postprocess(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                dirs = body["dirs"]
                if not isinstance(dirs, list) or not dirs or not all(isinstance(d, str) for d in dirs):
                    raise ValueError  # noqa: TRY301
                start_time = _ms_to_datetime(body.get("start_ms"))
                end_time = _ms_to_datetime(body.get("end_ms"))
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                self._send_json({"error": "dirs/start_ms/end_msが不正です"}, status=400)
                return
            paths = [recorder.output_dir / d for d in dirs]
            output_dir = recorder.output_dir / f"{'_'.join(dirs)}_output"
            try:
                from postprocess import postprocess  # noqa: PLC0415 (重いためこの時だけ読み込む)

                postprocess(
                    paths,
                    output_dir,
                    start_time=start_time,
                    end_time=end_time,
                    generate_combined_parquet=body.get("generate_combined_parquet", True),
                    generate_impulse_response=body.get("generate_impulse_response", True),
                )
            except Exception:  # postprocess由来の様々な例外をJSONエラーとして返す
                logger.exception("postprocessに失敗しました")
                self._send_json({"error": "postprocessに失敗しました"}, status=500)
                return
            self._send_json({"output_dir": str(output_dir)})

        def _handle_calibrate(self) -> None:
            if on_calibrate is None:
                self._send_json({"error": "キャリブレーション機能が利用できません"}, status=501)
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                reference_water_content = float(body["reference_water_content"])
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                self._send_json({"error": "reference_water_contentが不正です"}, status=400)
                return
            try:
                new_coefficient = on_calibrate(reference_water_content)
            except ValueError as e:
                self._send_json({"error": str(e)}, status=400)
                return
            except Exception:  # calibrate/.env書き込み由来の様々な例外をJSONエラーとして返す
                logger.exception("キャリブレーションに失敗しました")
                self._send_json({"error": "キャリブレーションに失敗しました"}, status=500)
                return
            self._send_json({"calibration_coefficient": new_coefficient})

        def _send_json(self, payload: object, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass  # 標準出力を汚さないため無効化

    handler_class = partial(Handler, directory=str(static_dir))
    server = ThreadingHTTPServer((host, port), handler_class)
    Thread(target=server.serve_forever, daemon=True).start()
    return server
