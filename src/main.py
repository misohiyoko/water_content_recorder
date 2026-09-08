import os
import time
from pathlib import Path

from dotenv import load_dotenv, set_key
from shared_python.log import setup_logger

from water_content_recorder.io import DEFAULT_STATIC_DIR, SignalRecorder, serve_latest_http
from water_content_recorder.signal_proccesing import SignalProcessing
from water_content_recorder.vna import CONNECTION_ERRORS, TRANSIENT_READ_ERRORS, open_device

START_FREQ = 50e3  # 50 kHz
STOP_FREQ = 6300e6  # 6300 MHz
POINTS = 1001
BUFFER_SIZE = 50  # この数だけ溜まったらparquetにまとめて保存する
HTTP_HOST = "127.0.0.1"
HTTP_PORT = 5290
HTTP_DECIMATE = 1  # フロントエンドへ渡す配列の間引き数
INTERVAL_SEC = 3.0  # データ取得間隔(秒)
PEAK_SEARCH_START_TIME = 2e-9  # ピーク検出の不感帯(この時刻未満は探索対象外、秒)
PEAK_MIN_DISTANCE = 0.7e-9  # 近接ピークを区別する最小間隔(秒)

LOG_FILE = Path("logs") / "water_content_recorder.log"
RECONNECT_INITIAL_DELAY = 1.0
RECONNECT_MAX_DELAY = 60.0
ENV_PATH = Path(".env")

load_dotenv(ENV_PATH)
CALIBRATION_COEFFICIENT = float(os.environ.get("CALIBRATION_COEFFICIENT", "1.0"))

logger = setup_logger("water_content_recorder", log_file=LOG_FILE)


def open_device_with_retry():
    """デバイスに接続できるまでバックオフしながら無期限に再試行する。"""
    delay = RECONNECT_INITIAL_DELAY
    while True:
        try:
            nv = open_device()
            nv.set_frequencies(start=START_FREQ, stop=STOP_FREQ, points=POINTS)
            nv.set_sweep(START_FREQ, STOP_FREQ)
        except (*CONNECTION_ERRORS, *TRANSIENT_READ_ERRORS) as e:
            logger.warning(f"デバイス接続に失敗: {e!r}. {delay:.0f}秒後に再試行します。")
            time.sleep(delay)
            delay = min(delay * 2, RECONNECT_MAX_DELAY)
        else:
            logger.info("デバイスに接続しました。")
            return nv


def main():
    nv = open_device_with_retry()
    processor = SignalProcessing(
        calibration_coefficient=CALIBRATION_COEFFICIENT,
        peak_search_start_time=PEAK_SEARCH_START_TIME,
        peak_min_distance=PEAK_MIN_DISTANCE,
    )
    logger.info(f"校正係数: {CALIBRATION_COEFFICIENT}")
    recorder = SignalRecorder(buffer_size=BUFFER_SIZE)

    def calibrate_from_reference(reference_water_content: float) -> float:
        """別方式で測定したwater_content(%)を基準にcalibration_coefficientを再設定し、.envにも保存する。"""
        latest = recorder.latest
        if latest is None:
            msg = "まだ測定データがありません"
            raise ValueError(msg)
        new_coefficient = processor.calibrate(latest.peak_distance, reference_water_content)
        set_key(ENV_PATH, "CALIBRATION_COEFFICIENT", str(new_coefficient))
        logger.info(f"校正係数を再設定しました: {new_coefficient} (基準water_content={reference_water_content})")
        return new_coefficient

    server = serve_latest_http(
        recorder,
        host=HTTP_HOST,
        port=HTTP_PORT,
        decimate=HTTP_DECIMATE,
        on_calibrate=calibrate_from_reference,
    )
    if DEFAULT_STATIC_DIR.exists():
        logger.info(f"監視画面: http://{HTTP_HOST}:{HTTP_PORT}/ をブラウザで開いてください")
    else:
        logger.warning("フロントエンドが未ビルドです。frontend で `pnpm build` を実行してください")
        logger.info(f"APIのみ起動しました: http://{HTTP_HOST}:{HTTP_PORT}/latest")

    try:
        while True:
            loop_start = time.perf_counter()

            try:
                s11 = nv.data(0)
                state = processor.process_signal(nv.frequencies, s11)
                recorder.add(state)
            except TRANSIENT_READ_ERRORS as e:
                logger.warning(f"読み取りエラー(スキップして継続): {e!r}")
            except CONNECTION_ERRORS:
                logger.exception("デバイス接続が失われました。再接続します。")
                nv.close()
                nv = open_device_with_retry()

            elapsed = time.perf_counter() - loop_start
            sleep_time = INTERVAL_SEC - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        logger.info("停止要求を受け取りました。終了処理を行います。")
    finally:
        server.shutdown()
        recorder.flush()
        nv.close()
        logger.info("終了しました。")


if __name__ == "__main__":
    main()
