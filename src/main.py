import time

from water_content_recorder.io import DEFAULT_STATIC_DIR, SignalRecorder, serve_latest_http
from water_content_recorder.signal_proccesing import SignalProcessing
from water_content_recorder.vna import open_device

START_FREQ = 50e3  # 50 kHz
STOP_FREQ = 6300e6  # 6300 MHz
POINTS = 1001
BUFFER_SIZE = 50  # この数だけ溜まったらparquetにまとめて保存する
HTTP_HOST = "127.0.0.1"
HTTP_PORT = 8000
HTTP_DECIMATE = 1  # フロントエンドへ渡す配列の間引き数
INTERVAL_SEC = 3.0  # データ取得間隔(秒)


def main():
    nv = open_device()
    nv.set_frequencies(start=START_FREQ, stop=STOP_FREQ, points=POINTS)
    nv.set_sweep(START_FREQ, STOP_FREQ)

    processor = SignalProcessing()
    recorder = SignalRecorder(buffer_size=BUFFER_SIZE)
    server = serve_latest_http(recorder, host=HTTP_HOST, port=HTTP_PORT, decimate=HTTP_DECIMATE)
    if DEFAULT_STATIC_DIR.exists():
        print(f"監視画面: http://{HTTP_HOST}:{HTTP_PORT}/ をブラウザで開いてください")
    else:
        print("警告: フロントエンドが未ビルドです。frontend で `pnpm build` を実行してください")
        print(f"APIのみ起動しました: http://{HTTP_HOST}:{HTTP_PORT}/latest")

    try:
        while True:
            loop_start = time.perf_counter()

            s11 = nv.data(0)
            state = processor.process_signal(nv.frequencies, s11)
            recorder.add(state)

            elapsed = time.perf_counter() - loop_start
            sleep_time = INTERVAL_SEC - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        recorder.flush()
        nv.close()


if __name__ == "__main__":
    main()
