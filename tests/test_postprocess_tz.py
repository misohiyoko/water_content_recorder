"""postprocessの時刻範囲指定とJST出力の確認。`uv run python tests/test_postprocess_tz.py` で実行する。"""

import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from postprocess import postprocess


def test_range_and_jst_output():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # ロガーがlogを開いたままのため
        data_dir, out_dir = Path(tmp) / "data", Path(tmp) / "out"
        data_dir.mkdir()
        # 記録時と同じく、tz付きローカル時刻 → polars内部でUTCとして保存される
        t0 = datetime(2026, 9, 8, 1, 0, tzinfo=UTC)  # = JST 10:00
        times = [t0 + timedelta(minutes=i) for i in range(60)]
        pl.DataFrame(
            {"timestamp": times, "water_content": [float(i) for i in range(60)], "peak_distance": [1e-9] * 60},
        ).write_parquet(data_dir / "records.parquet")

        # フロントエンドからはUNIX時刻(=tz付き)、CLIからはtz無し(JSTとして解釈)で渡される
        for start, end in [
            (t0 + timedelta(minutes=10), t0 + timedelta(minutes=19)),
            ("2026-09-08T10:10:00", "2026-09-08T10:19:00"),
        ]:
            postprocess(
                data_dir,
                out_dir,
                start_time=start,
                end_time=end,
                generate_combined_parquet=False,
                generate_impulse_response=False,
            )
            csv = pl.read_csv(out_dir / "water_content_trend.csv")
            assert csv.height == 10, csv
            assert csv["timestamp"][0].startswith("2026-09-08T10:10:00"), csv["timestamp"][0]
            assert csv["timestamp"][0].endswith("+0900"), csv["timestamp"][0]


if __name__ == "__main__":
    test_range_and_jst_output()
    print("ok")
