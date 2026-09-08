from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from rich.console import Console

from shared_python import make_progress, setup_logger

WATER_CONTENT_COLOR = "#eb6834"
IMPULSE_RESPONSE_COLOR = "#2a78d6"
PEAK_MARKER_COLOR = "#eb6834"
# スペクトログラム(インパルス応答ヒートマップ)用の配色(plotly組み込みのInferno)
SPECTROGRAM_COLORSCALE = "Inferno"
# スライダーのラベルが重なって読めなくなるのを防ぐため、この件数を超えたら間引いてラベル表示する
MAX_LABELED_SLIDER_STEPS = 30
# 長時間測定でも特定区間へズームしやすいよう、レンジスライダーとプリセットボタンを付ける
_TIME_RANGE_AXIS_OPTIONS = {
    "rangeslider": {"visible": True},
    "rangeselector": {
        "buttons": [
            {"count": 1, "label": "1h", "step": "hour", "stepmode": "backward"},
            {"count": 6, "label": "6h", "step": "hour", "stepmode": "backward"},
            {"count": 1, "label": "1d", "step": "day", "stepmode": "backward"},
            {"count": 7, "label": "7d", "step": "day", "stepmode": "backward"},
            {"step": "all", "label": "All"},
        ],
    },
}


def postprocess(
    data_path: str | Path,
    output_path: str | Path,
    max_impulse_frames: int = 200,
    max_trend_points: int = 5000,
) -> None:
    """parquetファイル群を読み込み、結合したparquet・water_contentの時間変化(CSV・グラフ)・impulse_responseの波形をグラフ出力する。"""
    data_path = Path(data_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    console = Console()
    logger = setup_logger("postprocess", log_file=output_path / "postprocess.log", console=console)
    if not data_path.exists():
        msg = f"{data_path} が見つかりません"
        raise FileNotFoundError(msg)

    parquet_files = sorted(data_path.glob("*.parquet"))
    if not parquet_files:
        msg = f"{data_path} にparquetファイルが見つかりません"
        raise FileNotFoundError(msg)
    logger.info(f"parquetファイルを {len(parquet_files)} 件見つけました")

    with make_progress(console=console, transient=True) as progress:
        task = progress.add_task("parquetファイルを結合中", total=None)
        combined_df = pl.scan_parquet(parquet_files).sort("timestamp").collect()
        progress.update(task, completed=1, total=1)

        n_rows = combined_df.height
        logger.info(f"{n_rows} 件のレコードを読み込みました")

        trend_df = combined_df.select(["timestamp", "water_content", "peak_distance"])
        trend_plot_df = trend_df[_evenly_spaced_indices(n_rows, max_trend_points)]
        logger.info(f"グラフ用に水分量トレンドを {trend_plot_df.height} 件に間引きました")

        task = progress.add_task("インパルス応答の波形を読み込み中", total=None)
        frame_indices = _evenly_spaced_indices(n_rows, max_impulse_frames)
        impulse_df = combined_df[frame_indices].select(
            ["timestamp", "t_axis", "impulse_response", "peak_positions"],
        )
        progress.update(task, completed=1, total=1)
        logger.info(f"波形グラフ用に {impulse_df.height} 件のレコードを間引いて読み込みました")

    _write_combined_parquet(combined_df, output_path / "combined.parquet")
    logger.info("combined.parquet を出力しました")

    _write_water_content_csv(trend_df, output_path / "water_content_trend.csv")
    logger.info("water_content_trend.csv を出力しました")

    _write_water_content_png(trend_plot_df, output_path / "water_content_trend.png")
    logger.info("water_content_trend.png を出力しました")

    _write_water_content_html(trend_plot_df, output_path / "water_content_trend.html")
    logger.info("water_content_trend.html を出力しました")

    _write_impulse_response_html(impulse_df, output_path / "impulse_response.html")
    logger.info("impulse_response.html を出力しました")

    _write_spectrogram_html(impulse_df, trend_plot_df, output_path / "spectrogram.html")
    logger.info("spectrogram.html を出力しました")


def _evenly_spaced_indices(n_rows: int, max_points: int) -> list[int]:
    """0〜n_rows-1から、最大max_points件になるよう均等間引きしたインデックス列を返す。"""
    n = min(max_points, n_rows)
    return sorted(set(np.linspace(0, n_rows - 1, num=n, dtype=int).tolist()))


def _write_combined_parquet(combined_df: pl.DataFrame, path: Path) -> None:
    combined_df.write_parquet(path)


def _write_water_content_csv(trend_df: pl.DataFrame, path: Path) -> None:
    trend_df.select(["timestamp", "water_content", "peak_distance"]).write_csv(path)


def _write_water_content_png(trend_df: pl.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(trend_df["timestamp"], trend_df["water_content"], color=WATER_CONTENT_COLOR, linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Water Content (%)")
    ax.grid(visible=True, alpha=0.3)
    # 測定期間が長いとティックラベルが重なるため、間隔を自動調整して見やすくする
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_water_content_html(trend_df: pl.DataFrame, path: Path) -> None:
    peak_distance_ns = trend_df["peak_distance"].to_numpy() * 1e9
    fig = go.Figure(
        go.Scatter(
            x=trend_df["timestamp"],
            y=trend_df["water_content"],
            mode="lines",
            line={"color": WATER_CONTENT_COLOR, "width": 2},
            name="Water Content",
            customdata=peak_distance_ns,
            hovertemplate="%{x}<br>Water Content: %{y:.3f}%<br>Peak Distance: %{customdata:.3f} ns<extra></extra>",
        ),
    )
    fig.update_layout(
        template="plotly_white",
        xaxis={"title": "Time", **_TIME_RANGE_AXIS_OPTIONS},
        yaxis_title="Water Content (%)",
        hovermode="x unified",
    )
    fig.write_html(path, include_plotlyjs=True)


def _nearest_y(t_axis: np.ndarray, impulse_response: np.ndarray, peak_x: float) -> float:
    idx = int(np.abs(t_axis - peak_x).argmin())
    return float(impulse_response[idx])


def _write_impulse_response_html(impulse_df: pl.DataFrame, path: Path) -> None:
    timestamps = impulse_df["timestamp"].to_list()
    t_axes = impulse_df["t_axis"].to_list()
    impulse_responses = impulse_df["impulse_response"].to_list()
    peak_positions_list = impulse_df["peak_positions"].to_list()

    frames = []
    x_min, x_max = float("inf"), float("-inf")
    y_min, y_max = float("inf"), float("-inf")
    for timestamp, t_axis, impulse_response, peak_positions in zip(
        timestamps,
        t_axes,
        impulse_responses,
        peak_positions_list,
        strict=True,
    ):
        t_axis_arr = np.asarray(t_axis)
        impulse_arr = np.asarray(impulse_response)
        x_min, x_max = min(x_min, t_axis_arr.min()), max(x_max, t_axis_arr.max())
        y_min, y_max = min(y_min, impulse_arr.min()), max(y_max, impulse_arr.max())

        peaks = (peak_positions or [])[:2]
        peak_ys = [_nearest_y(t_axis_arr, impulse_arr, x) for x in peaks]

        frames.append(
            go.Frame(
                name=str(timestamp),
                data=[
                    go.Scatter(
                        x=t_axis_arr,
                        y=impulse_arr,
                        mode="lines",
                        line={"color": IMPULSE_RESPONSE_COLOR, "width": 2},
                        name="Impulse Response",
                    ),
                    go.Scatter(
                        x=peaks,
                        y=peak_ys,
                        mode="markers",
                        marker={"color": PEAK_MARKER_COLOR, "size": 10, "symbol": "x"},
                        name="Peak Positions",
                        hovertemplate="t=%{x:.4e}s<br>amp=%{y:.4f}<extra></extra>",
                    ),
                ],
            ),
        )

    y_padding = (y_max - y_min) * 0.05
    # フレーム数が多い長時間測定でも読めるよう、ラベルは短い表記にしたうえで間引いて表示する
    label_stride = max(1, -(-len(frames) // MAX_LABELED_SLIDER_STEPS))
    steps = [
        {
            "method": "animate",
            "args": [
                [frame.name],
                {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}},
            ],
            "label": (
                timestamp.strftime("%m/%d %H:%M:%S")
                if i % label_stride == 0 or i == len(frames) - 1
                else ""
            ),
        }
        for i, (frame, timestamp) in enumerate(zip(frames, timestamps, strict=True))
    ]

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            template="plotly_white",
            xaxis={"title": "Time (s)", "range": [x_min, x_max]},
            yaxis={"title": "Amplitude", "range": [y_min - y_padding, y_max + y_padding]},
            sliders=[
                {
                    "active": 0,
                    "currentvalue": {"prefix": "Time: "},
                    "pad": {"t": 50},
                    "steps": steps,
                    "transition": {"duration": 0},
                },
            ],
        ),
    )
    fig.write_html(path, include_plotlyjs=True)


def _write_spectrogram_html(impulse_df: pl.DataFrame, trend_df: pl.DataFrame, path: Path) -> None:
    """インパルス応答をスペクトログラムとして水分量トレンドと時間軸を揃えて重ねて表示する。

    上段: 時間 x 遅延時間 のヒートマップ(色=振幅)
    下段: 水分量の時間変化(上段と同じ時間軸を共有し、ズーム連動する)
    """
    timestamps = impulse_df["timestamp"].to_list()
    # t_axisは計測設定が同じであればレコード間で共通のため、先頭行のものを使う
    t_axis_ns = np.asarray(impulse_df["t_axis"][0]) * 1e9
    z = np.array(impulse_df["impulse_response"].to_list()).T

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
        subplot_titles=("Impulse Response Spectrogram", "Water Content"),
    )
    fig.add_trace(
        go.Heatmap(
            x=timestamps,
            y=t_axis_ns,
            z=z,
            colorscale=SPECTROGRAM_COLORSCALE,
            colorbar={"title": "Amplitude", "len": 0.55, "y": 0.82},
            hovertemplate="Time=%{x}<br>Delay=%{y:.3f} ns<br>Amplitude=%{z:.4f}<extra></extra>",
            name="Impulse Response",
        ),
        row=1,
        col=1,
    )

    peak_distance_ns = trend_df["peak_distance"].to_numpy() * 1e9
    fig.add_trace(
        go.Scatter(
            x=trend_df["timestamp"],
            y=trend_df["water_content"],
            mode="lines",
            line={"color": WATER_CONTENT_COLOR, "width": 2},
            name="Water Content",
            customdata=peak_distance_ns,
            hovertemplate="Water Content: %{y:.3f}%<br>Peak Distance: %{customdata:.3f} ns<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="Delay (ns)", row=1, col=1)
    fig.update_yaxes(title_text="Water Content (%)", row=2, col=1)
    # 下段(共有x軸)にレンジスライダー/プリセットを付け、長時間測定でも区間を選んでズームできるようにする
    fig.update_xaxes(title_text="Time", row=2, col=1, **_TIME_RANGE_AXIS_OPTIONS)
    fig.update_layout(
        template="plotly_white",
        height=800,
        hovermode="x unified",
    )
    fig.write_html(path, include_plotlyjs=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="記録済みparquetファイルからグラフを生成する")
    parser.add_argument("data_path", help="parquetファイルが置かれたディレクトリ")
    parser.add_argument("output_path", help="グラフの出力先ディレクトリ")
    parser.add_argument(
        "--max-impulse-frames",
        type=int,
        default=200,
        help="インパルス応答スライダー/スペクトログラムに使う間引き後のフレーム数(既定値: 200)",
    )
    parser.add_argument(
        "--max-trend-points",
        type=int,
        default=5000,
        help="水分量トレンドグラフ(PNG/HTML)に使う間引き後の点数(既定値: 5000)",
    )
    args = parser.parse_args()
    postprocess(args.data_path, args.output_path, args.max_impulse_frames, args.max_trend_points)
