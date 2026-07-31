from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl
from rich.console import Console

from shared_python import make_progress, setup_logger

WATER_CONTENT_COLOR = "#eb6834"
IMPULSE_RESPONSE_COLOR = "#2a78d6"
PEAK_MARKER_COLOR = "#eb6834"


def postprocess(
    data_path: str | Path,
    output_path: str | Path,
    max_impulse_frames: int = 200,
) -> None:
    """parquetファイル群を読み込み、water_contentの時間変化(CSV・グラフ)とimpulse_responseの波形をグラフ出力する。"""
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
        task = progress.add_task("水分量トレンドを読み込み中", total=None)
        sorted_lf = pl.scan_parquet(parquet_files).sort("timestamp").with_row_index("row_idx")
        trend_df = sorted_lf.select(["row_idx", "timestamp", "water_content", "peak_distance"]).collect()
        progress.update(task, completed=1, total=1)

        n_rows = trend_df.height
        logger.info(f"{n_rows} 件のレコードを読み込みました")

        task = progress.add_task("インパルス応答の波形を読み込み中", total=None)
        n_frames = min(max_impulse_frames, n_rows)
        frame_indices = sorted(set(np.linspace(0, n_rows - 1, num=n_frames, dtype=int).tolist()))
        impulse_df = (
            sorted_lf.filter(pl.col("row_idx").is_in(frame_indices))
            .select(["timestamp", "t_axis", "impulse_response", "peak_positions"])
            .collect()
        )
        progress.update(task, completed=1, total=1)
        logger.info(f"波形グラフ用に {impulse_df.height} 件のレコードを間引いて読み込みました")

    _write_water_content_csv(trend_df, output_path / "water_content_trend.csv")
    logger.info("water_content_trend.csv を出力しました")

    _write_water_content_png(trend_df, output_path / "water_content_trend.png")
    logger.info("water_content_trend.png を出力しました")

    _write_water_content_html(trend_df, output_path / "water_content_trend.html")
    logger.info("water_content_trend.html を出力しました")

    _write_impulse_response_html(impulse_df, output_path / "impulse_response.html")
    logger.info("impulse_response.html を出力しました")


def _write_water_content_csv(trend_df: pl.DataFrame, path: Path) -> None:
    trend_df.select(["timestamp", "water_content", "peak_distance"]).write_csv(path)


def _write_water_content_png(trend_df: pl.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(trend_df["timestamp"], trend_df["water_content"], color=WATER_CONTENT_COLOR, linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Water Content (%)")
    ax.grid(visible=True, alpha=0.3)
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
        xaxis_title="Time",
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
    steps = [
        {
            "method": "animate",
            "args": [
                [frame.name],
                {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}},
            ],
            "label": frame.name,
        }
        for frame in frames
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="記録済みparquetファイルからグラフを生成する")
    parser.add_argument("data_path", help="parquetファイルが置かれたディレクトリ")
    parser.add_argument("output_path", help="グラフの出力先ディレクトリ")
    parser.add_argument(
        "--max-impulse-frames",
        type=int,
        default=200,
        help="インパルス応答スライダーに使う間引き後のフレーム数(既定値: 200)",
    )
    args = parser.parse_args()
    postprocess(args.data_path, args.output_path, args.max_impulse_frames)
