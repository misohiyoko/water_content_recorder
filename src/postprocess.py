import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

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
# スペクトログラム(インパルス応答/周波数応答ヒートマップ)用の配色(plotly組み込みのInferno)
SPECTROGRAM_COLORSCALE = "Inferno"
# スペクトログラムを対数(dB)表示する際の下限。これより弱い振幅は下限に張り付かせて表示する
SPECTROGRAM_DB_FLOOR = -60.0
# スライダーのラベルが重なって読めなくなるのを防ぐため、この件数を超えたら間引いてラベル表示する
MAX_LABELED_SLIDER_STEPS = 30
# water_contentがこの値以上(=ピーク誤検出などでcompute_water_contentが100%にクリップされた異常値)の
# レコードは、グラフを歪めるためcombined.parquet以外の出力から除外する
WATER_CONTENT_CLIP_THRESHOLD = 100.0
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
    data_path: str | Path | Sequence[str | Path],
    output_path: str | Path,
    max_impulse_frames: int = 800,
    max_trend_points: int = 20000,
    water_content_clip_threshold: float = WATER_CONTENT_CLIP_THRESHOLD,
    start_time: str | datetime | None = None,
    end_time: str | datetime | None = None,
) -> None:
    """parquetファイル群を読み込み、結合parquet・water_content推移・impulse_response波形・各種スペクトログラムを出力する。

    data_pathは単一フォルダ、または複数フォルダ(list等)を渡せる。複数指定した場合は
    全フォルダのparquetを1つに結合してから処理する。start_time/end_timeを指定すると、
    結合後のデータをその時刻範囲(両端含む)だけに絞り込んでから以降の処理・出力を行う。
    """
    data_paths = [Path(data_path)] if isinstance(data_path, (str, Path)) else [Path(p) for p in data_path]
    start_time = datetime.fromisoformat(start_time) if isinstance(start_time, str) else start_time
    end_time = datetime.fromisoformat(end_time) if isinstance(end_time, str) else end_time
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    console = Console()
    logger = setup_logger("postprocess", log_file=output_path / "postprocess.log", console=console)

    parquet_files = []
    for path in data_paths:
        if not path.exists():
            msg = f"{path} が見つかりません"
            raise FileNotFoundError(msg)
        parquet_files.extend(sorted(path.glob("*.parquet")))
    if not parquet_files:
        msg = f"{data_paths} にparquetファイルが見つかりません"
        raise FileNotFoundError(msg)
    logger.info(f"parquetファイルを {len(parquet_files)} 件見つけました")

    with make_progress(console=console, transient=True) as progress:
        task = progress.add_task("parquetファイルを結合中", total=None)
        combined_df = pl.scan_parquet(parquet_files).sort("timestamp").collect()
        # timestamp列はタイムゾーン付きなので、tz無しで指定された範囲もそれに合わせて解釈する
        tz = combined_df.schema["timestamp"].time_zone
        if start_time is not None and start_time.tzinfo is None and tz is not None:
            start_time = start_time.replace(tzinfo=ZoneInfo(tz))
        if end_time is not None and end_time.tzinfo is None and tz is not None:
            end_time = end_time.replace(tzinfo=ZoneInfo(tz))
        if start_time is not None:
            combined_df = combined_df.filter(pl.col("timestamp") >= start_time)
        if end_time is not None:
            combined_df = combined_df.filter(pl.col("timestamp") <= end_time)
        progress.update(task, completed=1, total=1)

        n_rows = combined_df.height
        if n_rows == 0:
            msg = "指定された範囲にレコードがありません"
            raise ValueError(msg)
        logger.info(f"{n_rows} 件のレコードを読み込みました")

        # water_contentがcompute_water_content側で100%にクリップされた異常値(ピーク誤検出等)を
        # グラフ用データから除外する。combined.parquetには影響しない(生データのまま出力する)
        plot_df = combined_df.filter(pl.col("water_content") < water_content_clip_threshold)
        n_clipped = n_rows - plot_df.height
        if n_clipped:
            logger.info(
                f"water_contentが{water_content_clip_threshold}%以上のレコードを"
                f"{n_clipped}件、グラフ用データから除外しました",
            )
        n_plot_rows = plot_df.height

        trend_df = plot_df.select(["timestamp", "water_content", "peak_distance"])
        trend_plot_df = trend_df[_evenly_spaced_indices(n_plot_rows, max_trend_points)]
        logger.info(f"グラフ用に水分量トレンドを {trend_plot_df.height} 件に間引きました")

        task = progress.add_task("インパルス応答の波形を読み込み中", total=None)
        frame_indices = _evenly_spaced_indices(n_plot_rows, max_impulse_frames)
        impulse_df = plot_df[frame_indices].select(
            ["timestamp", "t_axis", "impulse_response", "peak_positions"],
        )
        spectrum_df = plot_df[frame_indices].select(
            ["timestamp", "frequencies", "s11_real", "s11_imag"],
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

    _write_frequency_spectrogram_html(spectrum_df, trend_plot_df, output_path / "frequency_spectrogram.html")
    logger.info("frequency_spectrogram.html を出力しました")


def _zoom_rescale_script(div_id: str, targets: list[dict]) -> str:
    """ズーム(レンジスライダー/プリセット/ドラッグ)時に、表示範囲内のデータでy軸・ヒートマップの色範囲を自動調整するJS。

    plotlyはx軸をズームしてもy軸/色範囲を追随させないため、埋め込み済みのtrace data(gd.data)から
    表示範囲内の値を都度計算し直し、Plotly.relayout/restyleで更新する。
    targetsの各要素は {"type": "line", "trace": idx, "yaxis": "yaxis"|"yaxis2"} または
    {"type": "heatmap", "trace": idx, "floor_db": float}。
    """
    return f"""
(function() {{
    var gd = document.getElementById({div_id!r});
    var targets = {json.dumps(targets)};
    function toMs(v) {{ return (v instanceof Date) ? v.getTime() : new Date(v).getTime(); }}
    function getXRange(ed) {{
        var lo, hi, reset = false;
        Object.keys(ed).forEach(function(k) {{
            if (/^xaxis\\d*\\.autorange$/.test(k) && ed[k]) reset = true;
            if (/^xaxis\\d*\\.range\\[0\\]$/.test(k)) lo = ed[k];
            if (/^xaxis\\d*\\.range\\[1\\]$/.test(k)) hi = ed[k];
            if (/^xaxis\\d*\\.range$/.test(k) && Array.isArray(ed[k])) {{ lo = ed[k][0]; hi = ed[k][1]; }}
        }});
        if (reset) return null;
        if (lo === undefined || hi === undefined) return undefined;
        return [toMs(lo), toMs(hi)];
    }}
    gd.on('plotly_relayout', function(ed) {{
        var xr = getXRange(ed);
        if (xr === undefined) return;
        targets.forEach(function(t) {{
            var trace = gd.data[t.trace];
            if (xr === null) {{
                if (t.type === 'line') {{
                    var reset_update = {{}};
                    reset_update[t.yaxis + '.autorange'] = true;
                    Plotly.relayout(gd, reset_update);
                }}
                return;
            }}
            var colIdx = [];
            for (var i = 0; i < trace.x.length; i++) {{
                var tm = toMs(trace.x[i]);
                if (tm >= xr[0] && tm <= xr[1]) colIdx.push(i);
            }}
            if (colIdx.length === 0) return;
            if (t.type === 'line') {{
                var vals = colIdx.map(function(i) {{ return trace.y[i]; }});
                var mn = Math.min.apply(null, vals), mx = Math.max.apply(null, vals);
                var pad = (mx - mn) * 0.05 || Math.abs(mx) * 0.05 || 1;
                var update = {{}};
                update[t.yaxis + '.range'] = [mn - pad, mx + pad];
                update[t.yaxis + '.autorange'] = false;
                Plotly.relayout(gd, update);
            }} else if (t.type === 'heatmap') {{
                var zmn = Infinity, zmx = -Infinity;
                colIdx.forEach(function(i) {{
                    trace.z.forEach(function(row) {{
                        var v = row[i];
                        if (v < zmn) zmn = v;
                        if (v > zmx) zmx = v;
                    }});
                }});
                if (!isFinite(zmn) || !isFinite(zmx)) return;
                Plotly.restyle(gd, {{zmin: [Math.max(zmn, t.floor_db)], zmax: [zmx]}}, [t.trace]);
            }}
        }});
    }});
}})();
"""


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
            # ズーム時にJS側(gd.data)からyの値を読み直すため、plotlyのnumpy用バイナリ圧縮(dtype/bdata)を
            # 避けて素のJSON配列にする(list化しないとブラウザ側でtrace.y[i]がundefinedになる)
            y=trend_df["water_content"].to_list(),
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
    fig.write_html(
        path,
        include_plotlyjs=True,
        div_id="water_content_trend",
        post_script=_zoom_rescale_script("water_content_trend", [{"type": "line", "trace": 0, "yaxis": "yaxis"}]),
    )


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


def _to_db(z_linear: np.ndarray, floor_db: float = SPECTROGRAM_DB_FLOOR) -> np.ndarray:
    """振幅の配列を、全体の最大値を0dBとした対数(dB)スケールに変換する。

    floor_dbより弱い振幅はfloor_dbに張り付かせ、微弱なノイズ成分を強調しすぎないようにする。
    """
    z_abs = np.abs(z_linear)
    peak = np.max(z_abs)
    ref = peak if peak > 0 else 1.0
    ratio = np.maximum(z_abs / ref, 10 ** (floor_db / 20))
    return 20 * np.log10(ratio)


def _add_water_content_row(fig: go.Figure, trend_df: pl.DataFrame, *, row: int, col: int) -> None:
    """スペクトログラムの下段に、時間軸を共有した水分量トレンドを追加する(共通処理)。"""
    peak_distance_ns = trend_df["peak_distance"].to_numpy() * 1e9
    fig.add_trace(
        go.Scatter(
            x=trend_df["timestamp"],
            # 理由は_write_water_content_html関数と同じ、ズーム時JS側で読めるよう素のJSON配列にする
            y=trend_df["water_content"].to_list(),
            mode="lines",
            line={"color": WATER_CONTENT_COLOR, "width": 2},
            name="Water Content",
            customdata=peak_distance_ns,
            hovertemplate="Water Content: %{y:.3f}%<br>Peak Distance: %{customdata:.3f} ns<extra></extra>",
        ),
        row=row,
        col=col,
    )
    fig.update_yaxes(title_text="Water Content (%)", row=row, col=col)
    # 下段(共有x軸)にレンジスライダー/プリセットを付け、長時間測定でも区間を選んでズームできるようにする
    fig.update_xaxes(title_text="Time", row=row, col=col, **_TIME_RANGE_AXIS_OPTIONS)


def _write_spectrogram_html(impulse_df: pl.DataFrame, trend_df: pl.DataFrame, path: Path) -> None:
    """インパルス応答をスペクトログラムとして水分量トレンドと時間軸を揃えて重ねて表示する。

    上段: 時間 x 遅延時間 のヒートマップ(色=振幅、対数(dB)表示)
    下段: 水分量の時間変化(上段と同じ時間軸を共有し、ズーム連動する)
    """
    timestamps = impulse_df["timestamp"].to_list()
    # t_axisは計測設定が同じであればレコード間で共通のため、先頭行のものを使う
    t_axis_ns = np.asarray(impulse_df["t_axis"][0]) * 1e9
    z_db = _to_db(np.array(impulse_df["impulse_response"].to_list()).T)

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
            # ズームJSがgd.data[].zを読むため、numpy用バイナリ圧縮を避けて素のJSON配列にする
            z=z_db.tolist(),
            zmin=SPECTROGRAM_DB_FLOOR,
            zmax=0.0,
            colorscale=SPECTROGRAM_COLORSCALE,
            colorbar={"title": "Amplitude (dB)", "len": 0.55, "y": 0.82},
            hovertemplate="Time=%{x}<br>Delay=%{y:.3f} ns<br>Amplitude=%{z:.1f} dB<extra></extra>",
            name="Impulse Response",
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Delay (ns)", row=1, col=1)

    _add_water_content_row(fig, trend_df, row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        height=800,
        hovermode="x unified",
    )
    fig.write_html(
        path,
        include_plotlyjs=True,
        div_id="spectrogram",
        post_script=_zoom_rescale_script(
            "spectrogram",
            [
                {"type": "heatmap", "trace": 0, "floor_db": SPECTROGRAM_DB_FLOOR},
                {"type": "line", "trace": 1, "yaxis": "yaxis2"},
            ],
        ),
    )


def _write_frequency_spectrogram_html(spectrum_df: pl.DataFrame, trend_df: pl.DataFrame, path: Path) -> None:
    """時間ごとのS11(周波数応答)をスペクトログラムとして水分量トレンドと時間軸を揃えて重ねて表示する。

    上段: 時間 x 周波数 のヒートマップ(色=|S11|、対数(dB)表示)
    下段: 水分量の時間変化(上段と同じ時間軸を共有し、ズーム連動する)
    """
    timestamps = spectrum_df["timestamp"].to_list()
    # frequenciesは計測設定が同じであればレコード間で共通のため、先頭行のものを使う
    frequencies_mhz = np.asarray(spectrum_df["frequencies"][0]) / 1e6
    s11_real = np.array(spectrum_df["s11_real"].to_list())
    s11_imag = np.array(spectrum_df["s11_imag"].to_list())
    s11_mag = np.abs(s11_real + 1j * s11_imag)
    z_db = _to_db(s11_mag.T)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
        subplot_titles=("S11 Frequency Spectrogram", "Water Content"),
    )
    fig.add_trace(
        go.Heatmap(
            x=timestamps,
            y=frequencies_mhz,
            # 理由は_write_spectrogram_html関数と同じ、ズーム時JS側で読めるよう素のJSON配列にする
            z=z_db.tolist(),
            zmin=SPECTROGRAM_DB_FLOOR,
            zmax=0.0,
            colorscale=SPECTROGRAM_COLORSCALE,
            colorbar={"title": "|S11| (dB)", "len": 0.55, "y": 0.82},
            hovertemplate="Time=%{x}<br>Frequency=%{y:.1f} MHz<br>|S11|=%{z:.1f} dB<extra></extra>",
            name="S11 Magnitude",
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Frequency (MHz)", row=1, col=1)

    _add_water_content_row(fig, trend_df, row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        height=800,
        hovermode="x unified",
    )
    fig.write_html(
        path,
        include_plotlyjs=True,
        div_id="frequency_spectrogram",
        post_script=_zoom_rescale_script(
            "frequency_spectrogram",
            [
                {"type": "heatmap", "trace": 0, "floor_db": SPECTROGRAM_DB_FLOOR},
                {"type": "line", "trace": 1, "yaxis": "yaxis2"},
            ],
        ),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="記録済みparquetファイルからグラフを生成する")
    parser.add_argument(
        "data_path",
        nargs="+",
        help="parquetファイルが置かれたディレクトリ(複数指定すると結合して処理する)",
    )
    parser.add_argument("output_path", help="グラフの出力先ディレクトリ")
    parser.add_argument(
        "--max-impulse-frames",
        type=int,
        default=800,
        help="インパルス応答スライダー/スペクトログラムに使う間引き後のフレーム数(既定値: 800)",
    )
    parser.add_argument(
        "--max-trend-points",
        type=int,
        default=20000,
        help="水分量トレンドグラフ(PNG/HTML)に使う間引き後の点数(既定値: 20000)",
    )
    parser.add_argument(
        "--water-content-clip-threshold",
        type=float,
        default=WATER_CONTENT_CLIP_THRESHOLD,
        help=(
            "water_contentがこの値以上(compute_water_content側でクリップされた異常値)の"
            f"レコードをグラフから除外する(既定値: {WATER_CONTENT_CLIP_THRESHOLD})"
        ),
    )
    parser.add_argument("--start-time", help="この時刻(ISO8601)以降のレコードのみ処理する")
    parser.add_argument("--end-time", help="この時刻(ISO8601)以前のレコードのみ処理する")
    args = parser.parse_args()
    postprocess(
        args.data_path,
        args.output_path,
        args.max_impulse_frames,
        args.max_trend_points,
        args.water_content_clip_threshold,
        args.start_time,
        args.end_time,
    )
