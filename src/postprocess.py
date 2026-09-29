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
from rich.console import Console

from shared_python import make_progress, setup_logger

WATER_CONTENT_COLOR = "#eb6834"
IMPULSE_RESPONSE_COLOR = "#2a78d6"
PEAK_MARKER_COLOR = "#eb6834"
# スライダーのラベルが重なって読めなくなるのを防ぐため、この件数を超えたら間引いてラベル表示する
MAX_LABELED_SLIDER_STEPS = 30
# water_contentがこの値以上(=ピーク誤検出などでcompute_water_contentが100%にクリップされた異常値)の
# レコードは、グラフを歪めるためcombined.parquet以外の出力から除外する
WATER_CONTENT_CLIP_THRESHOLD = 100.0
# combined.parquet書き出し時の行グループサイズ。Polarsのparquet書き出しは既定だと1ファイル分を
# 丸ごと1行グループとしてメモリ上に構築してから圧縮するため、大量データではここがメモリの支配的な
# ボトルネックになる。SignalRecorderの1parquetファイルあたりの行数(main.pyのBUFFER_SIZE)に
# 合わせておくと、出力側が入力ファイル1つ分ずつだけ処理すればよくなり最もメモリ効率が良い
# (実測: 2フォルダ/15,763行で、値200だと書き出しピーク1.35GB/17.6秒、
# BUFFER_SIZEに合わせた50だと0.45GB/9.8秒。ファイルサイズはほぼ同じ)。
# ponytail: 固定値による簡易チューニング。BUFFER_SIZEを変更した場合はこちらも合わせて変更すること。
COMBINED_PARQUET_ROW_GROUP_SIZE = 50
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
    """parquetファイル群を読み込み、結合parquet・water_content推移・impulse_response波形を出力する。

    data_pathは単一フォルダ、または複数フォルダ(list等)を渡せる。複数指定した場合は
    全フォルダのparquetを1つに結合してから処理する。start_time/end_timeを指定すると、
    結合後のデータをその時刻範囲(両端含む)だけに絞り込んでから以降の処理・出力を行う。
    """
    data_paths = [Path(data_path)] if isinstance(data_path, (str, Path)) else sorted(Path(p) for p in data_path)
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

        # 各ファイルは記録順(時系列順)に書き出され、フォルダ・ファイル名も時系列順にソート済みのため、
        # 通常はtimestampで明示的にソートし直さなくてもすでに時系列順になっている。
        # 明示的なsort("timestamp")は全件をメモリに展開してしまう(Polarsのストリーミングエンジンは
        # ソートをストリーミング処理できず、この時点で全データがメモリに載ってしまう)ため、
        # timestamp列だけ軽量に読んで順序を確認し、崩れている場合のみソートする(フォールバック)。
        timestamps = pl.scan_parquet(parquet_files).select("timestamp").collect()["timestamp"]
        if timestamps.is_sorted():
            combined_lazy = pl.scan_parquet(parquet_files)
        else:
            logger.warning(
                "timestampの並びがファイル順と一致しないため、ソートします(メモリ使用量が増えます)",
            )
            combined_lazy = pl.scan_parquet(parquet_files).sort("timestamp")

        # timestamp列はタイムゾーン付きなので、tz無しで指定された範囲もそれに合わせて解釈する
        tz = combined_lazy.collect_schema()["timestamp"].time_zone
        if start_time is not None and start_time.tzinfo is None and tz is not None:
            start_time = start_time.replace(tzinfo=ZoneInfo(tz))
        if end_time is not None and end_time.tzinfo is None and tz is not None:
            end_time = end_time.replace(tzinfo=ZoneInfo(tz))
        if start_time is not None:
            combined_lazy = combined_lazy.filter(pl.col("timestamp") >= start_time)
        if end_time is not None:
            combined_lazy = combined_lazy.filter(pl.col("timestamp") <= end_time)

        # 全件を一度にプロセスメモリへ展開せず、ストリーミングでcombined.parquetへ直接書き出す
        # (data_pathの合計が数十GBあってもここでメモリを使い切らないようにするため)
        combined_path = output_path / "combined.parquet"
        combined_lazy.sink_parquet(combined_path, row_group_size=COMBINED_PARQUET_ROW_GROUP_SIZE)
        progress.update(task, completed=1, total=1)

        # 行数はメタデータから取得するだけなので、ここでも全件はメモリに載らない
        n_rows = pl.scan_parquet(combined_path).select(pl.len()).collect().item()
        if n_rows == 0:
            combined_path.unlink(missing_ok=True)
            msg = "指定された範囲にレコードがありません"
            raise ValueError(msg)
        logger.info(f"{n_rows} 件のレコードを読み込みました")
        logger.info("combined.parquet を出力しました")

        # 1回目: 軽量な列(timestamp/water_content/peak_distance)だけを全件読み込む。
        # 1行あたりのサイズが小さいため、行数がどれだけ多くてもこの読み込みでメモリを使い切ることはない。
        # water_contentがcompute_water_content側で100%にクリップされた異常値(ピーク誤検出等)は、
        # グラフ用データから除外する(combined.parquetには影響しない、生データのまま出力済み)。
        task = progress.add_task("水分量トレンドを読み込み中", total=None)
        light_df = (
            pl.scan_parquet(combined_path)
            .with_row_index("__row__")
            .filter(pl.col("water_content") < water_content_clip_threshold)
            .select(["__row__", "timestamp", "water_content", "peak_distance"])
            .collect()
        )
        n_clipped = n_rows - light_df.height
        if n_clipped:
            logger.info(
                f"water_contentが{water_content_clip_threshold}%以上のレコードを"
                f"{n_clipped}件、グラフ用データから除外しました",
            )
        n_plot_rows = light_df.height

        trend_df = light_df.select(["timestamp", "water_content", "peak_distance"])
        trend_plot_df = trend_df[_evenly_spaced_indices(n_plot_rows, max_trend_points)]
        logger.info(f"グラフ用に水分量トレンドを {trend_plot_df.height} 件に間引きました")
        progress.update(task, completed=1, total=1)

        # 2回目: インパルス応答用の重い列(1行あたり数万点の配列)は、
        # 間引き後に実際に使う行(既定で最大800行)だけをピンポイントで読み込む。
        # combined_df/plot_dfのように全行×重い列をまとめてメモリに載せることは行わない。
        # コマの選び方は均等間隔ではなく、water_contentが実際に変化した場面を優先する。
        task = progress.add_task("インパルス応答の波形を読み込み中", total=None)
        frame_indices = _water_content_change_indices(light_df, max_impulse_frames)
        target_rows = light_df["__row__"][frame_indices].to_list()
        impulse_df = (
            pl.scan_parquet(combined_path)
            .with_row_index("__row__")
            .filter(pl.col("__row__").is_in(target_rows))
            .sort("__row__")  # フィルタ後も時系列順を保つ(並列実行時の順序ゆらぎ対策)
            .select(["timestamp", "t_axis", "impulse_response", "peak_positions"])
            .collect()
        )
        progress.update(task, completed=1, total=1)
        logger.info(f"波形グラフ用に {impulse_df.height} 件のレコードを間引いて読み込みました")

    _write_water_content_csv(trend_df, output_path / "water_content_trend.csv")
    logger.info("water_content_trend.csv を出力しました")

    _write_water_content_png(trend_plot_df, output_path / "water_content_trend.png")
    logger.info("water_content_trend.png を出力しました")

    _write_water_content_html(trend_plot_df, output_path / "water_content_trend.html")
    logger.info("water_content_trend.html を出力しました")

    _write_impulse_response_html(impulse_df, output_path / "impulse_response.html")
    logger.info("impulse_response.html を出力しました")


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


def _water_content_change_indices(light_df: pl.DataFrame, max_points: int) -> list[int]:
    """water_contentの変化量(|Δ|)が大きい行を優先して選び、最大max_points件の
    インデックス(0始まり、時系列順)を返す。値がほぼ動いていない区間にコマ数を割かず、
    実際に変化した場面にコマ数を割り当てるための間引き方法。先頭・末尾は変化量によらず必ず含める。
    """
    n = light_df.height
    if n <= max_points:
        return list(range(n))
    deltas = light_df["water_content"].diff().abs().fill_null(0.0).to_numpy()
    order = [int(i) for i in np.argsort(-deltas) if i not in (0, n - 1)]
    budget = max(max_points - 2, 0)
    return sorted({0, n - 1, *order[:budget]})


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
