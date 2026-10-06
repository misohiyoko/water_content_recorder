"""datashaderを使い、指定フォルダ群のwater_content推移を画像として描画する。

postprocess.pyのグラフ(water_content_trend.html/png)は間引いた点数(既定2万点)でしか
描けないが、datashaderは画素ごとに集約してラスタ画像を作るため、行数が何百万件あっても
間引かずにそのままの密度で描画できる(大量データでも軽い)。
"""

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl

WATER_CONTENT_COLOR = "#eb6834"
BACKGROUND_COLOR = "white"
JST = timezone(timedelta(hours=9), "JST")


def render_water_content(
    data_path: str | Path | Sequence[str | Path],
    output_path: str | Path,
    width: int = 1600,
    height: int = 600,
) -> None:
    """指定フォルダ(群)のwater_content推移をdatashaderでラスタライズし、PNGとして保存する。"""
    data_paths = [Path(data_path)] if isinstance(data_path, (str, Path)) else [Path(p) for p in data_path]

    parquet_files = []
    for path in data_paths:
        if not path.exists():
            msg = f"{path} が見つかりません"
            raise FileNotFoundError(msg)
        parquet_files.extend(sorted(path.glob("*.parquet")))
    if not parquet_files:
        msg = f"{data_paths} にparquetファイルが見つかりません"
        raise FileNotFoundError(msg)

    # 軽量な2列だけを読み込む(間引きは行わない。datashader側でラスタライズするため不要)
    df = pl.scan_parquet(parquet_files).select(["timestamp", "water_content"]).sort("timestamp").collect()
    if df.height == 0:
        msg = f"{data_paths} にレコードがありません"
        raise ValueError(msg)

    # datashaderのCanvas.lineはx軸に実数値しか受け付けないため、timestampを秒単位の
    # エポック数値に変換してから渡し、表示側(matplotlib)でdatetime軸に描き戻す
    x_numeric = df["timestamp"].dt.epoch("s").to_numpy().astype(np.float64)
    pdf = pl.DataFrame({"x": x_numeric, "water_content": df["water_content"]}).to_pandas()

    canvas = ds.Canvas(plot_width=width, plot_height=height)
    agg = canvas.line(pdf, "x", "water_content", line_width=1)
    img = tf.set_background(tf.shade(agg, cmap=[WATER_CONTENT_COLOR], how="linear"), BACKGROUND_COLOR)

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    x_min, x_max = x_numeric.min(), x_numeric.max()
    y_min, y_max = float(np.nanmin(pdf["water_content"])), float(np.nanmax(pdf["water_content"]))
    ax.imshow(
        np.array(img.to_pil()),
        extent=(x_min, x_max, y_min, y_max),
        aspect="auto",
        origin="upper",
    )
    # x軸はdatashader用に秒単位のエポック数値にしているため、目盛りラベルだけdatetimeの表記に戻す
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _pos: datetime.fromtimestamp(x, tz=JST).strftime("%m/%d %H:%M")),
    )
    fig.autofmt_xdate()
    ax.set_xlabel("Time (JST)")
    ax.set_ylabel("Water Content (%)")
    ax.set_title(f"Water Content ({df.height:,} points)")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="datashaderでフォルダ内のwater_content推移を画像化する")
    parser.add_argument("data_path", nargs="+", help="parquetファイルが置かれたディレクトリ(複数指定すると結合して描画する)")
    parser.add_argument("output_path", help="出力する画像ファイルパス(.png)")
    parser.add_argument("--width", type=int, default=1600, help="画像の幅(px、既定値: 1600)")
    parser.add_argument("--height", type=int, default=600, help="画像の高さ(px、既定値: 600)")
    args = parser.parse_args()
    render_water_content(args.data_path, args.output_path, args.width, args.height)
