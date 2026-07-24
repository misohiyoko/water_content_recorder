from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

C = 3e8


@dataclass
class SignalState:
    frequencies: np.ndarray
    s11: np.ndarray
    t_axis: np.ndarray
    d_axis: np.ndarray
    step_response: np.ndarray
    impulse_response: np.ndarray
    peak_distance: float
    water_content: float


class SignalProcessing:
    def __init__(self, tdr_window: str = "hann", tdr_zero_pad_factor: int = 8, velocity_factor: float = 1.0):
        self.tdr_window = tdr_window
        self.tdr_zero_pad_factor = tdr_zero_pad_factor
        self.velocity_factor = velocity_factor

    def process_signal(self, frequencies: np.ndarray, s11: np.ndarray) -> SignalState:
        t_axis, d_axis, step_response, impulse_response = self.compute_tdr(frequencies, s11)
        peak_distance = self.get_peak_distance_ndarray(impulse_response, t_axis)
        water_content = self.compute_water_content(peak_distance)
        return SignalState(
            frequencies=frequencies,
            s11=s11,
            t_axis=t_axis,
            d_axis=d_axis,
            step_response=step_response,
            impulse_response=impulse_response,
            peak_distance=peak_distance,
            water_content=water_content,
        )

    def compute_tdr(
        self,
        frequencies: np.ndarray,
        s11: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """S11データからTDRインパルス応答・ステップ応答と時間/距離軸を計算する。

        処理手順:
        1. DC (0 Hz) へ線形外挿して片側スペクトルを構成
        2. 窓関数を適用（TDR_WINDOW 定数で切替可能）
        3. ゼロパディングでポイント数を増やす
        4. np.fft.irfft でエルミート対称IFFTを行い実数インパルス応答を得る
        5. cumsum でステップ応答に変換

        Args:
            frequencies: 周波数配列 (Hz)
            s11: S11複素配列

        Returns:
            (時間軸 (s), 距離軸 (m), ステップ応答, インパルス応答) のタプル

        """
        # 1. DC外挿: 最初の2点から0Hzへ線形外挿
        df = frequencies[1] - frequencies[0]
        slope = (s11[1] - s11[0]) / df
        s11_dc = s11[0] - slope * frequencies[0]
        s11_dc = complex(s11_dc.real, 0.0)

        # DC + 実測データを結合した片側スペクトル
        s11_onesided = np.concatenate([[s11_dc], s11])
        n_onesided = len(s11_onesided)

        # 2. 窓関数の適用
        if self.tdr_window == "hann":
            full_window = np.hanning(n_onesided * 2)
            window = full_window[n_onesided:]
        elif self.tdr_window == "blackman":
            window = np.blackman(n_onesided)
        else:
            window = np.ones(n_onesided)
        s11_windowed = s11_onesided * window

        # 3. ゼロパディング: 2の冪乗に切り上げて TDR_ZERO_PAD_FACTOR 倍
        n_fft_half = 2 ** int(np.ceil(np.log2(n_onesided))) * self.tdr_zero_pad_factor
        spectrum = np.zeros(n_fft_half, dtype=complex)
        spectrum[:n_onesided] = s11_windowed

        # 4. irfft: エルミート対称を自動構成して実数インパルス応答を得る
        #    irfft(x, n) の出力長 = n（nを明示することで安定した長さを確保）
        n_ifft = (n_fft_half - 1) * 2
        impulse = np.fft.irfft(spectrum, n_ifft)

        # 5. 累積和 → ステップ応答
        step = np.cumsum(impulse)

        # 時間軸・距離軸（サンプリング周波数 = 2 * f_max）
        dt = 1.0 / (n_ifft * df)
        t_axis = np.arange(n_ifft) * dt
        d_axis = t_axis * C * self.velocity_factor / 2  # 往復 → 片道

        valid = n_ifft // 2

        return t_axis[:valid], d_axis[:valid], step[:valid], impulse[:valid]

    def get_peak_distance_ndarray(
        self,
        y_array: np.ndarray,
        x_array: np.ndarray,
        peak1_rank: int = 0,
        peak2_rank: int = 1,
    ) -> float:
        """ndarrayを入力として、指定した順位（高さ降順）のピーク間のxの距離を計算する。

        Parameters
        ----------
        - y_array (np.ndarray): y軸の配列
        - x_array (np.ndarray): x軸の配列 (元の time_ns に相当)
        - peak1_rank (int): 比較する1つ目のピークの順位（0が最も高いピーク）
        - peak2_rank (int): 比較する2つ目のピークの順位（1が2番目に高いピーク）

        Returns
        -------
        - float: ピーク間の距離（絶対値）。ピークが足りない場合は np.nan を返す。

        """
        # 1. ピークのインデックスを取得
        peaks, _ = find_peaks(y_array, prominence=0.0)

        # ピーク数が指定した順位に満たない場合のエラー回避
        if len(peaks) <= max(peak1_rank, peak2_rank):
            return np.nan

        # 2. ピークの高さ (yの値) を取得し、降順にソートしたインデックス配列を作成
        peak_heights = y_array[peaks]
        sorted_peak_indices = peaks[np.argsort(peak_heights)[::-1]]

        # 3. 指定された順位のピークに対応する x_array の値を取得して距離を計算
        x1 = x_array[sorted_peak_indices[peak1_rank]]
        x2 = x_array[sorted_peak_indices[peak2_rank]]

        return abs(x1 - x2)

    def compute_water_content(self, peak_distance: float) -> float:
        if np.isnan(peak_distance):
            return np.nan
        if peak_distance <= 0:
            return 0.0
        if peak_distance <= 13.33e-10:
            return (peak_distance - 6.244e-10) / 1.552e-11
        if peak_distance <= 14.45e-10:
            return (peak_distance - 9.942e-10) / 7.42e-12
        return (peak_distance - 2.589e-11) / 2.334e-11
