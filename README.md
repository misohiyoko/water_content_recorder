# water_content_recorder

NanoVNA / LiteVNA を使ってTDR(Time Domain Reflectometry)測定を行い、S11から水分量を推定・記録するツール。バックエンド(Python)が測定・記録・簡易API配信を行い、フロントエンド(SvelteKit)が監視画面を表示する。

## 別PCへ移動する場合の手順

このリポジトリは以下を前提に環境非依存になっている: 依存関係は `uv`(Python)/`pnpm`(フロントエンド)がロックファイル通りに再構築し、機器固有の設定は `.env`(gitignore対象)に分離してある。**リポジトリ本体をコピーするのではなく、必ずgit経由で取得すること**(`.venv`やビルド成果物はコピーすると壊れやすいため)。

```bash
# 1. サブモジュール(shared_python)ごと取得
git clone --recurse-submodules https://github.com/misohiyoko/water_content_recorder.git
cd water_content_recorder
# すでにcloneしてしまっている場合は:
#   git submodule update --init --recursive

# 2. uvのインストール(未インストールの場合)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 3. Python依存関係を同期(uv.lock通りに.venvを作成)
uv sync

# 4. 校正係数などローカル設定を作成
copy .env.example .env
# 中身(CALIBRATION_COEFFICIENT)は新しい測定環境で再キャリブレーションするまで既定値のままでよい

# 5. フロントエンドをビルド(初回・および frontend/src 変更時)
cd frontend
pnpm install
pnpm build
cd ..

# 6. 実行
uv run python src\main.py
# もしくは
run_and_postprocess.bat
```

### 機器側の注意

- NanoVNA/LiteVNAはUSB接続時にVID/PIDで自動検出される([vna.py](src/water_content_recorder/vna.py))。COMポート等をコードに書く必要はない。
- 新しいPCで初めて機器を挿す場合、Windows側にUSB CDCシリアルドライバのインストールが必要になることがある。
- `.env` の `CALIBRATION_COEFFICIENT` は測定系(ケーブル長・センサ個体差など)に依存する値なので、PCではなく**機器構成ごと**に別方式で測定した水分量を使ってキャリブレーションし直すこと(監視画面の「キャリブレーション」ボタン、または `SignalProcessing.calibrate()`)。

### 移動時にコピーしてはいけないもの(ロックファイルから再生成する)

- `.venv/` — `uv sync` で再構築する
- `frontend/node_modules/`, `frontend/build/`, `frontend/.svelte-kit/` — `pnpm install && pnpm build` で再構築する

### 測定データを引き継ぎたい場合

`data/` はgitignore対象(容量が大きく、機器固有のためリポジトリに含めない)。過去の測定を引き継ぐ場合は `data/` フォルダを個別にコピーするか、`postprocess.py` の出力(グラフ・CSV)だけを共有する。

## 開発

```bash
uv run ruff check .      # lint
uv run ty check          # 型チェック
cd frontend && pnpm check  # フロントエンドの型チェック
```
