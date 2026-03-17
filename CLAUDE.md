# プロジェクトについて
Quasi GRU (QGRU) などのRNN系アーキテクチャを用いた日本語言語モデルの実験プロジェクトです。
minGRUライクなスキャンベース再帰を実装し、ストリーミング生成や隠れ状態の永続化をサポートします。

- Python: 3.13
- パッケージ管理: uv
- 設定管理: Hydra
- 学習フレームワーク: PyTorch (DDP対応)

# ディレクトリ構成

```
src/lm_playground/
├── model/
│   ├── qgru.py        # Quaternion GRUモデル
│   ├── qlstm.py       # Quaternion LSTMモデル
│   └── ttt.py         # TTT / Multi-head MLPレイヤー
├── train.py           # 訓練エントリーポイント
├── trainer.py         # Trainerクラス (DDP対応)
├── predict_chat.py    # 対話インターフェース
├── predict_stream.py  # ストリーミング生成
└── generator.py       # 生成ユーティリティ

src/config/
├── config.yaml        # Hydraメイン設定
├── experiment/        # 実験設定 (qgru.yaml など)
├── model/             # モデル設定
├── train/             # 学習設定
└── tokenizer/         # トークナイザ設定

resources/
├── checkpoints/       # モデルチェックポイント
└── tokenizers/        # トークナイザキャッシュ
```

# コマンド
Pythonの実行は`uv`コマンドを使用してください

- チャット例
```
uv run python src/lm_playground/predict_chat.py experiment=qgru
```

- ストリーミング生成例
```
uv run python src/lm_playground/predict_stream.py experiment=qgru
```

複数GPUを用いた訓練等には`torchrun`コマンドを使用してください

- 訓練例 (シングルGPU)
```
uv run torchrun --standalone --nproc_per_node=1 src/lm_playground/train.py experiment=qgru
```

- 訓練例 (マルチGPU、例: 4枚)
```
uv run torchrun --standalone --nproc_per_node=4 src/lm_playground/train.py experiment=qgru
```

- Docker経由での起動
```
docker compose up -d
docker compose exec lm-playground bash
```

# 設定 (Hydra)
実験設定は `src/config/experiment/` に YAML ファイルとして追加します。
`experiment=<name>` でオーバーライドして使用します。

主な設定パラメータ (`src/config/train/default.yaml`):
- `batch_size`: 2
- `max_length`: 1024
- `max_epochs`: 4
- `validation_checkpoint_interval`: 500
- optimizer: RAdamScheduleFree (lr=1e-4)

# ワークフロー
バージョン管理はGitを使用しています。作業の際はブランチを切って適宜コミットを積んでください。
プロジェクトが動いているDockerコンテナ上のコンソールで日本語が打てないようなのでコミットメッセージは英語でお願いします。
