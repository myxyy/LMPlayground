# プロジェクトについて
minGRUのようなRNN動作ができる言語モデルの実験です

# コマンド
Pythonの実行は`uv`コマンドを使用してください

- チャット例
```
uv run python src/lm_playground/predict_chat.py experiment=qgru
```
複数GPUを用いた訓練等には`torchrun`コマンドを使用してください
- 訓練例
```
uv run torchrun --standalone --nproc_per_node=1 src/lm_playground/train.py experiment=qgru
```

# ワークフロー
バージョン管理はGitを使用しています。作業の際はブランチを切って適宜コミットを積んでください。
プロジェクトが動いているDockerコンテナ上のコンソールで日本語が打てないようなのでコミットメッセージは英語でお願いします。
