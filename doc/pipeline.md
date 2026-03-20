# やること
任意のモデルでパイプライン並列の学習ができるようにする

# 現状できていることと課題
QGRUモデルのパイプライン並列学習は可能だがTTT等のモデルには対応していない

# 実装方針
* `trainer.py`の`Trainer`クラスに対応する形で`pipeline_trainer.py`に`PipelineTrainer`クラスを実装して`train_pipeline.py`内の訓練処理を移す
* `QGRUPipelineStage`と同様に`QLSTMPipelineStage`及び`TTTPipelineStage`を作成して学習できるようにする

# 実装チェックリスト

## PipelineStageクラス
- [x] `QGRUPipelineStage` に `reconstruct_full_state_dict` を追加
- [x] `QLSTMPipelineStage` を `qlstm.py` に追加 (`split_config`, `load_from_full_model`, `reconstruct_full_state_dict`)
- [x] `TTTPipelineStage` を `ttt.py` に追加 (hidden stateはブロック内部で管理、`_hidden_init`不要)

## Trainer
- [x] `pipeline_trainer.py` に汎用 `PipelineTrainer` と `SingleGPUTrainer` を実装
  - `stage_class` パラメータでモデル非依存に
  - `save_weight` は `stage_class.reconstruct_full_state_dict()` を呼び出し
- [x] `train_pipeline.py` をエントリーポイントとしてリファクタリング
  - Hydra config の `pipeline_stage` キーでステージクラスを解決

## Config
- [x] `model/qgru.yaml` に `pipeline_stage` を追加
- [x] `model/qlstm.yaml` 新規作成
- [x] `model/ttt.yaml` 新規作成
- [x] `experiment/qlstm.yaml` 新規作成
- [x] `experiment/ttt.yaml` 新規作成

## テスト
- [ ] QGRU パイプライン訓練が既存と同様に動作する
- [ ] QLSTM パイプライン訓練が動作する
- [ ] TTT パイプライン訓練が動作する
- [ ] シングルGPUフォールバックが動作する

# 使い方

```bash
# QGRU (2-stage pipeline)
uv run torchrun --standalone --nproc_per_node=2 src/lm_playground/train_pipeline.py experiment=qgru

# QLSTM (2-stage pipeline)
uv run torchrun --standalone --nproc_per_node=2 src/lm_playground/train_pipeline.py experiment=qlstm

# TTT (2-stage pipeline)
uv run torchrun --standalone --nproc_per_node=2 src/lm_playground/train_pipeline.py experiment=ttt

# シングルGPU (任意のモデル)
uv run torchrun --standalone --nproc_per_node=1 src/lm_playground/train_pipeline.py experiment=ttt
```
