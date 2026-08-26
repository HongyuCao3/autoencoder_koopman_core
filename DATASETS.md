# 数据集说明

本目录整理了当前 Autoencoder--Koopman 实验中最重要的 8 份原始轨迹数据。数据保持原始 JSON Lines 格式，每行对应一次轨迹中的一个交互 turn；没有混入 checkpoint、模型预测、绘图或调试产物。

实际文件已直接包含在项目中：

```text
autoencoder_koopman_core/datasets/
```

`datasets` 是真实目录而不是符号链接，因此 clone GitHub 仓库后即可直接运行。完整行数、轨迹数、文件大小与 SHA-256 见 `DATASET_MANIFEST.csv`，训练列映射见 `configs/datasets.json`。

## 已收录数据

| Key | 任务 | 输出维度 | Horizon | 轨迹数 | 备注 |
|---|---|---:|---:|---:|---|
| `sentence_length_t10` | 句子词数 5--100 | 1 | 10 | 252 | 主要长时域标量实验 |
| `character_length_t5` | 精确字符数 1--10 | 1 | 5 | 360 | 高度离散、接近定常 |
| `average_word_length_t5` | 平均词长 | 1 | 5 | 84 | smoke 规模 |
| `sentiment_t5` | 情感连续分数 | 1 | 5 | 60 | Cardiff readout，smoke 规模 |
| `formality_t5` | 正式度连续分数 | 1 | 5 | 252 | 受 readout 校准质量限制 |
| `even_odd_t5` | 奇偶整数类别 | 1 | 5 | 24 | 类别映射为标量，仅适合接口/负对照 |
| `vector_count_stage1_t10` | 词数 + 平均词长 | 2 | 10 | 288 | 同一条生成轨迹中的双目标控制 |
| `vector_count_stage2_t10` | 词数 + 平均词长 + 逗号数 | 3 | 10 | 540 | 同一条生成轨迹中的三目标控制 |

## 通用字段

所有数据至少提供：

- `trajectory_id`：轨迹级唯一标识；
- `topic_split`：预先锁定的 `train` / `validation` / `test` 划分；
- `turn`：从 1 开始的交互轮次；
- `topic`、`raw_generation`、prompt、反馈与解析/评分状态等原始收集字段。

标量任务使用 `normalized_output` 作为观测 `y_t`，使用 `effective_norm` 作为目标 `r`。Stage 1 使用
`[word_count_norm, avg_word_length_norm]`，Stage 2 再加入 `comma_count_norm`；对应目标列在注册表中逐项列出。

训练脚本只读取建模所需列，但保留原始文件中的文本与评分字段，便于后续重新做数据质量检查。

## 数据完整性

在项目根目录运行：

```bash
sha256sum -c <(awk -F, 'NR>1 {print $10 "  datasets/" $2}' DATASET_MANIFEST.csv)
```

如果在不支持 Bash process substitution 的环境中使用，可直接逐项运行 `sha256sum datasets/<relative_path>` 并与清单最后一列比较。

## 使用外部数据副本

仓库已包含数据；如果希望改用其他位置的副本，可覆盖数据根目录：

```bash
python scripts/train.py \
  --dataset-key sentence_length_t10 \
  --data-root /absolute/path/to/autoencoder_koopman_core_data \
  --state-family memory --lag 3
```

也可用 `--dataset /absolute/path/to/trajectories.jsonl` 直接训练自定义文件，并通过 `--output-columns` 与 `--target-columns` 指定列。
