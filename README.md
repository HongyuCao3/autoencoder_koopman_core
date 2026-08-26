# Autoencoder--Koopman 核心算法与数据集

这是从 `ml-genctrl` 实验代码中抽出的独立、可运行目录，集中包含：

- 受控 Autoencoder--Koopman 核心实现；
- Markov、Memory 和 Augmented 三种状态定义；
- `joint` 与 `reconstruction_then_ridge` 两种训练方式；
- 6 个标量任务和 2 个多变量任务的原始采集轨迹；
- 自动训练/评估入口、精确续训 checkpoint 和测试。

## 目录结构

```text
autoencoder_koopman_core/
├── src/koopman_ae/core.py       # 状态构造、AE、K/B/c、rollout、诊断
├── scripts/train.py             # 单一训练/评估 CLI
├── configs/datasets.json        # 数据集路径和输入/输出列注册表
├── datasets/                    # 8 份原始轨迹数据（直接纳入仓库）
├── DATASETS.md                  # 数据语义与使用说明
├── DATASET_MANIFEST.csv         # 行数、轨迹数、大小、SHA-256
├── tests/test_core.py           # 两种训练模式与精确续训测试
├── results/                     # 轻量 metrics/Koopman 诊断
└── CODE_DESIGN.md               # 代码级设计文档
```

## 上传到 GitHub

当前目录约 71 MB，最大单文件约 31.7 MiB，可以直接使用普通 Git，无需 Git LFS。GitHub 网页上传的单文件上限为 25 MiB，因此不要使用网页拖拽；请使用命令行。GitHub 对超过 50 MiB 的普通 Git 文件发出警告，并阻止超过 100 MiB 的文件；以后若加入更大的数据，应改用 Git LFS。

建议先创建一个 **Private** 空仓库，因为 JSONL 中保留了原始 prompt、模型生成文本和评分字段。创建仓库时不要预先添加 README、License 或 `.gitignore`，然后执行：

```bash
cd /home/ruimind/code/idea/idea-LLMControl/autoencoder_koopman_core

git init -b main
git config user.name "你的 GitHub 用户名"
git config user.email "你的 GitHub 邮箱"
git add .
git commit -m "Add standalone Autoencoder Koopman core and datasets"

git remote add origin git@github.com:YOUR_GITHUB_USERNAME/autoencoder_koopman_core.git
git push -u origin main
```

如果使用 HTTPS，将 remote 改为：

```bash
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/autoencoder_koopman_core.git
```

GitHub 已不接受账户密码进行 Git 推送；HTTPS 使用 Personal Access Token，或者先配置 SSH key。上传前可用下面的命令确认即将提交的文件和体积：

```bash
git status --short
git count-objects -vH
find datasets -type f -size +50M -print
```

官方参考：[添加本地代码到 GitHub](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)、[GitHub 大文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

## 核心算法

先把轨迹构造成状态：

- Markov：`z_t = [y_t]`；
- Memory-L：`z_t = [y_t, y_(t-1), ..., y_(t-L)]`；
- Augmented-L：在 Memory 状态后加入控制历史，默认 `u_t = r - y_t`。

Autoencoder 和受控 Koopman 动力学为：

```text
xi_t       = Encoder(z_t)
xi_(t+1)   = K xi_t + B r + c
z_hat_t    = Decoder(xi_t)
y_hat_t    = z_hat_t 的前 output_dim 个分量
```

`joint` 同时优化重构误差、解码后一步预测误差、latent 线性误差，以及可选的多步 rollout 误差：

```text
L = lambda_rec * L_rec
  + lambda_pred * L_pred
  + lambda_latent * L_latent
  + lambda_multi * L_multi
```

`reconstruction_then_ridge` 先只训练 AE 重构；固定 encoder 后，用带非正则化截距的 ridge 闭式拟合 `K`、`B`、`c`。当前实验中它通常是更稳健的默认基线，但不保证对所有任务都优于 joint。

## 环境安装

环境和缓存应放在 `/scratch`：

```bash
cd /home/ruimind/code/idea/idea-LLMControl/autoencoder_koopman_core
python3 -m venv /scratch/ruimind/envs/autoencoder-koopman-core
source /scratch/ruimind/envs/autoencoder-koopman-core/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

最低依赖是 Python 3.10、NumPy、pandas 和 PyTorch。CPU 可以运行；较大的 joint/K 扫描建议使用 GPU。

## 快速运行

### 1. Two-stage Memory-L3（推荐起点）

```bash
python scripts/train.py \
  --dataset-key sentence_length_t10 \
  --state-family memory \
  --lag 3 \
  --training-mode reconstruction_then_ridge \
  --latent-dim 16 \
  --epochs 200 \
  --device cpu
```

这里 `lag=3` 表示状态保存 4 个观测：`[y_t, y_(t-1), y_(t-2), y_(t-3)]`。

### 2. Joint Augmented-L3

```bash
python scripts/train.py \
  --dataset-key sentence_length_t10 \
  --state-family augmented \
  --lag 3 \
  --training-mode joint \
  --latent-dim 16 \
  --epochs 400 \
  --lambda-rec 1.0 \
  --lambda-pred 1.0 \
  --lambda-latent 0.1 \
  --lambda-multi 0.1 \
  --multi-step-horizon 3 \
  --device cuda
```

### 3. 多变量 Stage-2

数据注册表会自动选择三维输出和三维目标：

```bash
python scripts/train.py \
  --dataset-key vector_count_stage2_t10 \
  --state-family augmented \
  --lag 3 \
  --training-mode joint \
  --latent-dim 64 \
  --epochs 400 \
  --device cuda
```

## Checkpoint 与精确续训

默认 checkpoint 位于：

```text
/scratch/ruimind/checkpoints/idea-LLMControl/autoencoder_koopman_core/<run-name>/
```

每个完整 checkpoint 保存 encoder、decoder、`K/B/c`、AdamW optimizer、PyTorch/NumPy/Python RNG、CUDA RNG（如有）、DataLoader RNG、epoch 和训练配置。临时目录只有在 `state.pt` 完整写入后才原子重命名，并用 `_COMPLETE` 标记。启动时会自动跳过不完整 checkpoint 并恢复最新完整项。

续训只需把 `--epochs` 调大后再次运行同一命令：

```bash
python scripts/train.py \
  --dataset-key sentence_length_t10 \
  --state-family memory --lag 3 \
  --training-mode joint \
  --latent-dim 16 \
  --epochs 800 \
  --device cuda
```

允许改变总 epoch 数和 CPU/GPU 设备；改变学习率、网络结构、loss 权重、数据文件或状态定义会拒绝复用旧 checkpoint。若确实要从头训练，请改 `--run-name`；`--no-resume` 只允许用于空 checkpoint 目录。

## 输出

轻量结果默认写到：

```text
results/<run-name>/run.json
results/<run-name>/koopman_diagnostics.json
```

`run.json` 包含数据 SHA-256、状态定义、split 规模、one-step/recursive rollout 指标和 checkpoint 路径。`koopman_diagnostics.json` 包含完整 `K/B/c`、特征值实虚部、spectral radius、有限时域 controllability matrix/奇异值/Gramian 诊断。

原始数据已直接放在 `datasets/` 中以便随 Git 仓库上传。训练 checkpoint 仍保留在 `/scratch`，项目目录保存代码、数据、文档、清单和轻量结果。

## 自定义数据

自定义表至少需要：`trajectory_id`、`topic_split`、`turn`、观测列和目标列。每条轨迹的 turn 应从 1 开始且连续；split 必须在轨迹级预先确定，不能把同一轨迹的不同 turn 分到不同 split。

```bash
python scripts/train.py \
  --dataset /absolute/path/custom.jsonl \
  --output-columns y1 y2 \
  --target-columns r1 r2 \
  --state-family augmented --lag 2 \
  --common-seed-turns 3 \
  --training-mode joint
```

支持 `.jsonl`、`.csv` 和 `.parquet`；Parquet 需另行安装 pandas 对应 engine。

## 验证

```bash
pytest -q
```

测试覆盖 joint、two-stage、一步预测、跳过不完整 checkpoint，以及“训练 3 epoch 后精确续训至 6 epoch”和不间断 6 epoch 的数值一致性。

## 数据与适用边界

详细数据说明见 `DATASETS.md`。其中 even/odd 是类别到标量的接口/负对照，sentiment 和 formality 受到 scorer/readout 的限制；这些数据不应被统一表述为强 Koopman 证据。评估保持已有 `topic_split`，默认从共同前缀 turn 1--4 递归预测后续 turn。

核心实现从本工作区 `ml-genctrl/genctrl/operator_validation/model_iii.py` 整理而来，并增加独立 CLI、数据注册表和更严格的 checkpoint 配置/完整性检查。许可证见 `LICENSE`。
