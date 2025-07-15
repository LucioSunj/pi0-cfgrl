#!/bin/bash

# ==============================================================================
# Bash 脚本: 训练 OpenPI 模型（同时在终端打印并记录日志）
#
# 功能:
#   - 双重输出: 使用 'tee' 命令将所有输出同时打印到终端并写入日志文件。
#   - 日志管理: 自动创建带时间戳的日志文件，避免覆盖。
#   - 健壮性设计: 任何命令失败时立即退出 (set -e)。
# ==============================================================================

# --- 脚本设置 ---
# set -e: 当脚本中的任何命令返回非零退出码（表示错误）时，立即退出脚本。
# set -o pipefail: 如果管道中的任何命令失败（例如 cmd1 | tee cmd2），则整个管道的退出码为非零。
set -eo pipefail

# --- 日志文件设置 ---
LOG_DIR="training_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_run_${TIMESTAMP}.log"

# --- 脚本主体通过 tee 同时输出到终端和日志文件 ---
# 使用 {...} 2>&1 | tee "$LOG_FILE" 的结构
# 这会将大括号内所有命令的标准输出 (stdout) 和标准错误 (stderr)
# 都通过管道传递给 'tee' 命令。
# 'tee' 会将接收到的内容同时打印到终端并写入指定的日志文件。
{
    echo "========================================================"
    echo "--- 脚本开始执行: $(date) ---"
    echo "--- 所有日志将同时打印并在以下位置存档: $(pwd)/$LOG_FILE ---"
    echo "========================================================"
    echo

    echo "--- 步骤 1: 设置 Weights & Biases (W&B) 为离线模式 ---"
    export WANDB_MODE=offline
    export WANDB_OFFLINE=true
    echo "✅ W&B 已成功设置为离线模式。"
    echo

    echo "--- 步骤 2: 初始化并激活 Conda 环境 ---"
    CONDA_INIT_SCRIPT="/opt/conda/etc/profile.d/conda.sh"
    if [ -f "$CONDA_INIT_SCRIPT" ]; then
        # shellcheck disable=SC1090
        source "$CONDA_INIT_SCRIPT"
        echo "✅ Conda 初始化脚本加载成功。"
    else
        echo "❌ 错误: 未找到 Conda 初始化脚本 at '$CONDA_INIT_SCRIPT'。请检查路径。"
        exit 1
    fi
    conda activate pi
    echo "✅ Conda 环境 'pi' 已激活。"
    echo "   当前使用的 Python: $(which python)"
    echo

    echo "--- 步骤 3: 切换到项目工作目录 ---"
    PROJECT_DIR="/zhaohan/ZJH/openpi"
    if [ -d "$PROJECT_DIR" ]; then
        cd "$PROJECT_DIR"
        echo "✅ 已成功切换到项目目录: $(pwd)"
    else
        echo "❌ 错误: 项目目录 '$PROJECT_DIR' 不存在。"
        exit 1
    fi
    echo

    echo "--- 步骤 4: 设置 JAX 显存分配策略 ---"
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
    echo "✅ JAX 显存分配比例已设置为 95%。"
    echo

    echo "--- 步骤 5: 运行训练脚本 ---"
    echo "🚀 即将执行: uv run python scripts/train.py pi0_libero --exp-name=my_pi0_libero_train --overwrite"
    # uv run python scripts/train.py pi0_libero_low_mem_finetune \
    uv run python scripts/train.py pi0_libero \
        --exp-name=my_pi0_libero_train_full_final \
        --overwrite
    echo

    echo "========================================================"
    echo "🎉 训练脚本执行完毕 ---"
    echo "--- 脚本结束: $(date) ---"
    echo "--- 完整的执行日志已保存在: $(pwd)/$LOG_FILE ---"
    echo "========================================================"

} 2>&1 | tee "$LOG_FILE"
