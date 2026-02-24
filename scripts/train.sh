#!/bin/bash
# 遇到错误时退出
set -e

# 检查是否提供了参数
if [ $# -ne 1 ]; then
    echo "用法: $0 <config_name>"
    echo "示例: $0 pi05_bridge_rlds_finetune_cot_weight_1"
    exit 1
fi

CONFIG_NAME="$1"

# 进入工作目录（请根据实际情况调整路径，或使用绝对路径）
cd /inspire/hdd/global_user/gongjingjing-25039/zhdai/openpi || { echo "目录 openpi 不存在"; exit 1; }

# 激活 Python 虚拟环境
source .venv/bin/activate

# 退出 Conda 环境（如果有），忽略错误
conda deactivate 2>/dev/null || true

# 设置环境变量
export OPENPI_DATA_HOME="/inspire/hdd/global_user/gongjingjing-25039/zhdai/openpi_cache"
export WANDB_MODE="offline"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.9"

# 运行训练命令，使用传入的配置名称
uv run scripts/train.py "$CONFIG_NAME" \
    --exp-name="$CONFIG_NAME" \
    --overwrite