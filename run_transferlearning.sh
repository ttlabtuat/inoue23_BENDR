#!/bin/bash

# 実行モードの指定（debugまたはfull）
MODE=$1

# Pythonスクリプトのパス
PYTHON_SCRIPT="transferlearning.py"

# 共通の引数
COMMON_ARGS="--config configs/transferlearning.yml"

if [ "$MODE" = "debug" ]; then
    echo "デバッグモードで実行します。"
    python $PYTHON_SCRIPT $COMMON_ARGS --debug
else
    python $PYTHON_SCRIPT $COMMON_ARGS
fi
