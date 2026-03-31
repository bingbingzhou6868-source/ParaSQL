#!/bin/bash
# 下载 Spider 和 BIRD 数据集

set -e

DATA_DIR="./data/raw"
mkdir -p $DATA_DIR

echo "=== 下载 Spider 数据集 ==="
cd $DATA_DIR
if [ ! -d "spider" ]; then
    wget -q "https://drive.google.com/uc?export=download&id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J" -O spider.zip || \
    echo "请手动下载 Spider: https://yale-lily.github.io/spider 并解压到 data/raw/spider/"
    if [ -f spider.zip ]; then
        unzip -q spider.zip -d spider
        rm spider.zip
    fi
fi

echo "=== 下载 BIRD 数据集 ==="
if [ ! -d "bird" ]; then
    echo "请手动下载 BIRD: https://bird-bench.github.io/ 并解压到 data/raw/bird/"
    echo "BIRD 需要注册后下载，访问: https://bird-bench.github.io/"
fi

echo "完成。请确认 data/raw/spider/ 和 data/raw/bird/ 目录存在。"
