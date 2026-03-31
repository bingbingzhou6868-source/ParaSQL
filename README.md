# ParaSQL: Native Parallel SQL Generation for Text2SQL

基于 ParaThinker 并行思维范式，将 SQL 各子句并行生成，最终汇总为完整 SQL。

## 核心思想

将 ParaThinker 的"多条推理路径并行生成"迁移到 Text2SQL：
- 每个 `<clause_i>` token 对应一个 SQL 子句生成器（SELECT / FROM / JOIN / WHERE / GROUP BY / HAVING / ORDER BY / LIMIT）
- 各子句并行独立生成，互不干扰
- `<summary>` 阶段将所有子句合成完整、合法的 SQL

## 项目结构

```
ParaSQL/
├── data/                    # 数据处理
│   ├── download.sh          # 下载 BIRD / Spider 数据集
│   ├── preprocess_spider.py # Spider 数据预处理
│   ├── preprocess_bird.py   # BIRD 数据预处理
│   └── build_sft_data.py    # 构建并行子句 SFT 训练数据
├── model/
│   ├── attention_mask.py    # 两阶段注意力掩码
│   ├── thought_embedding.py # 子句特定位置嵌入
│   └── parasql_model.py     # ParaSQL 模型封装
├── train/
│   ├── sft_trainer.py       # SFT 训练脚本
│   └── train_config.yaml    # 训练配置
├── inference/
│   ├── engine.py            # 并行推理引擎
│   └── sql_assembler.py     # SQL 子句汇总器
├── eval/
│   ├── eval_spider.py       # Spider 评测
│   ├── eval_bird.py         # BIRD 评测
│   └── metrics.py           # EX / EM 指标
├── prompts/
│   ├── parallel_clause.py   # 并行子句生成 prompt
│   └── schema_linker.py     # Schema linking prompt
└── requirements.txt
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载数据集
bash data/download.sh

# 3. 预处理数据
python data/preprocess_spider.py
python data/preprocess_bird.py

# 4. 构建 SFT 训练数据
python data/build_sft_data.py

# 5. 训练
python train/sft_trainer.py --config train/train_config.yaml

# 6. 评测
python eval/eval_spider.py --model_path ./output/parasql
python eval/eval_bird.py   --model_path ./output/parasql
```
