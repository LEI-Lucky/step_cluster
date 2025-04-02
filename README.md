# 深度聚类模型

这是一个基于自编码器和聚类层的深度聚类模型实现，用于MNIST数据集的聚类任务。

## 项目结构

```
step_cluster/
├── data/               # 数据目录
├── save_file/         # 模型保存目录
├── experiments/       # 实验结果目录
├── configs/          # 配置文件目录
│   └── config.yaml   # 配置文件
├── logs/             # 日志目录
└── src/              # 源代码目录
    ├── data_loader.py  # 数据加载
    ├── main.py        # 主入口
    ├── train.py       # 训练逻辑
    ├── evaluation.py  # 评估和可视化
    ├── metrics.py     # 评估指标
    ├── modules.py     # 模型定义
    ├── pre_train.py   # 预训练
    └── utils.py       # 工具函数
```

## 安装

1. 克隆项目：
```bash
git clone [项目地址]
cd step_cluster
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 预训练自编码器：
```bash
python -m src.main --mode pretrain
```

2. 训练完整模型：
```bash
python -m src.main --mode train
```

3. 评估模型：
```bash
python -m src.main --mode evaluate
```

4. 运行完整流程：
```bash
python -m src.main --mode all
```

## 配置说明

配置文件 `configs/config.yaml` 包含以下主要部分：

- 模型配置：网络结构、维度等
- 训练参数：学习率、批次大小等
- 数据处理参数：数据加载配置
- 随机种子设置
- 文件路径配置
- 日志配置

## 实验结果

模型训练完成后，可以在 `experiments` 目录下找到：
- t-SNE可视化图
- 混淆矩阵
- 损失曲线
- 评估指标变化曲线

## 评估指标

模型使用以下指标进行评估：
- ARI (调整兰德指数)
- NMI (归一化互信息)
- 轮廓系数
- 聚类准确率

## 注意事项

1. 确保有足够的GPU内存（如果使用GPU）
2. 预训练阶段可能需要较长时间
3. 可以通过修改配置文件调整模型参数
