# 交大视觉印象数据集 - 图像检索与文字检测项目

## 项目简介

本项目基于「交大视觉印象数据集」实现图像检索算法与文字检测算法，并完成效果测评。

## 数据集结构

```
交大视觉印象数据集2026/
├── image_retrieval/
│   ├── base/              # 检索底库 (7728张)
│   │   └── BJTU/
│   ├── query/             # 查询集 (135张)
│   ├── base_statistics.json
│   └── query_statistics.json
└── object_detection/
    ├── data/              # 文字检测数据 (1494张)
    └── label_statistics.json
```

## 12个地标类别

`["fhy", "jx", "kx", "mh", "nm", "sjz", "sy", "tsg", "ty", "yf", "yk", "zx"]`

## 环境配置

### 安装依赖

```bash
pip install -r requirements.txt
```

### 下载预训练模型

```bash
python scripts/download_models.py
```

## 运行步骤

### 1. 图像检索

```bash
python scripts/run_retrieval.py
```

### 2. 文字检测

```bash
python scripts/run_detection.py
```

### 3. 效果测评

```bash
python scripts/run_evaluation.py
```

## 输出结果

### 检索结果
- `outputs/retrieval/retrieval_results.json` - 检索结果
- `outputs/retrieval/landmark_retrieval_results.json` - 按地标分组的检索结果

### 检测结果
- `outputs/detection/*.png` - 带检测框的图片
- `outputs/detection/detection_results.json` - 检测结果
- `outputs/detection/retrieval_detection_results.json` - 检索+检测结果

### 测评指标
- `outputs/metrics/*.png` - 12个地标对应的P@K曲线图
- `outputs/metrics/precision_metrics.json` - 量化指标数据

## 项目结构

```
project/
├── src/
│   ├── retrieval/          # 图像检索模块
│   │   ├── feature_extractor.py
│   │   ├── retrieval_engine.py
│   │   └── metrics.py
│   ├── detection/          # 文字检测模块
│   │   ├── text_detector.py
│   │   └── visualization.py
│   ├── utils/              # 工具函数
│   │   ├── dataset.py
│   │   └── io.py
│   ├── config.py           # 配置文件
│   └── __init__.py
├── scripts/                # 运行脚本
│   ├── run_retrieval.py
│   ├── run_detection.py
│   ├── run_evaluation.py
│   └── download_models.py
├── models/                 # 预训练模型
├── outputs/                # 输出结果
└── requirements.txt
```

## 核心功能

### 图像检索
- 使用ResNet50预训练模型提取图像特征
- 构建特征索引库
- 支持余弦相似度检索
- 计算P@K指标（K=20,40,60）

### 文字检测
- 使用EAST文本检测模型（或OpenCV MSER作为备选）
- 检测自然场景文字
- 可视化检测结果

## 注意事项

1. 图像检索训练时禁止使用图片的类别信息，仅可使用图像本身的视觉特征
2. 文字检测标注文件为JSON格式，位于object_detection/data目录
3. 建议使用GPU加速以提高运行效率

## License

MIT License