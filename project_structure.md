# 交大视觉印象数据集 - 图像检索与文字检测项目

## 项目结构
```
project/
├── src/
│   ├── retrieval/          # 图像检索模块
│   │   ├── feature_extractor.py  # 特征提取器
│   │   ├── retrieval_engine.py   # 检索引擎
│   │   └── metrics.py            # 测评指标
│   ├── detection/          # 文字检测模块
│   │   ├── text_detector.py      # 文字检测器
│   │   └── visualization.py      # 可视化工具
│   ├── utils/              # 工具函数
│   │   ├── dataset.py            # 数据集处理
│   │   └── io.py                 # 文件读写
│   └── config.py           # 配置文件
├── data/                   # 数据集软链接
├── models/                 # 预训练模型
├── outputs/                # 输出结果
│   ├── retrieval/          # 检索结果
│   ├── detection/          # 检测结果
│   └── metrics/            # 指标图表
├── scripts/                # 运行脚本
│   ├── run_retrieval.py    # 运行图像检索
│   ├── run_detection.py    # 运行文字检测
│   └── run_evaluation.py   # 运行测评
├── requirements.txt        # 依赖
└── README.md               # 说明文档
```

## 核心功能

### 1. 图像检索
- 使用预训练CNN提取图像特征
- 构建特征索引库
- 支持余弦相似度检索
- 计算P@K指标

### 2. 文字检测
- 使用EAST或CRNN模型
- 检测自然场景文字
- 可视化检测结果

## 运行流程
1. 安装依赖
2. 运行检索脚本构建索引
3. 运行检测脚本训练/推理
4. 运行测评脚本生成指标