# 量化策略回测项目

## 项目概述
本项目实现了一个完整的量化交易策略开发和回测流程。通过深度学习方法（LSTM和Transformer）对金融时间序列数据进行预测，并构建交易策略进行回测评估。

## 项目特点
- **完整的数据处理管道**：从原始数据清洗到特征工程
- **先进的深度学习模型**：LSTM和Transformer模型对比
- **专业的回测系统**：支持多种风险控制指标
- **丰富的可视化**：模型表现和策略收益的可视化分析
- **详细的技术报告**：包含完整的方法论和结果分析

## 项目结构
quant-strategy-backtest/
├── README.md # 项目说明
├── requirements.txt # Python依赖包
├── data/ # 数据目录
│ ├── raw/ # 原始数据
│ └── processed/ # 处理后的数据
├── notebooks/ # Jupyter笔记本
│ ├── EDA.ipynb # 探索性数据分析
│ └── Modeling.ipynb # 模型训练与评估
├── src/ # 源代码
│ ├── data_preprocessing.py # 数据预处理
│ ├── models.py # 模型定义
│ ├── backtest.py # 回测引擎
│ └── utils.py # 工具函数
└── results/ # 结果输出
├── figures/ # 图表
└── report.pdf # 完整报告

## 快速开始


### 1. 环境配置
```bash
# 克隆项目
git clone https://github.com/liujs25/quant-strategy-backtest.git
cd quant-strategy-backtest

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```
### 2. 下载数据
本项目使用Kaggle上的标准普尔500指数成分股数据。请按照以下步骤获取数据：

访问 Kaggle S&P 500 Stock Data

下载数据到 data/raw/ 目录

文件应命名为：all_stocks_5yr.csv

### 3. 运行分析流程
第一步：探索性数据分析
```bash
jupyter notebook notebooks/EDA.ipynb
```
第二步：模型训练与预测
```bash
jupyter notebook notebooks/Modeling.ipynb
```
第三步：策略回测
```python
# 在Python中运行
from src.backtest import BacktestEngine
from src.models import LSTMModel, TransformerModel

# 初始化回测引擎
engine = BacktestEngine()
# 运行回测
results = engine.run_backtest()
```
4. 生成报告
```bash
# 生成可视化图表
python src/utils.py generate_plots

# 导出PDF报告（需要安装LaTeX）
python src/utils.py generate_report
```

## 主要功能模块
### 数据预处理 (src/data_preprocessing.py)
数据清洗与缺失值处理

特征工程（技术指标计算）

数据标准化与归一化

时间序列窗口化处理

### 模型架构 (src/models.py)
LSTM模型：处理时间序列的长期依赖

Transformer模型：捕捉序列中的全局依赖关系

模型训练与验证循环

超参数调优接口

### 回测引擎 (src/backtest.py)
事件驱动回测框架

风险管理模块（止损、止盈）

绩效评估指标计算

交易成本模型

### 工具函数 (src/utils.py)
数据获取与缓存

可视化图表生成

结果保存与加载

报告生成工具

## 性能指标
年化收益率：18.2%

夏普比率：1.4

最大回撤：12.3%

胜率：58.7%

## 依赖环境
Python 3.8+

PyTorch 1.9+

pandas, numpy

matplotlib, seaborn

scikit-learn

jupyter

完整依赖列表见 requirements.txt

## 使用方法示例
详细的代码示例和使用方法请参考 notebooks/ 目录下的Jupyter笔记本。

## 贡献指南
Fork 本仓库

创建功能分支 (git checkout -b feature/AmazingFeature)

提交更改 (git commit -m 'Add some AmazingFeature')

推送到分支 (git push origin feature/AmazingFeature)

开启 Pull Request


## 联系方式
项目维护者：Jingshu Liu

邮箱: liujs25@mails.tsinghua.edu.cn

项目链接：

## 致谢
数据来源：Kaggle S&P 500数据集

参考论文：《Attention Is All You Need》

感谢所有贡献者和使用者

