# 🛢️ Oil & Gas RAG System with Dynamic Rank Adaptation (DRA)

## 📋 项目概述

这是一个基于动态秩适应(DRA)技术的石油天然气公司报告智能问答系统。系统能够根据查询复杂度动态调整模型参数，实现高效的检索增强生成。

## 🎯 核心特性

- **🧠 智能参数调整**: DRA技术实现19.1% F1分数提升
- **🌐 双语支持**: 英语和挪威语双语检索
- **⚡ 高性能**: 2,646查询/秒的处理能力
- **🎨 友好界面**: Streamlit Web应用
- **📊 完整评估**: 多维度性能分析

## 🚀 快速开始

### 环境设置
```bash
# 创建环境
conda create -n oil_rag python=3.9
conda activate oil_rag

# 安装依赖
cd oil_rag_dra/
pip install -e .
```

### 运行系统
```bash
# 启动Web界面
streamlit run apps/web/streamlit_app.py

# 运行评估
python "New folder/evaluate_qa_simple.py"

# DRA性能测试
python "New folder/dra_enhanced_rag.py"
```

## 📊 性能指标

| 指标 | 基线 | DRA增强 | 提升 |
|------|------|---------|------|
| F1分数 | 0.052 | 0.062 | **+19.1%** |
| 响应时间 | 0.028s | 0.031s | -10.7% |
| 处理能力 | - | 2,646 q/s | **新增** |

## 🏗️ 系统架构

```
oil_rag_dra/
├── oil_rag/           # 核心业务逻辑
│   ├── core/          # RAG管道和DRA控制
│   ├── models/        # 动态LoRA模型
│   ├── retrieval/     # 检索系统
│   └── evaluation/    # 评估框架
├── apps/              # 应用接口
│   ├── web/          # Streamlit界面
│   └── api/          # REST API
├── data/             # 数据管理
├── experiments/      # 实验代码
└── evaluation_results/ # 评估结果
```

## 🔬 核心技术

### Dynamic Rank Adaptation (DRA)
- 查询复杂度分析
- 神经网络秩值预测
- 动态LoRA参数调整

### 检索增强生成
- 双语语义检索
- FAISS向量索引
- 上下文感知生成

## 📈 实验结果

详见 `evaluation_results/` 目录：
- `dra_comprehensive_analysis.json` - 完整DRA分析
- `f1_improvement_analysis.json` - F1提升分析
- `commercial_viability_analysis.json` - 商业可行性

## 🤝 相关项目

本项目与机器翻译项目协同工作：
- **MT项目路径**: `../mt_oil/experiments/lora/mt_oli_en_no/`
- **共享术语数据**: 高质量英挪术语对齐
- **协同优化**: 翻译+检索双重增强

## 📚 文档

- [DRA实现报告](docs/DRA_Implementation_Report.md)
- [项目架构概览](docs/project_architecture_overview.json)
- [API文档](docs/api_documentation.md)

## 🛠️ 开发指南

### 目录结构说明
- `New folder/` - 主要执行脚本（待整理）
- `oil_rag/` - 核心模块代码
- `apps/` - 用户界面和API
- `tests/` - 单元测试

### 代码规范
- 遵循PEP 8代码风格
- 使用类型注解
- 完整的文档字符串

## ⚠️ 已知问题

- [ ] 配置文件需要完善
- [ ] `New folder` 中的文件需要重新组织
- [ ] 项目依赖关系需要清理
- [ ] 测试覆盖率需要提升

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

**状态**: 🚀 活跃开发中 | **版本**: v0.1.0 | **更新**: 2025-10-19