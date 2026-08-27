# <a name="ml-coding"></a> 2. 机器学习/数据编程 :robot:

本页是 ML 编码面试练习的权威索引。建议按类别逐步练习：先独立实现题目，再对照链接中的答案，补充边界测试，并口头说明关键取舍。

题目按面试能力划分，而不是按框架堆叠：

1. **经典机器学习** — 算法、数据预处理、评估指标与训练基础。
2. **语言模型（LM）** — 分词、注意力、解码与序列批处理。
3. **生成式 AI（GenAI）** — 参数高效微调、检索、偏好学习与生成基础组件。
4. **智能体 AI 编程** — 工具、编排、记忆、重试、权限与终止条件。

目录中的每道题都有答案。Python 解答刻意保持为适合面试现场阅读和实现的独立小文件；少数适合可视化推导的经典题继续使用已有 notebook 作为答案。

## 如何使用本章

1. 选择一道题，先明确输入、输出、shape 与前提假设。
2. 不看答案，先完成一个正确的基线实现。
3. 至少测试一个正常场景，以及一个边界或失败场景。
4. 对照链接中的答案，再说明复杂度与生产环境中的取舍。
5. 建议用 15 分钟复习**简单**题，20–35 分钟完成**中等**题，40–60 分钟以上完成**困难**题。

在仓库根目录运行全部 Python 答案测试：

```bash
uv run --with numpy python -m unittest discover -s src/MLC/tests -p "test_*.py"
```

难度颜色：![简单（Easy）](../../../src/MLC/assets/difficulty-easy.svg) ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) ![困难（Hard）](../../../src/MLC/assets/difficulty-hard.svg)

## ML 编码重点题目

### 经典机器学习

这些题目覆盖通用 ML、Applied Scientist 与数据方向面试中最常见的实现模式。

| 题目 | 难度 | 主题标签 | 公司标签 | 答案 | 面试重点 |
| --- | --- | --- | --- | --- | --- |
| 数值稳定的 softmax 与交叉熵 | ![简单（Easy）](../../../src/MLC/assets/difficulty-easy.svg) | `NumPy`、概率、数值稳定性 | Apple、Meta、Google、Amazon | [Python](../../../src/MLC/problems/classic_ml/softmax_cross_entropy.py) | 最大值平移、log-sum-exp、shape、类别索引校验 |
| 使用梯度下降的线性回归 | ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) | 回归、优化、向量化 | — | [Python](../../../src/MLC/problems/classic_ml/linear_regression.py) · [Notebook](./notebooks/linear_regression_md.ipynb) | MSE 梯度、偏置项、特征尺度、收敛性 |
| 使用梯度下降的逻辑回归 | ![困难（Hard）](../../../src/MLC/assets/difficulty-hard.svg) | 分类、优化、数值稳定性 | Google、Meta、Amazon | [Python](../../../src/MLC/problems/classic_ml/logistic_regression.py) · [Notebook](./notebooks/logistic_regression_md.ipynb) | 稳定 sigmoid、二元标签、梯度、决策阈值 |
| k 最近邻 | ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) | 距离、向量化、分类 | Uber、LinkedIn、Meta | [Python](../../../src/MLC/problems/classic_ml/knn.py) · [Notebook](./notebooks/k_nearest_neighbors.ipynb) | 成对距离、top-k、平票处理、时间与内存开销 |
| k-means 聚类 | ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) | 聚类、无监督学习、收敛 | Uber、LinkedIn、Google、Amazon | [Python](../../../src/MLC/problems/classic_ml/kmeans.py) · [Notebook](./notebooks/k_means_2.ipynb) | 初始化、分配/更新步骤、空簇处理 |
| 决策树划分 | ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) | 决策树、Gini 不纯度、搜索 | — | [Python](../../../src/MLC/problems/classic_ml/decision_tree_split.py) · [Notebook](./notebooks/decision_tree.ipynb) | 候选阈值、加权不纯度、停止条件 |
| 主成分分析（PCA） | ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) | 线性代数、降维、SVD | — | [Python](../../../src/MLC/problems/classic_ml/pca.py) | 中心化、主成分排序、解释方差 |
| 二维卷积 | ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) | 深度学习、计算机视觉、张量 shape | — | [Python](../../../src/MLC/problems/classic_ml/conv2d.py) · [Notebook](./notebooks/convolution.ipynb) | 输出 shape、步幅、互相关与卷积的区别 |
| 二分类指标与 ROC-AUC | ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) | 评估、排序、类别不平衡 | — | [Python](../../../src/MLC/problems/classic_ml/binary_metrics_roc_auc.py) | 分母为零、分数并列、基于排序的解释 |
| 蓄水池抽样 | ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) | 流式处理、概率、抽样 | — | [Python](../../../src/MLC/problems/classic_ml/reservoir_sampling.py) | 流长度未知、均匀入选概率、O(k) 内存 |
| 线性 SVM 与 hinge loss | ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) | 间隔方法、优化、分类 | — | [Notebook](./notebooks/svm.ipynb) | hinge loss、间隔违规、正则化 |
| 感知机学习规则 | ![简单（Easy）](../../../src/MLC/assets/difficulty-easy.svg) | 在线学习、线性分类 | — | [Notebook](./notebooks/perceptron.ipynb) | 更新条件、收敛假设、决策边界 |
| 前馈神经网络与反向传播 | ![困难（Hard）](../../../src/MLC/assets/difficulty-hard.svg) | 神经网络、链式法则、梯度 | — | [Notebook](./notebooks/feedforward.ipynb) | 前向缓存、梯度流、参数更新、shape 校验 |
| 多分类指标与宏平均 | ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) | 评估、混淆矩阵、平均策略 | — | [Python](../../../src/MLC/problems/classic_ml/multiclass_metrics.py) | 按类别统计、宏平均与微平均、缺失类别 |
| 用于文本分类的多项式朴素贝叶斯 | ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) | NLP、概率、平滑 | — | [Python](../../../src/MLC/problems/classic_ml/multinomial_naive_bayes.py) | 对数空间、拉普拉斯平滑、未见 token、条件独立假设 |
| 推荐系统中的矩阵分解 | ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) | 推荐、embedding、SGD | — | [Python](../../../src/MLC/problems/classic_ml/matrix_factorization.py) | 只拟合已观测值、正则化、隐向量维度、冷启动 |
| 梯度提升中的残差拟合步骤 | ![困难（Hard）](../../../src/MLC/assets/difficulty-hard.svg) | 集成学习、残差、回归树 | — | [Python](../../../src/MLC/problems/classic_ml/gradient_boosting_step.py) | 负梯度、弱学习器、learning-rate shrinkage |
| 无数据泄漏的训练/验证/测试集划分 | ![简单（Easy）](../../../src/MLC/assets/difficulty-easy.svg) | 数据划分、可复现性、分层抽样 | — | [Python](../../../src/MLC/problems/classic_ml/dataset_split.py) | 索引互斥、类别比例、仅在训练集拟合 |
| 使用训练集统计量做特征标准化 | ![简单（Easy）](../../../src/MLC/assets/difficulty-easy.svg) | 预处理、数据泄漏、数值稳定性 | — | [Python](../../../src/MLC/problems/classic_ml/feature_standardization.py) | fit/transform 分离、常量特征、shape 校验 |
| 处理缺失值与未见类别 | ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) | 预处理、缺失值填补、类别编码 | — | [Python](../../../src/MLC/problems/classic_ml/categorical_preprocessing.py) | 仅使用训练集状态、未知类别桶、整列缺失 |
| 均匀、加权与分层抽样 | ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) | 抽样、类别不平衡、可复现性 | — | [Python](../../../src/MLC/problems/classic_ml/sampling_strategies.py) | 有放回/无放回、权重归一化、类别数量保证 |
| 样本加权 loss 与流式指标 | ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) | 评估、流式计算、分布式聚合 | — | [Python](../../../src/MLC/problems/classic_ml/streaming_weighted_mean.py) | 按样本数加权、可合并状态、空状态 |

### 语言模型（LM）

这些题目从传统文本特征逐步过渡到现代语言模型的 token、注意力与解码机制。

| 题目 | 难度 | 主题标签 | 公司标签 | 答案 | 面试重点 |
| --- | --- | --- | --- | --- | --- |
| TF-IDF | ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) | NLP、稀疏特征、信息检索 | — | [Python](../../../src/MLC/problems/language_models/tfidf.py) | token 计数、文档频率、平滑、词表顺序 |
| 缩放点积注意力 | ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) | 注意力、Transformer、mask | — | [Python](../../../src/MLC/problems/language_models/scaled_dot_product_attention.py) | Q/K/V shape、1/sqrt(d_k)、在稳定 softmax 前应用 mask |
| 学习并应用 BPE | ![简单（Easy）](../../../src/MLC/assets/difficulty-easy.svg) | 分词、BPE、词表 | — | [Python](../../../src/MLC/problems/language_models/byte_pair_encoding.py) | 相邻 pair 计数、确定性合并、未见单词编码 |
| 构造因果注意力 mask | ![简单（Easy）](../../../src/MLC/assets/difficulty-easy.svg) | Transformer、mask、自回归解码 | — | [Python](../../../src/MLC/problems/language_models/causal_attention_mask.py) | query/key 语义、padding、布尔 mask 约定 |
| 正弦位置编码 | ![简单（Easy）](../../../src/MLC/assets/difficulty-easy.svg) | Transformer、位置、向量化 | — | [Python](../../../src/MLC/problems/language_models/sinusoidal_position_encoding.py) | sin/cos 通道交替、奇数维宽度、shape |
| Temperature、top-k 与 top-p 采样 | ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) | 解码、概率、生成 | Anthropic、OpenAI、DeepMind | [Python](../../../src/MLC/problems/language_models/top_k_top_p_sampling.py) | 过滤顺序、重新归一化、确定性测试 |
| 只追加的 KV cache | ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) | 推理、注意力、延迟、内存 | Anthropic、OpenAI、Meta、Perplexity | [Python](../../../src/MLC/problems/language_models/kv_cache.py) | cache shape、容量、prefill 与 decode、内存增长 |
| 对变长 token 序列做 padding | ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) | batching、padding、mask | — | [Python](../../../src/MLC/problems/language_models/padded_batches.py) | padding ID、有效 token mask、空序列、batch shape |

### 生成式 AI（GenAI）

这些题目聚焦微调、检索、对齐与生成流程中常见的基础组件。

| 题目 | 难度 | 主题标签 | 公司标签 | 答案 | 面试重点 |
| --- | --- | --- | --- | --- | --- |
| 线性层的 LoRA 更新 | ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) | 参数高效微调、低秩、线性代数 | Meta、Google、Anthropic、OpenAI、Databricks | [Python](../../../src/MLC/problems/genai/lora_linear.py) | 矩阵 shape、alpha/r 缩放、参数节省、权重合并 |
| 对称式对比学习 loss | ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) | embedding、InfoNCE、多模态 | OpenAI、Anthropic、DeepMind、Midjourney | [Python](../../../src/MLC/problems/genai/contrastive_loss.py) | 归一化、temperature、batch 内负样本、双向 loss |
| 面向 RAG 的精确余弦 top-k 检索 | ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) | RAG、embedding、检索、排序 | — | [Python](../../../src/MLC/problems/genai/rag_retrieval.py) | 归一化、top-k 顺序、零向量、生产级 ANN 取舍 |
| 直接偏好优化（DPO）loss | ![困难（Hard）](../../../src/MLC/assets/difficulty-hard.svg) | 对齐、偏好学习、微调 | Anthropic、OpenAI、DeepMind、Meta | [Python](../../../src/MLC/problems/genai/dpo_loss.py) | 策略/参考模型 log-prob 差值、beta、稳定 log-sigmoid |
| Classifier-free guidance 合成 | ![简单（Easy）](../../../src/MLC/assets/difficulty-easy.svg) | Diffusion、guidance、生成 | — | [Python](../../../src/MLC/problems/genai/classifier_free_guidance.py) | 条件/无条件预测、guidance scale 的取舍 |

### 智能体 AI 编程

这些题目考察模型外围的确定性控制层。参考答案把权限、预算、重试与终止条件放在代码中，而不是仅依赖 prompt。

| 题目 | 难度 | 主题标签 | 公司标签 | 答案 | 面试重点 |
| --- | --- | --- | --- | --- | --- |
| 校验 JSON 工具参数 | ![简单（Easy）](../../../src/MLC/assets/difficulty-easy.svg) | 工具、schema、校验 | — | [Python](../../../src/MLC/problems/agentic_ai/tool_argument_validation.py) | 必填字段、类型、枚举值、额外参数 |
| 带权限控制的工具注册表 | ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) | 工具、权限、function calling | — | [Python](../../../src/MLC/problems/agentic_ai/tool_registry.py) | 注册、函数签名绑定、副作用审批边界 |
| Planner/Executor 边界 | ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) | 规划、编排、trace | — | [Python](../../../src/MLC/problems/agentic_ai/planner_executor.py) | 有界计划、结构化结果、部分失败 |
| 带幂等保护的重试 | ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) | 可靠性、重试、副作用 | — | [Python](../../../src/MLC/problems/agentic_ai/retry_with_idempotency.py) | 瞬时错误、指数退避、避免重复执行 |
| 滑动窗口式会话记忆 | ![中等（Medium）](../../../src/MLC/assets/difficulty-medium.svg) | 记忆、上下文窗口、token 预算 | — | [Python](../../../src/MLC/problems/agentic_ai/sliding_window_memory.py) | 固定系统指令、保留最近历史、淘汰与摘要的取舍 |
| 有界的智能体工具调用循环 | ![困难（Hard）](../../../src/MLC/assets/difficulty-hard.svg) | 智能体、工具、终止、安全 | — | [Python](../../../src/MLC/problems/agentic_ai/bounded_agent_loop.py) | step 预算、重复调用检测、未知工具、可追踪性 |

## 扩展练习

- [PyTorch ML 编码题库](./pytorch-ml-coding.md) 是按框架组织的扩展练习；上面的四张表才是本章的权威题目索引。
- [旧版 notebooks](./notebooks/) 继续用于可视化与探索性练习。权威目录只链接当前仍作为答案或补充材料使用的 notebook。
- [Agentic AI Systems 面试准备](https://github.com/alirezadir/Agentic-AI-Systems/tree/main/06_interview_prep) 提供与本页智能体编程题互补的系统设计内容。

## 难度与公司标签规则

- **简单（Easy）：** 聚焦一个核心操作，通常约 15 分钟。
- **中等（Medium）：** 包含多个步骤或重要边界情况，通常需要 20–35 分钟。
- **困难（Hard）：** 涉及多个组件、高级调试或系统级取舍，通常需要 40–60 分钟以上。

难度标签是针对完整题目范围做出的编辑判断。可精确匹配的题目参考 [TorchLeet](https://github.com/Exorust/TorchLeet) 与 [Deep-ML](https://www.deep-ml.com/problems) 校准；其余题目沿用同一套基于范围的标准。只有当参考来源将某家公司与同一道实现题明确关联时，才会添加公司标签。这些标签只是历史上的备考信号，并不代表该公司当前面试流程。本仓库仅使用元数据，不复制第三方题目描述或答案。

## 面试中应该说明什么

1. 编码前先明确 shape、dtype、前提假设与预期输出。
2. 先完成正确的基线实现，再优化真正的瓶颈。
3. 讨论时间与空间复杂度，包括大型中间矩阵与模型状态。
4. 处理数值稳定性、空输入、非法标签、平票与权限边界。
5. 编写一个正常场景测试，并至少覆盖一个失败或边界场景。
6. 说明面对大规模数据、GPU、分布式执行或生产服务时，实现需要如何调整。
