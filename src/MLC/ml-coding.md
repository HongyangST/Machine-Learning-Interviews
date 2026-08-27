# <a name="ml-coding"></a> 2. ML/Data Coding :robot:

ML coding rounds vary by company. Some focus on implementing classical algorithms from scratch, while others test practical Python and PyTorch skills such as tensor operations, preprocessing, metrics, and training loops. In either format, interviewers evaluate correctness, numerical stability, code quality, edge-case handling, complexity, and your ability to explain design choices.

## How to use this chapter

- Open the answer linked from each question; every canonical problem has its own executable Python file under [`problems/`](./problems/).
- Run [`tests/test_problem_answers.py`](./tests/test_problem_answers.py) to verify the implementations and study useful edge cases.
- Use the older [notebooks](./notebooks/) as supplementary, exploratory material. Some predate the canonical solutions and may be less complete.
- Practice writing each priority problem without looking at the reference, then compare correctness, complexity, and edge-case handling.

## Difficulty and company tags

Every coding question uses a LeetCode-style difficulty label:

- **Easy:** Usually one core operation, limited state, and a focused implementation that should take about 15 minutes.
- **Medium:** Multiple steps or non-trivial edge cases; a strong interview implementation usually takes 20–35 minutes.
- **Hard:** A 40–60+ minute problem involving several interacting components, advanced debugging, or systems-level trade-offs.

Difficulty is an editorial judgment about the complete prompt in this repository, not a claim that every source rates an equivalent problem identically. Exact matches were calibrated against [TorchLeet](https://github.com/Exorust/TorchLeet) and [Deep-ML](https://www.deep-ml.com/problems); unmatched prompts were rated with the same rubric. When sources disagree, the scope and required edge cases of this repository’s prompt determine the final label.

Company tags are added only when a reference associates a company with the same implementation problem. They are historical preparation signals, not a guarantee that the company currently asks the question. Only metadata facts are used; no third-party problem statements or solutions are copied.

Run the reference tests from the repository root:

```bash
uv run --with numpy python -m unittest discover -s src/MLC/tests -p "test_*.py"
```

## PyTorch ML Coding

Modern ML coding interviews may test practical PyTorch skills rather than only asking candidates to implement algorithms from scratch. The [PyTorch ML Coding Problems](./pytorch-ml-coding.md) guide includes:

- A 60-minute mock interview covering tensors, preprocessing, metrics, training/evaluation loops, and debugging
- Python utility and clean-code questions for ML workflows
- Tensor operations, datasets, batching, device handling, and autograd
- Training, optimization, mixed precision, checkpointing, and reproducibility
- Testing, debugging, deployment, and advanced PyTorch questions
- Coding challenges and concise reference responses

## Priority ML coding problems

The following set combines classic questions that remain common with modern primitives increasingly expected in AI/ML interviews.

| Problem | Difficulty | Company tags | Canonical solution | Supplemental notebook | What a strong solution should cover |
| --- | --- | --- | --- | --- | --- |
| Numerically stable softmax and cross-entropy | Easy | Apple, Meta, Google, Amazon | [Python answer](./problems/classic_ml/softmax_cross_entropy.py) | — | Max subtraction, log-sum-exp, shapes, class-index validation |
| Linear regression with gradient descent | Medium | — | [Python answer](./problems/classic_ml/linear_regression.py) | [Linear regression](./notebooks/linear_regression_md.ipynb) | Vectorized gradients, bias, MSE scaling, convergence |
| Logistic regression with gradient descent | Hard | Google, Meta, Amazon | [Python answer](./problems/classic_ml/logistic_regression.py) | [Logistic regression](./notebooks/logistic_regression_md.ipynb) | Stable sigmoid, binary cross-entropy gradient, thresholds |
| k-nearest neighbors | Medium | Uber, LinkedIn, Meta | [Python answer](./problems/classic_ml/knn.py) | [k-NN](./notebooks/k_nearest_neighbors.ipynb) | Pairwise distances, top-k selection, ties, complexity |
| k-means clustering | Medium | Uber, LinkedIn, Google, Amazon | [Python answer](./problems/classic_ml/kmeans.py) | [k-means](./notebooks/k_means_2.ipynb) | Initialization, vectorized assignment, convergence, empty clusters |
| Decision-tree split | Medium | — | [Python answer](./problems/classic_ml/decision_tree_split.py) | [Decision tree](./notebooks/decision_tree.ipynb) | Candidate thresholds, weighted impurity, stopping conditions |
| Principal component analysis | Medium | — | [Python answer](./problems/classic_ml/pca.py) | — | Centering, SVD/eigendecomposition, component ordering, variance |
| 2D convolution | Medium | — | [Python answer](./problems/classic_ml/conv2d.py) | [Convolution](./notebooks/convolution.ipynb) | Output shape, stride, cross-correlation vs convolution |
| Scaled dot-product attention | Medium | — | [Python answer](./problems/language_models/scaled_dot_product_attention.py) | — | Q/K/V shapes, `1/sqrt(d_k)`, masking before stable softmax |
| Binary metrics and ROC-AUC | Medium | — | [Python answer](./problems/classic_ml/binary_metrics_roc_auc.py) | — | Zero denominators, class imbalance, ties, rank interpretation |
| Reservoir sampling | Medium | — | [Python answer](./problems/classic_ml/reservoir_sampling.py) | — | Unknown stream length, uniform probability, O(k) memory |
| TF-IDF | Medium | — | [Python answer](./problems/language_models/tfidf.py) | — | Token counts, document frequency, smoothing, sparse scaling |

Each canonical question now owns one answer file under [`problems/`](./problems/).

## Additional classic algorithms

These are useful follow-up exercises, especially when they match the target team's domain:

- Linear SVM and hinge loss ([notebook](./notebooks/svm.ipynb)) — **Difficulty:** Medium
- Perceptron learning rule ([notebook](./notebooks/perceptron.ipynb)) — **Difficulty:** Easy
- Feedforward neural network and backpropagation ([notebook](./notebooks/feedforward.ipynb)) — **Difficulty:** Hard
- Multiclass or multilabel extensions of metrics and losses — **Difficulty:** Medium
- Naive Bayes for text classification — **Difficulty:** Medium
- Matrix factorization for recommendation systems — **Difficulty:** Medium
- Gradient boosting: explain the training loop and implement a simple residual-fitting step — **Difficulty:** Hard

## Data and sampling questions

- Implement train/validation/test splitting without leakage — **Difficulty:** Easy
- Standardize features using training-only statistics — **Difficulty:** Easy
- Handle missing values and unseen categories consistently — **Difficulty:** Medium
- Implement uniform, stratified, weighted, and reservoir sampling — **Difficulty:** Medium
- Build mini-batches and pad variable-length sequences — **Difficulty:** Medium
- Aggregate sample-weighted losses and streaming metrics correctly — **Difficulty:** Medium

## What to explain during the interview

1. State input shapes, dtypes, assumptions, and expected outputs before coding.
2. Start with a correct baseline, then vectorize or optimize the bottleneck.
3. Discuss time and space complexity, including the cost of pairwise matrices.
4. Handle numerical stability, empty inputs, ties, constant features, and invalid labels.
5. Write small tests for normal cases and at least one failure or boundary case.
6. Explain how the implementation would change for large datasets, GPUs, distributed training, or production libraries.
