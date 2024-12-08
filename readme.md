# Second-Order Optimization for Graph Neural Networks with Hyperbolic Embeddings

---

## Project Aim
This project explores advanced optimization techniques for Graph Neural Networks (GNNs) using hyperbolic embeddings. By leveraging the unique properties of hyperbolic space, particularly its ability to represent hierarchical relationships with minimal distortion, we aim to enhance GNN performance. The key contributions include:
- Implementing second-order optimization methods like Riemannian Hessian approximations and quasi-Newton methods (e.g., L-BFGS).
- Adapting curvature as a tunable parameter to further optimize hyperbolic embeddings.

Our goal is to achieve improved efficiency, stability, and performance for GNNs, with a focus on hierarchical structure learning.

---

## Methods

### Why Hyperbolic GNNs?
Hyperbolic Graph Neural Networks (HGCNs) excel at embedding hierarchical relationships with lower distortion compared to Euclidean methods. These methods are more expressive and yield superior results:
- **63.1% error reduction** in ROC AUC for link prediction.
- **47.5% improvement** in F1 score for node classification (2019 NeurIPS HGCN paper).

### Optimization Framework
1. **Poincaré Ball Model**: Chosen for its conformal properties and compatibility with existing frameworks.
2. **Riemannian L-BFGS**:
   - **Initialization**: Parameters are initialized on the manifold.
   - **Forward Pass**: Message passing aggregates embeddings.
   - **Loss Computation**: Calculates task-specific loss (e.g., cross-entropy).
   - **Backward Pass**:
     - Euclidean gradients are projected to the tangent space for Riemannian gradients.
     - Hessian updates using L-BFGS rules.
   - **Parameter Update**: Updates are projected back to the manifold via exponential maps.
   - **Convergence**: Optimization stops when criteria such as gradient norms or loss changes are met.

### Computational Efficiency
- **Geoopt Library**: Extends PyTorch for Riemannian manifold operations.
- **Key Features**:
  - Manifold-aware parameterization.
  - Mini-batch stochastic training for scalability.
  - GPU acceleration for computational efficiency.

---

## Dataset: FB15K
The FB15K dataset, derived from the Freebase knowledge graph, is a benchmark for knowledge graph completion and link prediction. It contains:
- **14,000+ entities** and **1,345 relations** structured as triples (e.g., `head entity, relation, tail entity`).
- Rich hierarchical structures that are well-suited for hyperbolic embeddings.

### Preliminary Analysis:
- **Nodes**: 14,497  
- **Edges**: 246,231  
- **Average Node Degree**: 33.97  

Entities are categorized (e.g., film, location, sports, business) with 3–8 hierarchical levels.

---

## Evaluation Metrics
1. **Node Classification Accuracy**: Validates hierarchical relationship modeling.
2. **Link Prediction Metrics**: Measures precision and recall.
3. **Convergence Rate**: Assesses iterations required for optimization.
4. **Computational Efficiency**: Benchmarks against baseline first-order methods.

---

## References
1. **Nickel, M., & Kiela, D. (2017)**: "Poincaré Embeddings for Learning Hierarchical Representations." [Paper](https://arxiv.org/abs/1705.08039)
2. **Ganea, O.-E., Bécigneul, G., & Hofmann, T. (2018)**: "Hyperbolic Neural Networks." [Paper](https://arxiv.org/abs/1805.09112)
3. **Chami, I., Ying, Z., Ré, C., & Leskovec, J. (2019)**: "Hyperbolic Graph Convolutional Neural Networks." [Paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/0415740eaa4d9decbc8da001d3fd805f-Paper.pdf)

---
