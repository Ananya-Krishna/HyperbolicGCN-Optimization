# Dynamic Curvature Optimization for Hyperbolic GCNs

This repository contains the implementation of **Dynamic Curvature Optimization for Hyperbolic GCNs**, a method to dynamically optimize a single global curvature parameter for Hyperbolic Graph Convolutional Networks (HGCNs). Our method enables interpretable and efficient embeddings of hierarchical graph data by treating curvature as a trainable parameter.

---

## Repository Overview

- **`HGCNs_all_datasets_and_visualizations.ipynb`**: Experiments and visualizations for various datasets.
- **`compute_hyperbolicity.ipynb`**: Code to compute Gromov δ-hyperbolicity of graphs.
- **`requirements.txt`**: Dependencies required to run the project.
- **`Full HGCN Experiments Data (3 runs each).pdf`**: Results from experimental runs.
- **`README.md`**: This document.

## Datasets

### Download
Datasets used in this project can be downloaded from the following link:
[Dataset Download](https://drive.google.com/drive/folders/1t3ZuPY_u0DZntSMGJCAdazrBtPOxoRzh?usp=sharing)

### Description
- **WordNet**: Lexical database with hierarchical relationships.
- **PubMed**: Biomedical literature citations.
- **FB15K**: Subset of the Freebase knowledge graph.
- **Diseases**: Hierarchical relationships of diseases and their properties.

---

## Requirements

To install all dependencies, run:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `torch`
- `geoopt`
- `networkx`
- `numpy`
- `scipy`
- `matplotlib`

---

## Usage

### Step 1: Clone the Repository

```bash
git clone https://github.com/ajkohl/HyperbolicGCN-Optimization.git
cd HyperbolicGCN-Optimization
```

### Step 2: Prepare the Dataset
Download and place datasets in the appropriate directory. Follow instructions in the `.ipynb` files for preprocessing.

### Step 3: Running Experiments

#### Compute Hyperbolicity
Run the `compute_hyperbolicity.ipynb` notebook to compute δ-hyperbolicity for graphs:

```bash
jupyter notebook compute_hyperbolicity.ipynb
```

#### Training HGCNs
To train HGCNs on all datasets, use the `HGCNs_all_datasets_and_visualizations.ipynb` notebook:

```bash
jupyter notebook HGCNs_all_datasets_and_visualizations.ipynb
```

Configure the dataset (`WordNet`, `FB15K`, `PubMed`, etc.) and params(learning rate, starting curvature, optimizer) in the notebook.

#### Visualize Results
The same notebook includes code for embedding visualizations and curvature analysis.

---

## Reproducing Main Results

### Run Script
To reproduce the main results, follow these steps:

1. Download the datasets.
2. Select and run the corresponding script for your desired dataset in `HGCNs_all_datasets_and_visualizations.ipynb`.
3. Train the model by running the notebook cells following dataset load-in.

### Results Summary
#### Key Findings:
- Single global curvature optimization improves representation efficiency for hierarchical datasets.
- Validation accuracy improves as curvature aligns with the dataset’s intrinsic δ-hyperbolicity.
- Mixed-precision Riemannian Adam optimizer yields stable convergence and improved accuracy.

| Dataset    | Mean Test Accuracy (±Std) | Mean Curvature |
|------------|-----------------------|----------------|
| Diseases   | 91.42% (±0.68%)   | -0.9953        |
| WordNet    | 76.78% (±1.46%)   | -0.0186        |
| PubMed     | 86.11% (±0.21%)   | -0.8188        |
| FB15K      | 94.34% (±0.03%)   | -0.1320        |

---

## Summary

Hyperbolic Graph Convolutional Networks (HGCNs) effectively represent hierarchical graph data. Our dynamic curvature optimization framework simplifies curvature modeling while enhancing interpretability and performance. The method leverages a single, globally optimized curvature parameter to improve node classification and link prediction tasks.

**Highlights:**
- Improved accuracy over baseline Euclidean GCNs.
- Direct relationship between curvature and δ-hyperbolicity.
- Visual insights into curvature-convergence dynamics.

---

## Citation

If you find this work useful, please cite:

```
@article{krishna2024dynamiccurvature,
  title={Dynamic Curvature Optimization for Hyperbolic GCNs},
  author={Krishna, Ananya and Kohli, Arjan},
  journal={Yale University},
  year={2024}
}
```

---

## License
This project is licensed under the MIT License.

