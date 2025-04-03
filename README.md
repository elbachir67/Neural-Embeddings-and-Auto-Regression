# Token Pair and Neural Embedding Bidirectional Validation Framework

This repository contains the implementation of a bidirectional validation framework for model transformations using token pairs and neural embeddings, as described in our paper "Towards Bidirectional Semantic Validation in Model Transformations Using Neural Embeddings and Token Pairs".

## Repository Structure

```
token-pair-embeddings/
├── bidirectional_validator.py     # Core validation framework
├── embedding_generator.py         # Neural embedding generation
├── modelset_loader.py             # ModelSet dataset loader
├── modelset_adapter.py            # Adapter for ModelSet
├── token_pair_adapter.py          # Token pair conversion utilities
├── run_embedding_experiments.py   # Main experiment script
├── embedding_autoregression_demo.py # Embedding-enhanced autoregression demo
├── visualize_results.py           # Visualization utilities
├── paper_figures/                 # Generated figures for the paper
│   ├── intent_comparison_chart.pdf
│   ├── similarity_improvement_chart.pdf
│   ├── contribution_chart.pdf
│   └── embedding_autoregression_*.pdf
└── README.md                      # This file
```

## Prerequisites

- Python 3.8 or higher
- Required Python packages:
  ```
  pip install torch transformers matplotlib numpy pandas scikit-learn networkx
  ```
- The ModelSet dataset (instructions below)

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/elbachir67/token-pair-embeddings.git
cd token-pair-embeddings
```

### 2. Obtain the ModelSet Dataset

Download the ModelSet dataset and extract it to a folder named `modelset-dataset` in the project directory. The ModelSet dataset contains UML and Ecore models used in our experiments.

You can download a sample of the dataset from [here](https://figshare.com/s/5a6c02fa8ed20782935c?file=24495371).

### 3. Running the Basic Experiment

To run the basic experiment comparing token pairs and embedding-enhanced validation:

```bash
python run_embedding_experiments.py --modelset ./modelset-dataset --output results --experiment basic
```

This will generate results in the `results` directory and figures in `results/figures`.

### 4. Running the Intent-Aware Comparison Experiment

To run the intent-aware comparison experiment evaluating translation vs. revision transformations:

```bash
python run_embedding_experiments.py --modelset ./modelset-dataset --output results --experiment compare
```

### 5. Running All Experiments

To run all experiments:

```bash
python run_embedding_experiments.py --modelset ./modelset-dataset --output results --experiment all
```

### 6. Running the Embedding-Enhanced Autoregression Demo

To test the embedding-enhanced autoregression mechanism:

```bash
python embedding_autoregression_demo.py --modelset ./modelset-dataset --output paper_figures
```

### 7. Visualizing Results

To generate visualizations from existing experiment results:

```bash
python visualize_results.py --results results/detailed_comparison.json --output paper_figures
```

## Reproducing Paper Results

The key findings of our paper can be reproduced by following these steps:

1. Run the complete experiment suite:

   ```bash
   python run_embedding_experiments.py --modelset ./modelset-dataset --output results --experiment all
   ```

2. Examine the results in `results/detailed_comparison.json` to verify:

   - Embedding enhancement improves validation for revision transformations (+0.23% on average)
   - Translation transformations maintain consistent performance (-0.35% on average)
   - The highest improvement (0.76%) corresponds to the highest embedding similarity (0.9896)

3. View the generated figures in `results/figures/` or generate paper-quality figures:
   ```bash
   python visualize_results.py --results results/detailed_comparison.json --output paper_figures
   ```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{bidirectional2025,
  title={Neural Embeddings + Auto Regression for Model Transformation},
  author={},
  booktitle={Proceedings of the International ...},
  year={2025},
  organization={IEEE}
}
```

## Contact

For questions or issues, please contact [ballbass67@gmail.com](mailto:ballbass67@gmail.com).
