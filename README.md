Token Pair and Neural Embedding Bidirectional Validation Framework
This repository contains the implementation of a bidirectional validation framework for model transformations using token pairs and neural embeddings, as described in our paper "Towards Bidirectional Semantic Validation in Model Transformations Using Neural Embeddings and Token Pairs".
Repository Structure
token-pair-embeddings/
├── bidirectional*validator.py # Core validation framework
├── embedding_generator.py # Neural embedding generation
├── modelset_loader.py # ModelSet dataset loader
├── modelset_adapter.py # Adapter for ModelSet
├── token_pair_adapter.py # Token pair conversion utilities
├── run_embedding_experiments.py # Main experiment script
├── embedding_autoregression_demo.py # Embedding-enhanced autoregression demo
├── visualize_results.py # Visualization utilities
├── paper_figures/ # Generated figures for the paper
│ ├── intent_comparison_chart.pdf
│ ├── similarity_improvement_chart.pdf
│ ├── contribution_chart.pdf
│ └── embedding_autoregression*\*.pdf
└── README.md # This file
Prerequisites

1. Python 3.8 or higher
2. Required Python packages:
3. pip install torch transformers matplotlib numpy pandas scikit-learn networkx
4. The ModelSet dataset (instructions below)
   Getting Started
5. Clone the Repository
   git clone https://github.com/elbachir67/token-pair-embeddings.git
   cd token-pair-embeddings
6. Obtain the ModelSet Dataset
   Download the ModelSet dataset and extract it to a folder named modelset-dataset in the project directory. The ModelSet dataset contains UML and Ecore models used in our experiments.
   You can download a sample of the dataset from https://figshare.com/s/5a6c02fa8ed20782935c?file=24495371.
7. Running the Basic Experiment
   To run the basic experiment comparing token pairs and embedding-enhanced validation:
   python run_embedding_experiments.py --modelset ./modelset-dataset --output results --experiment basic
   This will generate results in the results directory and figures in results/figures.
8. Running the Intent-Aware Comparison Experiment
   To run the intent-aware comparison experiment evaluating translation vs. revision transformations:
   python run_embedding_experiments.py --modelset ./modelset-dataset --output results --experiment compare
9. Running All Experiments
   To run all experiments:
   python run_embedding_experiments.py --modelset ./modelset-dataset --output results --experiment all
10. Running the Embedding-Enhanced Autoregression Demo
    To test the embedding-enhanced autoregression mechanism:
    python embedding_autoregression_demo.py --modelset ./modelset-dataset --output paper_figures
11. Visualizing Results
    To generate visualizations from existing experiment results:
    python visualize_results.py --results results/detailed_comparison.json --output paper_figures
    Reproducing Paper Results
    The key findings of our paper can be reproduced by following these steps:
12. Run the complete experiment suite:
13. python run_embedding_experiments.py --modelset ./modelset-dataset --output results --experiment all
14. Examine the results in results/detailed_comparison.json to verify:
    o Embedding enhancement improves validation for revision transformations (+0.23% on average)
    o Translation transformations maintain consistent performance (-0.35% on average)
    o The highest improvement (0.76%) corresponds to the highest embedding similarity (0.9896)
15. View the generated figures in results/figures/ or generate paper-quality figures:
16. python visualize_results.py --results results/detailed_comparison.json --output paper_figures
    Citation
    If you use this code in your research, please cite our paper:
    @inproceedings{toure2025bidirectional,
    title={Towards Bidirectional Semantic Validation in Model Transformations Using Neural Embeddings and Token Pairs},
    author={Toure, El Hadji Bassirou and [Other Authors]},
    booktitle={Proceedings of the International Conference on Model-Driven Engineering},
    year={2025},
    organization={IEEE}
    }
    Contact
    For questions or issues, please contact elhadjibassirou.toure@ucad.edu.sn.

---

You can adjust the documentation based on your specific repository organization and provide additional details as needed. This setup guide provides clear instructions for readers to test your results and reproduce the key findings of your paper.
For the GitHub link, you can add it prominently at the top of your paper:
\title{Towards Bidirectional Semantic Validation in Model Transformations Using Neural Embeddings and Token Pairs\\
\small{\textit{Implementation available at: \url{https://github.com/elbachir67/token-pair-embeddings.git}}}}
Or add it to the footnote on the first page:
\footnotetext{Implementation and data available at: \url{https://github.com/elbachir67/token-pair-embeddings.git}}
