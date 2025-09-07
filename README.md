# Gilgamesh-AgeQuest
**Predicting biological age from DNA methylation patterns using machine learning.**

## Overview
Gilgamesh-AgeQuest is an end-to-end machine learning pipeline for predicting biological age from DNA methylation (methylome) data, leveraging CpG-site correlations and dysregulation signals through cosine dissimilarity to a young, healthy referent. Building on my PhD research in computational biology, this project improves upon traditional epigenetic clocks by integrating novel dysregulation-based features, showcasing robust feature engineering, data preprocessing, and model training.

## Background

DNA methylation patterns are key biomarkers for biological aging, but conventional age-prediction models often overlook geometric signals of dysregulation in high-dimensional methylome space. This project defines a composite youth referent vector by averaging methylome vectors from young individuals. Each sample’s cosine dissimilarity to the referent is then computed, quantifying angular deviation and loss of colinearity with the youthful baseline. This metric captures the projection deficit of a sample vector relative to the referent direction, providing a rigorous measure of dysregulation. Embedding this angular feature into downstream models improves robustness and interpretability, with applications in aging research, disease risk stratification, and personalized medicine.

## Key Features

- **Data Preprocessing:** Standardizes and harmonizes high-dimensional methylome data, including quality control and normalization.  
- **Feature Engineering:** Constructs a composite youth referent vector and computes cosine dissimilarity for each sample as a dysregulation feature.  
- **Deep Learning Models:** Implements neural architectures (e.g., autoencoders, feedforward networks) for robust biological-age prediction, integrating dysregulation metrics directly into the feature space.  
- **Reproducibility:** Provides a Makefile-driven workflow and configuration management, with planned Docker support for full pipeline portability.  

## Installation

1. Clone the repository:  
    ```bash
    git clone https://github.com/SkinnerCM/gilgamesh-agequest
    cd gilgamesh-agequest

2. Create a virtual environment (Python 3.8+ recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: venv\Scripts\activate

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    pip install -e . # Installs the package locally

## Usage

🚧 Work in progress — pipeline entrypoints and example notebooks will be added in a future update.  


## Project Organization

````text
├── LICENSE
├── Makefile           <- Makefile with commands like `make data`, `make preprocess`, `make noise`, or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed (e.g., correlation, noise outputs).
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-cs-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- Makes project pip installable (`pip install -e .`) so `src` can be imported
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes `src` a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   ├── build_features.py
│   │   └── build_noise.py
│   │
│   ├── pipeline       <- CLI entrypoints and orchestration
│   │   ├── preprocess.py         <- Compute CpG–age correlations
│   │   └── run_build_noise.py    <- Build noise residuals from correlations
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   ├── train_model.py
│   │   └── predict_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results-oriented visualizations
│       └── visualize.py
│
└── tox.ini            <- Tox file with settings for running tests; see tox.readthedocs.io
```            
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

````
