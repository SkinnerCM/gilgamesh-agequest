Gilgamesh-AgeQuest
==============================

Robust biological-age prediction from methylome patterns and residual noise signals.

Project Organization
------------

 ├── LICENSE
├── Makefile           <- Makefile with commands like `make data`, `make preprocess`, `make noise`, or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed (e.g., correlation, noise outputs).
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
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
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- Makes project pip installable (`pip install -e .`) so `src` can be imported
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes `src` a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   ├── build_features.py
│   │   └── build_noise.py
│   │
│   ├── pipeline       <- CLI entrypoints and orchestration
│   │   ├── preprocess.py         <- Compute CpG–age correlations
│   │   └── run_build_noise.py    <- Build noise residuals from correlations
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   ├── train_model.py
│   │   └── predict_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results-oriented visualizations
│       └── visualize.py
│
└── tox.ini            <- Tox file with settings for running tests; see tox.readthedocs.io
