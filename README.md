# A Variational Inference Approach to Learning Multivariate Wold Processes

The code to run the experiments in the paper is provided here. Below are instructions to run the examples.

## 1. Installation

```
# Create and activate environment
conda create -n env python=3.7 && conda activate env
# Install internal lib
pip install -e .
# Install external lib (baseline algorithm)
git clone https://github.com/flaviovdf/granger-busca.git
cd granger-busca/ && \
    git checkout -b cba79841d16d523bc05004025b6e16691ec074bd && \
    pip install cython && \
    pip install -e .
# Install jupyter to run notebooks
pip install jupyter
```

## 2. Example Notebook

An example notebook is provided in

    example/1-example-notebook.ipynb

to show how to run each of the methods discussed in the paper


## 3. Preprocessing of Real-World Datasets

The code used to preprocess the dataset is detailed in two notebooks

    datasets/1-dataset-email-Eu-core-preprocessing.ipynb
    datasets/2-dataset-memetracker-preprocessing.ipynb
