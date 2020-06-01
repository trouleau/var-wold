# Variational Wold

## Installation

Install the internal lib `tsvar` using

    pip install -e .


## Experimental results

### Synthetic Experiments

#### Generating experiment parameters

To generate all the parameters for an experiment on synthetic data, use the script

    python script_make_job.py -e <DIRECTORY> -d <NUM DIMENSIONS> -n <NUM EVENTS> -g <NUM GRAPHS> -s <NUM REALIZATIONS>

to prepare an experiment in directory `<DIRECTORY>` with processes in `<NUM DIMENSIONS>` dimensions with `<NUM EVENTS>` training events, on `<NUM GRAPHS>` graphs, with `<NUM REALIZATIONS>` per graph. `<NUM REALIZATIONS>` is used to generate a random seed to simulate the data, to enable reproducing the results.

Some `bash` scripts were written to automatically generate random parameters for the experiments in the paper:

* To prepare the experiment w.r.t. the number of dimensions:

    ```
    bash script_make_experiment_dim_regime.sh
    ```

* To prepare the experiment w.r.t. the amount of training data (with fixed dimension):

    ```
    bash script_make_experiment_data_regime.sh
    ```

#### Running an experiment

To run an experiment, use

    python script_run_many_jobs.py -e <DIRECTORY> -p <NUM PROCESSES> -s <NUM REALIZATIONS> [--no-std-redirect] [--filter-algo <LIST OF ALGORITHMS>]

to run all the experiments in `<DIRECTORY>` in parallel on a pool of `<NUM PROCESSES>` processes, with `<NUM REALIZATIONS>` per graph.
Use `--no-std-redirect` to keep standard outputs in the main shell. By default, stdout/stderr outputs are redirected to files in their respective experiment directories.
USe `--filter-algo` to run only a subset of all the availalble inference algorithms.

### Real Data Experiments

First, to preprocess the raw datasets into point processes for the experiments, use

    python script_build_datasets.py

Then
