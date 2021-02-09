import numpy as np
import pandas as pd
import networkx as nx
import pickle
import argparse
import json
import sys
import os

import torch

from tsvar.preprocessing import Dataset
from experiments_utils import (run_mle, run_bbvi, run_vi_fixed_beta, run_vi, run_gb, print_report)


# Set torch parallel threads to 1 to allow for multiple parallel jobs to run
torch.set_num_threads(24)

# Threshold for reports
THRESH = 0.05


def load_dataset(input_path, top):
    # Load the dataset
    print()
    print('Dataset:')
    print('========')
    print()
    print(f"Input file: {input_path}")
    print(f"Top: {top:d}")

    if input_path.endswith('.pk'):
        with open(input_path, 'rb') as f:
            dataset = pickle.load(f)
        assert dataset.top == top, "Mismatch between preprocessed dataset and top value"
    else:
        dataset = Dataset(input_path, top=top, timescale='busca')

    # Print stats about the dataset
    print()
    print(f"Num. of dimensions: {len(dataset.timestamps):,d}")
    print(f"    Num. of events: {sum(map(len, dataset.timestamps)):,d}")
    num_edges = dataset.graph.number_of_edges()
    print(f"               %NZ: {100 * num_edges / (dataset.dim ** 2):.2f}%")
    print()
    print("Stats. of num. of events per dim:")
    num_jumps_per_dim = np.array(list(map(len, dataset.timestamps)))
    print(pd.Series(num_jumps_per_dim).describe())

    print()
    busca_betas = np.array([np.median(np.hstack((ev[0], np.diff(ev))))
                            for ev in dataset.timestamps])
    print('Busca estimators of **beta_j**:')
    print(pd.Series(busca_betas.flatten()).describe())
    print()

    return dataset


def run_inference(dataset, out_fname, algo_filter, prior_dict):

    events = dataset.timestamps
    end_time = dataset.end_time

    param_dict = {
        'baseline': 1 / np.array([ev[0]+1e-5 for ev in events]),
        'adjacency': nx.adjacency_matrix(dataset.graph).toarray(),
        'beta': 1.0 * np.ones((dataset.dim, dataset.dim))
    }

    sim_seed = np.random.randint(0, 2**31 - 1)

    res_dict = {'prior_dict': None}

    if 'mle' in algo_filter:
        print()
        print('Run MLE')
        print('-------')
        res_dict['mle'] = run_mle(events, end_time, param_dict, seed=sim_seed)
        print()
        print('-'*80, flush=True)

    if 'bbvi' in algo_filter:
        print()
        print('Run BBVI')
        print('--------')
        res_dict['bbvi'] = run_bbvi(events, end_time, param_dict, seed=sim_seed)
        print()
        print('-'*80, flush=True)

    if 'vifb' in algo_filter:
        print()
        print('Run VI Beta-Fixed')
        print('-----------------')
        print()
        res_dict['prior_dict'] = prior_dict
        print('Prior:')
        for k, v in prior_dict.items():
            print(f"  - {k}: {v:.2e}")
        print()
        res_dict['vi-fixed-beta'] = run_vi_fixed_beta(
            events, end_time, param_dict, seed=sim_seed, prior_dict=prior_dict)
        print()

        as_po = np.array(res_dict['vi-fixed-beta']['coeffs']['as_po'])
        ar_po = np.array(res_dict['vi-fixed-beta']['coeffs']['ar_po'])
        adj_hat = as_po[1:, :] / ar_po[1:, :]
        print_report(adj_hat=adj_hat, adj_true=param_dict['adjacency'], thresh=THRESH)
        print()
        print('-'*80, flush=True)

    if 'vi' in algo_filter:
        print()
        print('Run VI')
        print('------')
        res_dict['vi'] = run_vi(events, end_time, param_dict, seed=sim_seed, prior_dict=prior_dict)
        print()

        as_po = np.array(res_dict['vi-fixed-beta']['coeffs']['as_po'])
        ar_po = np.array(res_dict['vi-fixed-beta']['coeffs']['ar_po'])
        adj_hat = as_po[1:, :] / ar_po[1:, :]
        print_report(adj_hat=adj_hat, adj_true=param_dict['adjacency'], thresh=THRESH)
        print()
        print('-'*80, flush=True)

    if 'gb' in algo_filter:
        print()
        print('Run GrangerBusca')
        print('----------------')
        res_dict['gb'] = run_gb(events, end_time, param_dict, seed=sim_seed)

        adj_hat = np.array(res_dict['gb']['adjacency'])
        adj_hat = adj_hat / adj_hat.sum(axis=1)[:, None]
        print_report(adj_hat=adj_hat, adj_true=param_dict['adjacency'], thresh=THRESH)
        print()
        print('-'*80, flush=True)

    # Save all results
    with open(out_fname, 'w') as out_f:
        json.dump(res_dict, out_f)

    print()
    print('Job Finished', flush=True)


def run_single_job(args, name_suffix=None):
    # Build output filename
    input_fname = os.path.splitext(os.path.splitext(os.path.split(args.input_path)[1])[0])[0]
    exp_name = f"output-{input_fname}-top{args.top:d}"
    if name_suffix:
        exp_name += f"-{name_suffix}"
    output_fname = os.path.join(args.output_path, f"{exp_name}.json")

    # Build stdout/stderr filenames
    if not args.no_std_redirect:
        stdout = os.path.join(args.output_path, f'stdout-{exp_name}')
        stderr = os.path.join(args.output_path, f'stderr-{exp_name}')
        sys.stdout = open(stdout, 'w')
        sys.stderr = open(stderr, 'w')

    # Build prior dict
    prior_dict = {
        'as_pr': args.as_pr,
        'ar_pr': args.ar_pr
    }

    # Load dataset
    dataset = load_dataset(input_path=args.input_path, top=args.top)

    # Run inference
    run_inference(dataset, output_fname, args.algo_filter, prior_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp', dest='input_path', type=str,
                        required=True, help="Input dataset path")
    parser.add_argument('--top', dest='top', type=int,
                        required=True, help="Number of top nodes to use")
    parser.add_argument('-o', '--output_path', dest='output_path', type=str,
                        required=True, help="Output path to save results")
    parser.add_argument('--filter-algo', nargs='+', dest='algo_filter', type=str,
                        required=False, help='Filter algorithms to work with')
    parser.add_argument('--no-std-redirect', dest='no_std_redirect',
                        action="store_true", help="Do not redirect stdout/stderr")

    parser.add_argument('--as_pr', dest='as_pr', type=float, required=False,
                        default=0.1, help="`as_pr` for vi/vifb methods")
    parser.add_argument('--ar_pr', dest='ar_pr', type=float, required=False,
                        default=1.0, help="`as_pr` for vi/vifb methods")
    parser.add_argument('--bs_pr', dest='bs_pr', type=float, required=False,
                        default=0.1, help="`as_pr` for vi/vifb methods")
    parser.add_argument('--br_pr', dest='br_pr', type=float, required=False,
                        default=1.0, help="`as_pr` for vi/vifb methods")


    args = parser.parse_args()

    run_single_job(args)
