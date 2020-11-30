from multiprocessing import Pool, cpu_count
from datetime import datetime
import numpy as np
import argparse
import pandas as pd
import time
import json
import glob
import sys
import os

import torch

from experiments_utils import (generate_data, run_mle, run_mle_other, run_bbvi,
                               run_vi_fixed_beta, run_vi, run_gb)


# Set torch parallel threads to 1 to allow for multiple parallel jobs to run
torch.set_num_threads(1)


def build_prior_dict_arr(N=50):
    a_mean = 0.1 * np.ones(N)
    a_var = np.logspace(-2, 2, N)
    as_pr = a_mean ** 2 / a_var
    ar_pr = a_mean / a_var

    b_mean = 1.0 * np.ones(N)
    b_var = np.logspace(-2, 2, N)
    bs_pr = b_mean ** 2 / b_var + 2
    br_pr = b_mean * (b_mean ** 2 / b_var + 1)

    prior_dict_list = list()

    for i in range(N):
        for j in range(N):
            prior_dict_list.append({
                'as_pr': as_pr[i], 'ar_pr': ar_pr[i],
                'bs_pr': bs_pr[j], 'br_pr': br_pr[j],
            })

    return prior_dict_list


PRIOR_DICT_ARR = build_prior_dict_arr(N=20)


def pre_run(param_fname, sim_idx):
    # Load parameters
    with open(param_fname, 'r') as in_f:
        sim_param_dict = json.load(in_f)
    param_dict = {k: np.array(v) for k, v in sim_param_dict['params'].items()}
    sim_seed = sim_param_dict['sim_seed_list'][sim_idx]
    max_jumps = sim_param_dict['max_jumps']
    # Simulate data
    events, end_time, _ = generate_data(max_jumps=max_jumps, sim_seed=sim_seed,
                                        **param_dict)
    return events, end_time


def run_single_job(events, end_time, param_fname, out_fname, sim_idx, prior_dict):

    # Load parameters
    with open(param_fname, 'r') as in_f:
        sim_param_dict = json.load(in_f)
    param_dict = {k: np.array(v) for k, v in sim_param_dict['params'].items()}
    sim_seed = sim_param_dict['sim_seed_list'][sim_idx]

    # Log in main std before redirecting
    job_args = (param_fname, out_fname, sim_idx)
    job_start_time = str(datetime.now())
    print('Starting job:', job_args, 'at', job_start_time,
          file=sys.__stdout__, flush=True)

    res_dict = {}
    res_dict.update(prior_dict)

    print()
    print('Run VI')
    print('------')
    print()
    print('With prior:', prior_dict)
    print()
    res_dict['vi'] = run_vi(events, end_time, param_dict, seed=sim_seed, prior_dict=prior_dict)
    print()
    print('-'*80, flush=True)

    with open(out_fname, 'w') as out_f:
        print('Save:', out_fname, file=sys.__stdout__, flush=True)
        json.dump(res_dict, out_f)

    print()
    print('Job Finished', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--input', dest='exp_dir', type=str,
                        required=True, help="Input directory")
    parser.add_argument('-s', '--n_sims', dest='n_sims', type=int,
                        required=True, help="Number of simulatins per sub-exp")
    args = parser.parse_args()

    # Pattern to extract list of parameter files
    search_pattern = os.path.join(args.exp_dir, '*', 'params.json')

    # For each sub-experiment (each set of parameters)
    for param_fname in sorted(glob.glob(search_pattern)):

        # Extract sub-experiment directory
        sub_exp_dir = os.path.split(param_fname)[0]

        # For each simulation
        for sim_idx in range(args.n_sims):

            # Simulate dataset
            print(f'Simulate data for fname:{param_fname}, idx: {sim_idx}...')
            events, end_time = pre_run(param_fname, sim_idx)
            print('done.')
            print()

            print('Run inference')
            print('=' * 80, flush=True)

            # Init the list of arguments for the workers
            pool_args = list()

            # For each prior value
            for p_idx, prior_dict in enumerate(PRIOR_DICT_ARR):

                # Build output filename
                out_fname = os.path.join(sub_exp_dir, f'output-{sim_idx:02d}-{p_idx:04d}.json')

                run_single_job(events, end_time, param_fname, out_fname, sim_idx, prior_dict)

            print()
