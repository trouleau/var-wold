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


def build_prior_dict_arr(N):
    a_mean = 0.1 * np.ones(N)
    a_var = np.logspace(-2, 2, N)
    as_pr = a_mean ** 2 / a_var
    ar_pr = a_mean / a_var

    b_mean = 1.0 * np.ones(N)
    b_var = np.logspace(-2, 2, N)
    bs_pr = b_mean ** 2 / b_var + 2
    br_pr = b_mean * (b_mean ** 2 / b_var + 1)

    df = pd.DataFrame({'as_pr': as_pr, 'ar_pr': ar_pr,
                       'bs_pr': bs_pr, 'br_pr': br_pr})
    return df.to_dict('records')


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


def run_single_job(events, end_time, param_fname, out_fname, sim_idx, prior_dict, stdout=None, stderr=None):

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

    # Redirect stdout/stderr
    if stdout is not None:
        sys.stdout = open(stdout, 'w')
    if stderr is not None:
        sys.stderr = open(stderr, 'w')

    res_dict = {}

    print()
    print('Run VI')
    print('------')
    res_dict['vi'] = run_vi(events, end_time, param_dict, seed=sim_seed, prior_dict=prior_dict)
    print()
    print('-'*80, flush=True)

    with open(out_fname, 'w') as out_f:
        json.dump(res_dict, out_f)

    print()
    print('Job Finished', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--input', dest='exp_dir', type=str,
                        required=True, help="Input directory")
    parser.add_argument('-p', '--pool', dest='n_workers', type=int,
                        required=False, default=cpu_count() - 1,
                        help="Size of the parallel pool")
    parser.add_argument('-s', '--n_sims', dest='n_sims', type=int,
                        required=True, help="Number of simulatins per sub-exp")
    parser.add_argument('--no-std-redirect', dest='no_std_redirect',
                        action="store_true", help="Do not redirect stdout/stderr")
    args = parser.parse_args()

    # Pattern to extract list of parameter files
    search_pattern = os.path.join(args.exp_dir, '*', 'params.json')

    # For each sub-experiment (each set of parameters)
    for param_fname in sorted(glob.glob(search_pattern)):

        # Extract sub-experiment directory
        sub_exp_dir = os.path.split(param_fname)[0]

        # For each simulation
        for sim_idx in range(args.n_sims):

            # Build output filename
            out_fname = os.path.join(sub_exp_dir, f'output-{sim_idx:02d}.json')

            # Simulate dataset
            print(f'Simulate data for fname:{param_fname}, idx: {sim_idx}...')
            events, end_time = pre_run(param_fname, sim_idx)
            print('done.')
            print()

            # Init the list of arguments for the workers
            pool_args = list()

            # For each prior value
            for p_idx, prior_dict in enumerate(PRIOR_DICT_ARR):

                # Build stdout/stderr filenames
                if args.no_std_redirect:
                    stdout = None
                    stderr = None
                else:
                    stdout = os.path.join(sub_exp_dir, f'stdout-{sim_idx:02d}-{p_idx:04d}')
                    stderr = os.path.join(sub_exp_dir, f'stderr-{sim_idx:02d}-{p_idx:04d}')
                # Add tuple of arguments to list
                pool_args.append((events, end_time, param_fname, out_fname, sim_idx, stdout, stderr))

            print(f"Start {len(pool_args):d} experiments on a pool of {args.n_workers:d} workers")
            print(f"=============================================================================")

            if args.n_workers > 1:
                # Init pool of workers
                pool = Pool(args.n_workers)
                # Run all simulations
                pool.starmap_async(run_single_job, pool_args)
                # Close pool
                pool.close()
                # Wait for them to finish
                pool.join()
            else:
                # Single process
                for args in pool_args:
                    run_single_job(*args)

            # Reset std redirect
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

            print('Job Done.')
            print()
