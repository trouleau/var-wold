from multiprocessing import Process, cpu_count
import numpy as np
import argparse
import json
import glob
import sys
import os

import torch

from experiments_utils import (generate_data, run_mle, run_bbvi,
                               run_vi_fixed_beta, run_vi, run_gb)


# Set torch parallel threads to 1 to allow for multiple parallel jobs to run
torch.set_num_threads(1)


def run_single_job(param_fname, out_fname, sim_idx, stdout=None, stderr=None):

    # Redirect stdout/stderr
    if stdout is not None:
        sys.stdout = open(stdout, 'w')
    if stderr is not None:
        sys.stderr = open(stderr, 'w')

    # Load parameters
    with open(param_fname, 'r') as in_f:
        sim_param_dict = json.load(in_f)
    param_dict = {k: np.array(v) for k, v in sim_param_dict['params'].items()}
    sim_seed = sim_param_dict['sim_seed_list'][sim_idx]
    max_jumps = sim_param_dict['max_jumps']

    print()
    n_sim = len(sim_param_dict['sim_seed_list'])
    print(f'Simulation {sim_idx+1:>2d} / {n_sim:>2d}', flush=True)

    # Simulate data
    events, end_time, _ = generate_data(max_jumps=max_jumps, sim_seed=sim_seed,
                                        **param_dict)

    res_dict = {}

    print()
    print('Run MLE')
    print('-------')
    res_dict['mle'] = run_mle(events, end_time, param_dict, seed=sim_seed)
    print()
    print('-'*80, flush=True)

    print()
    print('Run BBVI')
    print('--------')
    res_dict['bbvi'] = run_bbvi(events, end_time, param_dict, seed=sim_seed)
    print()
    print('-'*80, flush=True)

    print()
    print('Run VI Beta-Fixed')
    print('-----------------')
    res_dict['vi-fixed-beta'] = run_vi_fixed_beta(events, end_time, param_dict, seed=sim_seed)
    print()
    print('-'*80, flush=True)

    print()
    print('Run VI With-Beta')
    print('----------------')
    res_dict['vi'] = run_vi(events, end_time, param_dict, seed=sim_seed)
    print()
    print('-'*80, flush=True)

    print()
    print('Run GrangerBusca')
    print('----------------')
    res_dict['gb'] = run_gb(events, end_time, param_dict, seed=sim_seed)

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
    args = parser.parse_args()

    # Pattern to extract list of parameter files
    search_pattern = os.path.join(args.exp_dir, '*', 'params.json')

    # Init the list of arguments for the workers
    pool_args = list()

    # For each sub-experiment (each set of parameters)
    for param_fname in glob.glob(search_pattern):
        # Extract sub-experiment directory
        sub_exp_dir = os.path.split(param_fname)[0]
        # For each simulation
        for sim_idx in range(args.n_sims):
            # Build output filename
            out_fname = os.path.join(sub_exp_dir, f'output-{sim_idx:02d}.json')
            # Build stdout/stderr filenames
            stdout = os.path.join(sub_exp_dir, f'stdout-{sim_idx:02d}')
            stderr = os.path.join(sub_exp_dir, f'stderr-{sim_idx:02d}')
            # Add tuple of arguments to list
            pool_args.append(
                (param_fname, out_fname, sim_idx, stdout, stderr))

    # Reverse list of args (will be popped from last element)
    pool_args = pool_args[::-1]

    print(f"Start {len(pool_args):d} experiments on a pool of {args.n_workers:d} workers")
    print(f"=============================================================================")

    proc_list = list()

    while len(pool_args) > 0:

        this_args = pool_args.pop()

        print("Start process with parameters:", this_args)

        proc = Process(target=run_single_job, args=this_args)
        proc_list.append(proc)
        proc.start()

        if len(proc_list) == args.n_workers:
            # Wait until all processes are done
            for proc in proc_list:
                proc.join()
            # Reset process list
            proc_list = list()
            print()

    print('Job Done.')
