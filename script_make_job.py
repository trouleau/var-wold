"""
Generate a parameters for a job, i.e.:

    For 1:`num_graphs`, generate:
        - a set of parameters to simulate a process in `dim` dimensions
        - a list of `num_sims` simulation seeds
"""
import argparse
import time
import json
import os

import numpy as np

from experiments_utils import generate_parameters

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', dest='exp_dir', type=str, required=True,
                        help="Experiment directory")

    parser.add_argument('-d', dest='dim', type=int, required=True,
                        help="Number of dimensions")
    parser.add_argument('-n', dest='max_jumps', type=int, required=True,
                        help="Number of events to simulate")

    parser.add_argument('-s', dest='num_sims', type=int, required=True,
                        help="Number of simulation seeds to generate")
    parser.add_argument('-g', dest='num_graphs', type=int, required=True,
                        help="Number of graphs to generate")

    args = parser.parse_args()

    # Set timestamps for experiment dir suffix
    exp_suffix = f"{time.time():.0f}"

    # if os.path.exists(args.exp_dir) and len(os.listdir(args.exp_dir)) > 0:
    #     raise ValueError('Experiment directory already exists. Abort!')

    # Empty dir might exists from previous bogus call
    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)

    for graph_idx in range(args.num_graphs):

        # Set seed to generate the parameters
        gen_seed = np.random.randint(2**31 - 1)

        # Generate parameters
        param_dict = generate_parameters(dim=args.dim, seed=gen_seed)

        # Generate list of seeds for simulations
        sim_seed_list = np.random.randint(2**31 - 1,
                                          size=args.num_sims).tolist()

        data = {
            'dim': args.dim,
            'max_jumps': args.max_jumps,
            'params': param_dict,
            'sim_seed_list': sim_seed_list,
            'gen_seed': gen_seed,
        }

        # Set the sub-experiment directory (for this set of parameters)
        sub_exp_name = (f'{exp_suffix:s}'           # time of generation
                        f'-g{graph_idx:02d}'        # graph
                        f'-d{args.dim:02d}'         # dim
                        f'-n{args.max_jumps:06d}')  # num events
        sub_exp_dir = os.path.join(
            args.exp_dir, sub_exp_name)
        # Creat sub-exp diretory
        os.mkdir(sub_exp_dir)

        # Save parameters
        with open(os.path.join(sub_exp_dir, 'params.json'), 'w') as out_f:
            json.dump(data, out_f)
