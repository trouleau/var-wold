from multiprocessing import Pool, cpu_count
from collections import namedtuple
import numpy as np

from script_run_real_data_job import run_single_job

Arguments = namedtuple('Arguments', ('input_path', 'top', 'output_path',
                                     'algo_filter', 'no_std_redirect',
                                     'as_pr', 'ar_pr'))

if __name__ == "__main__":

    globals()["Arguments"] = Arguments

    INPUT_PATH = 'data/enron_dataset_splitted_receivers.csv.gz'
    TOP = 100
    OUTPUT_PATH = 'output/real-data/'
    ALGO_FILTER = ['vi', 'vifb', 'gb']

    arg_list = list()

    for i in range(10):

        as_pr = np.random.uniform(0.01, 50.0)
        ar_pr = np.random.uniform(0.01, 50.0)

        args = Arguments(
            input_path=INPUT_PATH, top=TOP,
            output_path=OUTPUT_PATH,
            algo_filter=ALGO_FILTER,
            no_std_redirect=False,
            as_pr=as_pr, ar_pr=ar_pr)

        name_suffix = str(i)

        arg_list.append((args, name_suffix))

    # Init pool of processes
    pool = Pool(2)
    # Run in parallel
    pool.starmap(run_single_job, arg_list)
    # Close pool
    pool.close()
    # Wait for them to finish
    pool.join()