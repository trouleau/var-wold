from multiprocessing import Pool
import os, gzip, pickle, glob
from itertools import product
import pandas as pd
import numpy as np



# Define the parameters
#
# d: the number of most cited sites you want to keep
#
# list_df: the list of months you want to keep
#


d = 100

list_df = [
    'data/df_2008-08.csv', 'data/df_2008-09.csv',
    'data/df_2008-10.csv', 'data/df_2008-11.csv',
    'data/df_2008-12.csv', 'data/df_2009-01.csv',
    'data/df_2009-02.csv', 'data/df_2009-03.csv',
    'data/df_2009-04.csv']

start_month = '2008-08'

dir_name = "top{}_{}months_start_{}".format(d, len(list_df), start_month)

if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

if __name__ == '__main__':

    from nphc.datasets.memetracker.processing import count_top, create_pp, true_G

    # counts the occurences of the sites for each month
    def worker1(x):
        return count_top.worker(x, dir_name)
    pool1 = Pool(processes=len(list_df))
    pool1.map(worker1, list_df)

    # aggregate the counts and save the top d sites
    count_top.save_top_d(d, dir_name)

    # useful variables for the worker below
    start = pd.to_datetime(start_month + '-01 00:00:00')
    top_d = pd.read_csv(dir_name + '/top_' + str(d) + '.csv')
    ix2url = {ix: url for ix, url in enumerate(top_d['url'])}

    # create multivariate point process for the top d sites
    for filename in list_df:
        def worker2(x):
            return create_pp.worker(x, filename, start, ix2url, dir_name)
        indices = np.arange(d, dtype=int)
        pool2 = Pool(40)
        pool2.map(worker2, indices)
        print("Work done for {}.".format(filename[5:]))

    # reduce the processes divided by month into one process per website
    list_of_list_files = []
    for i in range(d):
        num = create_pp.ix2str(i)
        list_of_list_files.append(glob.glob(dir_name + "/process_{}*.pkl.gz".format(num)))

    def worker3(x):
        return create_pp.reducer(x)
    pool3 = Pool()
    pool3.map(worker3, list_of_list_files)

    # estimate G from the labelled links
    def worker4(x):
        return true_G.worker(x, list_df, ix2url)
    tuple_indices = list(product(range(d), repeat=2))
    pool4 = Pool(40)

    # save the results
    res = pool4.map(worker4, tuple_indices)
    res_mat = np.array(res).reshape(d, d)
    f = gzip.open(dir_name+'/true_G.pkl.gz', 'wb')
    pickle.dump(res_mat, f, protocol=2)
    f.close()
