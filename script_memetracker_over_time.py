import numpy as np
import pandas as pd
import torch
import pickle

import gb

import tsvar
from tsvar.preprocessing import Dataset


def split_train_test(dataset, chunk_idx, chunk_total):
    # Set start and end times of train and test sets
    train_t0 = dataset.end_time * chunk_idx / chunk_total
    train_t1 = dataset.end_time * (chunk_idx + 1) / chunk_total
    test_t0 = dataset.end_time * (chunk_idx + 1) / chunk_total
    test_t1 = dataset.end_time * (chunk_idx + 2) / chunk_total
    # Filter the events in train/test observation windows
    train_events = [ev[(ev >= train_t0) & (ev < train_t1)] for ev in dataset.timestamps]
    test_events = [ev[(ev >= test_t0) & (ev < test_t1)] for ev in dataset.timestamps]
    # Remove dimensions with no nodes
    nodes_to_keep = np.array([(len(train_events[i]) > 0) & (len(test_events[i]) > 0)
                              for i in range(dataset.dim)])
    train_events = np.array(train_events)[nodes_to_keep].tolist()
    train_events = [ev - train_t0 for ev in train_events]
    test_events = np.array(test_events)[nodes_to_keep].tolist()
    test_events = [ev - test_t0 for ev in test_events]
    return train_events, test_events


def run_vi(train_events, test_events, chunk_idx):
    dim = len(train_events)
    # Set prior: Alpha
    as_pr = 10.0 * np.ones((dim + 1, dim))
    ar_pr = 100.0 * np.ones((dim + 1, dim))
    # Set prior: Beta
    bs_pr = 101.0 * np.ones((dim, dim))
    br_pr = 100.0 * np.ones((dim, dim))
    # Set prior: Z
    zc_pr = [1.0 * np.ones((len(train_events[i]), dim+1)) for i in range(dim)]

    # Define model
    vi_model = tsvar.models.WoldModelVariational(verbose=True)
    vi_model.observe(train_events)

    # Set callback (parameters of callback are just the posterior mean of alpha)
    callback = tsvar.utils.callbacks.LearnerCallbackMLE(
        x0=(as_pr[1:, :] / ar_pr[1:, :]).flatten(), print_every=1,
        coeffs_true=np.ones(dim ** 2),
        acc_thresh=0.05, dim=dim,
        widgets={'f1score', 'relerr'},
        default_end='\n')

    # Fit model
    vi_model.fit(as_pr=as_pr, ar_pr=ar_pr, bs_pr=bs_pr, br_pr=br_pr, zc_pr=zc_pr,
                 max_iter=20, tol=1e-5, callback=callback)

    # Init the test model
    test_model = tsvar.models.WoldModelOther()
    test_model.observe(test_events)

    # Extract mean of posteriors
    mu_hat = vi_model._as_po[0, :] / vi_model._ar_po[0, :]
    adj_hat = vi_model._as_po[1:, :] / vi_model._ar_po[1:, :]
    beta_hat = vi_model._br_po[:, :] / (vi_model._bs_po[:, :] + 1) + 1
    coeffs_hat = torch.tensor(np.hstack((
        mu_hat, beta_hat.flatten(), adj_hat.flatten()
    )))

    # Compute heldout log-likelihood on test set
    vi_ll = float(test_model.log_likelihood(coeffs_hat)) / sum(map(len, test_events))

    print(f'Result VI: chunk={chunk_idx:d} ll_mean={vi_ll:.4f}')

    return vi_ll, coeffs_hat


def run_gb(train_events, test_events, chunk_idx):
    # Define model
    granger_model = gb.GrangerBusca(
        alpha_prior=1.0/len(train_events),
        num_iter=300,
        metropolis=True,
        beta_strategy=1.0,
        num_jobs=48,
    )

    # Fit the model
    granger_model.fit(train_events)

    # Extract estimate of parameters
    dim = len(granger_model.mu_)
    mu_hat = granger_model.mu_
    adj_hat = granger_model.Alpha_.toarray()
    adj_hat = adj_hat / adj_hat.sum(axis=1)[:, None]
    beta_hat = np.ones((dim, dim)) * granger_model.beta_
    coeffs_hat = torch.tensor(np.hstack((
        mu_hat, beta_hat.flatten(), adj_hat.flatten()
    )))

    # Init the test model
    test_model = tsvar.models.WoldModelOther()
    test_model.observe(test_events)

    # Compute heldout log-likelihood on test set
    gb_ll = float(test_model.log_likelihood(coeffs_hat)) / sum(map(len, test_events))

    print(f'Result GB: chunk={chunk_idx:d} ll={gb_ll:.4f}')

    return gb_ll, coeffs_hat


if __name__ == "__main__":

    # Load the dataset
    INPUT_PATH = "/root/workspace/var-wold/data/memetracker/dataset_memetracker_good.pk"
    dataset = Dataset.from_pickle(INPUT_PATH)

    # Number of chunks to use
    chunk_total = 20

    res = list()

    for chunk_idx in range(chunk_total - 1):

        print()
        print(f'-------- Start chunk {chunk_idx}')
        print()

        # Extract train/test sets for this chunk
        train_events, test_events = split_train_test(dataset, chunk_idx, chunk_total)

        # Print stats on the training set
        print(f"Num. of dimensions: {len(train_events):,d}")
        print(f"    Num. of events: {sum(map(len, train_events)):,d}")
        print()
        print("Stats. of num. of events per dim:")
        num_jumps_per_dim = np.array(list(map(len, train_events)))
        print(pd.Series(num_jumps_per_dim).describe())

        # Run VI
        vi_ll, vi_coeffs_hat = run_vi(train_events, test_events, chunk_idx)

        # Run GB
        gb_ll, gb_coeffs_hat = run_gb(train_events, test_events, chunk_idx)

        # Store result
        res.append({
            'chunk_idx': chunk_idx,
            'chunk_total': chunk_total,
            'vi_ll': vi_ll,
            'vi_coeffs_hat': vi_coeffs_hat,
            'gb_ll': gb_ll,
            'gb_coeffs_hat': gb_coeffs_hat
        })

        print('-' * 50)

    # Save the results
    with open('memetracker-results.pk', 'wb') as f:
        pickle.dump(res, f)
