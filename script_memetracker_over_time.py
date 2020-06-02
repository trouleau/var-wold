import numpy as np
import networkx as nx
import argparse
import torch
import pickle

import gb
import tsvar


def run_vi(train_events, test_events, adjacency_true, prior):
    dim = len(train_events)
    # Set prior: Alpha
    as_pr = prior['as_pr'] * np.ones((dim + 1, dim))
    ar_pr = prior['ar_pr'] * np.ones((dim + 1, dim))
    # Set prior: Beta
    bs_pr = prior['bs_pr'] * np.ones((dim, dim))
    br_pr = prior['br_pr'] * np.ones((dim, dim))
    # Set prior: Z
    zc_pr = [1.0 * np.ones((len(train_events[i]), dim+1)) for i in range(dim)]

    # Define model
    vi_model = tsvar.models.WoldModelVariationalOther(verbose=True)
    vi_model.observe(train_events)

    # Init the test model
    test_model = tsvar.models.WoldModelOther()
    test_model.observe(test_events)

    class MyCallback(tsvar.utils.callbacks.LearnerCallbackMLE):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, learner_obj, end=None, force=False):
            super().__call__(learner_obj, end, force)
            t = learner_obj._n_iter_done + 1
            if (t % self.print_every == 0) or force:
                # Extract mean of posteriors
                mu_hat = learner_obj._as_po[0, :] / learner_obj._ar_po[0, :]
                adj_hat = learner_obj._as_po[1:, :] / learner_obj._ar_po[1:, :]
                beta_hat = learner_obj._br_po[:, :] / (learner_obj._bs_po[:, :] + 1) + 1
                coeffs_hat = torch.tensor(np.hstack((
                    mu_hat, beta_hat.flatten(), adj_hat.flatten()
                )))
                loglik = float(test_model.log_likelihood(coeffs_hat)) / sum(map(len, test_events))
                print(f'------> vi_ll: {loglik:.2f}')

    callback = MyCallback(
        x0=(as_pr[1:, :] / ar_pr[1:, :]).flatten(), print_every=1,
        coeffs_true=adjacency_true, acc_thresh=0.05, dim=dim,
        widgets={'f1score', 'relerr', 'prec@5', 'prec@10', 'prec@20'},
        default_end='\n')

    # Fit model
    vi_model.fit(as_pr=as_pr, ar_pr=ar_pr, bs_pr=bs_pr, br_pr=br_pr, zc_pr=zc_pr,
                 max_iter=20, tol=1e-5, callback=callback)

    # Extract mean of posteriors
    mu_hat = vi_model._as_po[0, :] / vi_model._ar_po[0, :]
    adj_hat = vi_model._as_po[1:, :] / vi_model._ar_po[1:, :]
    beta_hat = vi_model._br_po[:, :] / (vi_model._bs_po[:, :] + 1) + 1
    coeffs_hat = torch.tensor(np.hstack((
        mu_hat, beta_hat.flatten(), adj_hat.flatten()
    )))

    # Compute heldout log-likelihood on test set
    vi_ll = float(test_model.log_likelihood(coeffs_hat)) / sum(map(len, test_events))

    return vi_ll, vi_model


def run_gb(train_events, test_events):
    # Define model
    granger_model = gb.GrangerBusca(
        alpha_prior=10.0,
        num_iter=3000,
        metropolis=True,
        beta_strategy=1.0,
        num_jobs=10,
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

    return gb_ll, granger_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', dest='out_path', type=str, required=False,
                        default='memetracker-results.pk',
                        help="Experiment directory")
    parser.add_argument('--aspr', dest='as_pr', type=float, required=False,
                        default=None, help="Experiment directory")
    parser.add_argument('--arpr', dest='ar_pr', type=float, required=False,
                        default=None, help="Experiment directory")
    parser.add_argument('--bspr', dest='bs_pr', type=float, required=False,
                        default=None, help="Experiment directory")
    parser.add_argument('--brpr', dest='br_pr', type=float, required=False,
                        default=None, help="Experiment directory")
    args = parser.parse_args()

    prior = {
        'as_pr': args.as_pr or 10.0,
        'ar_pr': args.ar_pr or 100.0,
        'bs_pr': args.bs_pr or 101.0,
        'br_pr': args.br_pr or 100.0
    }


    # Load the dataset
    INPUT_PATH = "/root/workspace/var-wold/data/memetracker/memetracker-top100-clean.pickle.gz"
    dataset = tsvar.preprocessing.MemeTrackerDataset(INPUT_PATH)
    dataset.data.Timestamp /= 426.3722723177017

    # Number of chunks to use
    t0 = dataset.data.Timestamp.min()
    t1 = dataset.data.Timestamp.max()

    res = list()

    for chunk_idx, chunk_start in enumerate(np.arange(0, 1.0 - 0.05 - 0.02, 0.02)):

        train_start = t0 + (t1 - t0) * chunk_start
        train_end = t0 + (t1 - t0) * (chunk_start + 0.05)
        test_end = t0 + (t1 - t0) * (chunk_start + 0.05 + 0.02)

        print()
        print('-' * 20, f'Start chunk {chunk_idx}')
        print(f'- Train window: [{train_start:.2f}, {train_end:.2f}]')
        print(f'- Test window: [{train_end:.2f}, {test_end:.2f}]')

        # Extract train/test sets for this chunk
        train_events, train_graph, test_events, test_graph = dataset.build_train_test(train_start, train_end, test_end)
        nodelist = sorted(train_events.keys())
        adjacency_true = nx.adjacency_matrix(train_graph, nodelist).toarray()
        train_events = [train_events[v] for v in nodelist]
        test_events = [test_events[v] for v in nodelist]

        # Print stats on the training set
        print('Train set')
        print(f"  Num. of dimensions: {len(train_events):,d}")
        print(f"      Num. of events: {sum(map(len, train_events)):,d}")
        print(f"  Observation window: [{min(map(min, train_events)):.2f}, {max(map(max, train_events)):.2f}]")
        print()
        print('Test set')
        print(f"  Num. of dimensions: {len(test_events):,d}")
        print(f"      Num. of events: {sum(map(len, test_events)):,d}")
        print(f"  Observation window: [{min(map(min, test_events)):.2f}, {max(map(max, test_events)):.2f}]")
        print()

        # Run VI
        print()
        print('--- VI')
        vi_ll, vi_model = run_vi(train_events, test_events, adjacency_true, prior)
        print(f'Result VI: chunk={chunk_idx:d} ll_mean={vi_ll:.4f}')

        # Run GB
        print()
        print('--- GB')
        gb_ll, gb_model = run_gb(train_events, test_events)
        print(f'Result GB: chunk={chunk_idx:d} ll={gb_ll:.4f}')

        # Store result
        print()
        print('Save results...')
        res.append({
            'chunk_idx': chunk_idx,
            'chunk_start': chunk_start,
            'dim': len(nodelist),
            'vi_prior': prior,
            'vi_ll': vi_ll,
            'vi_model': vi_model,
            'gb_ll': gb_ll,
            'gb_model': gb_model,
        })

        # Save the results
        with open(args.out_path, 'wb') as f:
            pickle.dump(res, f)

        print('=' * 50)
