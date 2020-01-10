import numpy as np
import torch

import tsvar


def test_wold_model_loglikelihood():
    print('\nTesting: WoldModel.log_likelihood:')

    # Toy events
    events = [
        torch.tensor([1.0, 3.0, 4.0, 5.0]),
        torch.tensor([2.0, 4.5, 6.0])
    ]
    # Toy end time of observation
    end_time = 10.0
    # Parameters of the model, to evaluate the log-likelihood
    mu = torch.tensor([0.2, 0.1])
    beta = torch.tensor([1.0, 2.0])
    A = torch.tensor([[0.1, 0.9],
                      [0.4, 0.6]])
    coeffs = torch.cat((mu, beta, A.flatten()))
    # Create an instance of `WoldModel`, set the data and evaluate the 
    # log-likelihood
    model = tsvar.wold_model.WoldModel()
    model.set_data(events, end_time)
    ll_computed = model.log_likelihood(coeffs)

    # Define the groud-truth intensity at each observed event
    lam_true = [
        np.array([
            0.2, 
            0.2, 
            0.2 + 0.1 / (1 + 2) + 0.4 / (2 + 1), 
            0.2 + 0.1 / (1 + 1) + 0.4 / (2 + 2)
        ]),
        np.array([
            0.1, 
            0.1 + 0.9 / (1 + 1), 
            0.1 + 0.9 / (1 + 0.5) + 0.6 / (2 + 2.5)
        ])
    ]
    # Define the groud-truth intensity at the end time
    lam_end_true = [
        0.2 + 0.1 / (1 + 1) + 0.4 / (2 + 0.5),
        0.1 + 0.9 / (1 + 1) + 0.6 / (2 + 1.5)
    ]
    # Compute the ground-truth log-likelood value
    ll_true = (
        np.log(lam_true[0]).sum() 
        - np.sum(lam_true[0] * np.hstack((events[0][0], np.diff(events[0]))))
        - lam_end_true[0] * (end_time - events[0][-1])
        
        + np.log(lam_true[1]).sum() 
        - np.sum(lam_true[1] * np.hstack((events[1][0], np.diff(events[1]))))
        - lam_end_true[1] * (end_time - events[1][-1])
    )

    assert np.isclose(ll_computed, ll_true)
    print('  - Test succeeded!')


def test_wold_model_mle():
    print('\nTesting: WoldModel MLE:')

    print('  - Generate random model parameters...')
    # Define random parameters
    dim = 2  # Dimensionality of the process
    end_time = 2e5  # Choose a long observation window
    mu = torch.tensor([0.3, 0.1])
    beta = torch.tensor([1.11, 1.5])
    alpha = torch.tensor([
        [0.7, 0.3],
        [0.1, 0.9]
    ])
    coeffs_true = torch.cat((mu, beta, alpha.flatten())).numpy()
    print('  - Simulate lots data...')
    np.random.seed(42)  # Fix random seed for data generation
    # Simulate lots of data
    wold_sim = tsvar.simulate.GrangeBuscaSimulator(
        mu_rates=mu, Alpha_ba=alpha, Beta_b=beta)
    events = wold_sim.simulate(end_time)
    events = [torch.tensor(ev, dtype=torch.float) for ev in events]
    print((f"    - Simulated {sum(map(len, events)):,d} events "
           f"with end time: {end_time}"))
    print('  - Run MLE...')
    # Run MLE inference
    model = tsvar.wold_model.WoldModel()
    model.set_data(events, end_time)
    coeffs_start = torch.cat((
        0.5 * torch.ones(dim, dtype=torch.float),
        2.0 * torch.ones(dim, dtype=torch.float),
        0.5 * torch.ones((dim, dim), dtype=torch.float).flatten()
    ))
    C = 10000.0 * torch.ones(len(coeffs_true))
    prior = tsvar.priors.GaussianPrior(dim=None, n_params=len(coeffs_true), C=C)
    optimizer = torch.optim.Adam([coeffs_start], lr=0.05)
    learner = tsvar.learners.MLELearner(model=model, prior=prior, 
        optimizer=optimizer, tol=1e-5, max_iter=100000, debug=False)
    learner_callback = tsvar.utils.callbacks.LearnerCallbackMLE(
        coeffs_start, coeffs_true, print_every=10)
    coeffs_hat = learner.fit(
        events, end_time, coeffs_start, callback=learner_callback)
    coeffs_hat = coeffs_hat.detach().numpy()
    max_diff = np.max(np.abs(coeffs_true - coeffs_hat))
    print(f'  - coeffs_hat:  {coeffs_hat.round(2)}')
    print(f'  - coeffs_true: {coeffs_true.round(2)}')
    print(f'  - max_diff: {max_diff:.4f}')
    if max_diff < 0.1:
        print('  - Test succeeded!')
    else:
        print('  - Test FAILED!')

if __name__ == "__main__":
    test_wold_model_loglikelihood()
    test_wold_model_mle()
