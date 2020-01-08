import numpy as np
import torch

import tsvar


def test_wold_model():
    print('Testing: WoldModel.log_likelihood...')

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
    print('Test succeeded!')

if __name__ == "__main__":
    test_wold_model()
