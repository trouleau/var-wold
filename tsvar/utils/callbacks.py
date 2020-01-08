import torch


class LearnerCallbackMLE:
    
    def __init__(self, x0, xtrue, acc_thresh=0.05, print_every=10):
        self.print_every = print_every
        self.last_coeffs = x0.clone()
        self.xtrue = xtrue.ravel()

    def __call__(self, learner_obj, end=""):
        t = learner_obj._n_iter_done + 1
        if t % self.print_every == 0:
            # Split parameters
            xt = learner_obj.coeffs.detach()
            x_diff = torch.abs(self.last_coeffs - xt).max()
            self.last_coeffs = xt.clone()
            loss = float(learner_obj._loss)
            print(f"\riter: {t:>5d} | dx: {x_diff:.2e} | loss: {loss:.2e}"
                  "    ", end=end, flush=True)