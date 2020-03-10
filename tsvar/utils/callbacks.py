import torch


class LearnerCallbackMLE:
    
    def __init__(self, x0, xtrue, acc_thresh=0.05, print_every=10):
        self.print_every = print_every
        self.last_coeffs = x0.clone()
        self.xtrue = xtrue.ravel()

        self.history = {
            'coeffs': [],
            'loss': [],
        }

    def __call__(self, learner_obj, end=""):
        t = learner_obj._n_iter_done + 1
        
        if t % self.print_every == 0:
            xt = learner_obj.coeffs.detach().clone()
            loss = float(learner_obj._loss.detach())

            self.history['coeffs'].append(xt)
            self.history['loss'].append(loss)

            x_diff = torch.abs(self.last_coeffs - xt).max()
            print(f"\riter: {t:>5d} | dx: {x_diff:.4e} | loss: {loss:.4e}"
                  "    ", end=end, flush=True)
            
        self.last_coeffs = learner_obj.coeffs.detach().clone()