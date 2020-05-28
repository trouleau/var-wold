from collections import namedtuple
import json
import time
import torch
import numpy as np

from . import metrics


class History:

    def __init__(self, field_names):
        self._list = list()
        self._item = namedtuple('HistoryItem', field_names)

    def append(self, **kwargs):
        try:
            item = self._item(**kwargs)
            self._list.append(item)
        except TypeError:
            print('not ok')
            raise ValueError('Invalid field name provided')

    def __getitem__(self, attr):
        if attr in self._item._fields:  # Get field
            return [getattr(item, attr) for item in self]
        elif isinstance(attr, int):  # Get list index
            return self._list[attr]
        else:
            raise ValueError(f'`{attr}` is not a valid field name')

    @property
    def _fields(self):
        return self._item._fields

    def __repr__(self):
        return repr(self._list)

    def __str__(self):
        return str(self._list)

    def __len__(self):
        return len(self._list)


class LearnerCallbackMLE:

    def __init__(self, x0, print_every=10, coeffs_true=None, acc_thresh=None,
                 dim=None, link_func=None, default_end="", widgets=None,
                 conv_n=10, conv_threshold=1e-4):
        self.print_every = print_every
        self.default_end = default_end
        self.n_params = len(x0)

        self.conv_n = conv_n
        self.conv_threshold = conv_threshold

        if widgets is not None:
            self.widgets = widgets
        else:
            self.widgets = {'acc', 'f1score', 'relerr',
                            'prec@5', 'prec@10', 'prec@20'}

        if coeffs_true is not None:  # If ground truth is provided
            self.has_ground_truth = True
            self.coeffs_true = coeffs_true.flatten()
            self.dim = dim
            self.acc_thresh = acc_thresh
            if link_func is None:
                def link_func(coeffs): return coeffs
            self.link_func = link_func
            # Init history (with ground truth metrics)
            self.history = History(field_names=(
                # 'coeffs',
                # 'loss',
                'iter', 'time',
                # 'accuracy',
                'f1score', 'relerr'))

        else:  # If ground truth is **not** provided
            self.has_ground_truth = False
            # Init history (without ground truth metrics)
            self.history = History(field_names=(
                # 'coeffs',
                # 'loss',
                'iter', 'time'))

        # Init previous variables for differential computation
        self.last_time = time.time()
        self.last_iter = 0
        self.last_coeffs = x0.numpy() if isinstance(x0, torch.Tensor) else x0
        self.last_loss = float("Inf")

    def __call__(self, learner_obj, end=None, force=False):
        t = learner_obj._n_iter_done + 1
        # Convert to numpy
        if isinstance(learner_obj.coeffs, torch.Tensor):
            coeffs = learner_obj.coeffs.detach().clone().numpy()
        else:
            coeffs = learner_obj.coeffs.copy()
        # Apply link function
        coeffs = self.link_func(coeffs)
        # Sanity check
        assert len(coeffs) == self.n_params, "Inconsistent number of parameters provided"
        # Set loss
        loss = np.nan
        if hasattr(learner_obj, '_loss'):
            loss = float(learner_obj._loss.detach())
        # Print & record
        if (t % self.print_every == 0) or force:
            # Set call time
            call_time = time.time()
            # Compute difference in vars since last call
            x_diff = np.abs(self.last_coeffs - coeffs).max()
            time_diff = (call_time - self.last_time) / (t - self.last_iter)
            # Write message
            # base
            message = f"\riter: {t:>5d} | dx: {x_diff:+.4e}"
            # loss widget
            if hasattr(learner_obj, '_loss'):
                loss_diff = (loss - self.last_loss)
                message += f" | loss: {loss:.4e} | dloss: {loss_diff:+.2e}"
            # ground truth widget
            if self.has_ground_truth:
                if 'acc' in self.widgets:
                    acc = metrics.accuracy(adj_test=coeffs[-self.dim**2:],
                                           adj_true=self.coeffs_true[-self.dim**2:],
                                           threshold=self.acc_thresh)
                    message += f" | acc: {acc:.2f}"

                if 'f1score' in self.widgets:
                    f1score = metrics.fscore(adj_test=coeffs[-self.dim**2:],
                                             adj_true=self.coeffs_true[-self.dim**2:],
                                             threshold=self.acc_thresh)
                    message += f" | f1-score: {f1score:.2f}"

                if 'relerr' in self.widgets:
                    relerr = metrics.relerr(adj_test=coeffs[-self.dim**2:],
                                            adj_true=self.coeffs_true[-self.dim**2:])
                    message += f" | relerr: {relerr:.2e}"

                if 'prec@5' in self.widgets:
                    precAt5 = metrics.precision_at_n(adj_test=coeffs[-self.dim**2:],
                                                     adj_true=self.coeffs_true[-self.dim**2:],
                                                     n=5)
                    message += f" | p@5: {precAt5:.2f}"

                if 'prec@10' in self.widgets:
                    precAt10 = metrics.precision_at_n(adj_test=coeffs[-self.dim**2:],
                                                      adj_true=self.coeffs_true[-self.dim**2:],
                                                      n=10)
                    message += f" | p@10: {precAt10:.2f}"

                if 'prec@20' in self.widgets:
                    precAt20 = metrics.precision_at_n(adj_test=coeffs[-self.dim**2:],
                                                      adj_true=self.coeffs_true[-self.dim**2:],
                                                      n=20)
                    message += f" | p@20: {precAt20:.2f}"

                # Add to history
                self.history.append(
                    # coeffs=coeffs.tolist(),
                    # loss=loss,
                    iter=t, time=time_diff,
                    # accuracy=acc,
                    f1score=f1score, relerr=relerr)
            else:
                # Add to history
                self.history.append(
                    # coeffs=coeffs.tolist(),
                    # loss=loss,
                    iter=t, time=time_diff)

            # runtime widget
            message += f" | time/it: {time_diff:.2e}"

            # print message
            print(message + " "*5, end=end or self.default_end, flush=True)
            # Update last vars
            self.last_time = time.time()
            self.last_iter = t
        # Update last vars
        self.last_coeffs = coeffs
        self.last_loss = loss

    def has_converged(self, metric='relerr', n=None, threshold=None):
        n = n or self.conv_n
        threshold = threshold or self.conv_threshold
        # NOTE: history must have enough elements for averaging
        if len(self.history) >= max(2, n):
            val = np.mean(np.abs(np.diff(self.history[metric][-n:])))
            # print('-->', val, threshold)
            if val < threshold:
                return True
        return False

    def to_dict(self):
        """Serialize the history into a dict of field: list of values"""
        return {attr: self.history[attr] for attr in self.history._fields}
