import matplotlib
from matplotlib.backends.backend_pgf import FigureCanvasPgf
import numpy as np

from ..utils import metrics

THRESHOLD = 0.05  # Default threshold for small values


def set_neurips_matplotib_rcparams():
    NEURIPS_RCPARAMS = {
        "figure.autolayout": False,         # Makes sure nothing the feature is neat & tight.
        "figure.figsize": (5.5, 2.95),      # Text width: 5.5in (double figure minus spacing 0.2in).
        "figure.dpi": 150,                  # Displays figures nicely in notebooks.
        "axes.linewidth": 0.5,              # Matplotlib's current default is 0.8.
        "xtick.major.width": 0.5,
        "xtick.minor.width": 0.5,
        "ytick.major.width": 0.5,
        "ytick.minor.width": 0.5,
        "text.usetex": True,                # use LaTeX to write all text
        "font.family": "serif",             # use serif rather than sans-serif
        "font.serif": "Times New Roman",    # use "Times New Roman" as the standard font
        "font.size": 7,
        "axes.titlesize": 7,                # LaTeX default is 10pt font.
        "axes.labelsize": 7,                # LaTeX default is 10pt font.
        "legend.fontsize": 7,               # Make the legend/label fonts a little smaller
        "legend.frameon": True,             # Remove the black frame around the legend
        "patch.linewidth": 0.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "grid.linewidth": 0.3,
        "pgf.texsystem": "xelatex",         # use Xelatex which is TTF font aware
        "pgf.rcfonts": False,               # Use pgf.preamble, ignore standard Matplotlib RC
        "pgf.preamble": [
            r'\usepackage{fontspec}',
            r'\usepackage{unicode-math}',
            r'\setmainfont{Times New Roman}',
        ],
        'errorbar.capsize': 3.0,
    }
    matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
    matplotlib.rcParams.update(NEURIPS_RCPARAMS)


def make_metric(df, func, col, col_true='adjacency', **kwargs):
    def processed_func(row):
        return func(row[col].flatten(), row[col_true].flatten(), **kwargs)
    return df.apply(processed_func, axis=1)


def make_perf(df, func, prefix, suf_col_dict, **kwargs):
    col_list = list()
    for suf, col in suf_col_dict.items():
        name = '_'.join([prefix, suf])
        df[name] = make_metric(df, func, col=col, **kwargs)
        col_list.append(name)
    return col_list


def get_runtime(row, start_idx=0, unit='sec'):
    if unit == 'sec':
        factor = 1
    elif unit == 'min':
        factor = 1 / 60
    elif unit == 'hour':
        factor = 1 / 3600
    else:
        raise ValueError('Unknown unit')
    times = row['time'][start_idx:]
    if len(times) > 0:
        last_iter = row['iter'][-1]
        return np.mean(times) * last_iter * factor
    return np.nan


def make_plot_df(df, suf_col_dict, agg_col, threshold=THRESHOLD):
    # Compute all desired performance metrices
    cols_acc = make_perf(df, metrics.accuracy, prefix='acc',
                         suf_col_dict=suf_col_dict, threshold=threshold)
    cols_f1score = make_perf(df, metrics.fscore, prefix='f1score',
                             suf_col_dict=suf_col_dict, threshold=threshold)
    cols_fp = make_perf(df, metrics.false_positive, prefix='fp',
                        suf_col_dict=suf_col_dict, threshold=threshold)
    cols_fn = make_perf(df, metrics.false_negative, prefix='fn',
                        suf_col_dict=suf_col_dict, threshold=threshold)
    cols_relerr = make_perf(df, metrics.relerr, prefix='relerr',
                            suf_col_dict=suf_col_dict)
    # Make plotting df
    required_cols = (cols_acc + cols_relerr + cols_f1score
                     + cols_fp + cols_fn + [agg_col])
    agg_funcs = ['min', 'max', 'mean', 'std', 'count']
    df_plot = df[required_cols].groupby(agg_col).agg(agg_funcs)
    return df_plot
