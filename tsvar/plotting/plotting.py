import matplotlib
from matplotlib.backends.backend_pgf import FigureCanvasPgf
import numpy as np

from ..utils import metrics


THRESHOLD = 0.05  # Default threshold for small values


AISTAT_RCPARAMS = {
    "figure.autolayout": False,         # Makes sure nothing the feature is neat & tight.
    "figure.figsize": (3.25, 2.95),     # Column width: 3.25in
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


def set_neurips_matplotib_rcparams():
    matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
    matplotlib.rcParams.update(NEURIPS_RCPARAMS)


def set_aistat_matplotib_rcparams():
    matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
    matplotlib.rcParams.update(AISTAT_RCPARAMS)


def make_metric(df, func, col, col_true='adjacency', **kwargs):
    def processed_func(row):
        try:
            return func(row[col].flatten(), row[col_true].flatten(), **kwargs)
        except Exception:
            return np.nan
    return df.apply(processed_func, axis=1)


def make_perf(df, func, prefix, suf_col_dict, **kwargs):
    col_list = list()
    for suf, col in suf_col_dict.items():
        name = '_'.join([prefix, suf])
        df[name] = make_metric(df, func, col=col, **kwargs)
        col_list.append(name)
    return col_list


def compute_runtime(row, start_idx=0, unit='sec', scale='lin'):
    try:
        if unit == 'sec':
            factor = 1
        elif unit == 'min':
            factor = 1 / 60
        elif unit == 'hour':
            factor = 1 / 3600
        else:
            raise ValueError('Unknown unit')
        times = row['time']
        if len(times) > start_idx:
            last_iter = row['iter'][-1]
            val = np.mean(times[start_idx:]) * last_iter * factor
            if scale == 'log':
                val = np.log10(val)
            return val
        else:  # Use all iterations
            last_iter = row['iter'][-1]
            val = np.mean(times) * last_iter * factor
            if scale == 'log':
                val = np.log10(val)
            return val
    except TypeError:  # If no history, then `row` is not a dict, but is `NaN`
        return np.nan


def make_runtime_col(df, suf_col_dict, **kwargs):
    col_list = list()
    for suf, _ in suf_col_dict.items():
        history_col = f"{suf}_history"
        runtime_col = f"runtime_{kwargs.get('scale')}_{suf}"
        df[runtime_col] = df[history_col].apply(compute_runtime, **kwargs)
        col_list.append(runtime_col)
    return col_list


def compute_num_iter(row):
    try:
        return row['iter'][-1]
    except TypeError:  # If no history, then `row` is not a dict, but is `NaN`
        return np.nan


def make_num_iter(df, suf_col_dict):
    col_list = list()
    for suf, _ in suf_col_dict.items():
        history_col = f"{suf}_history"
        runtime_col = f"num_iter_{suf}"
        df[runtime_col] = df[history_col].apply(compute_num_iter)
        col_list.append(runtime_col)
    return col_list


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

    col_precAt_list = list()
    for n in [5, 10, 20, 50, 100, 200]:
        col_precAt_list += make_perf(df, metrics.precision_at_n,
                                     prefix=f'precAt{n}',
                                     suf_col_dict=suf_col_dict, n=n)

    col_runtime_lin = make_runtime_col(df, suf_col_dict, start_idx=5, unit='min',
                                       scale='lin')
    col_runtime_log = make_runtime_col(df, suf_col_dict, start_idx=5, unit='min',
                                       scale='log')

    col_num_iter = make_num_iter(df, suf_col_dict)

    if agg_col is None:
        return df
    else:
        # Make plotting df
        required_cols = (cols_acc + cols_relerr + cols_f1score
                         + col_precAt_list + col_num_iter
                         + col_runtime_lin + col_runtime_log
                         + cols_fp + cols_fn + [agg_col])
        agg_funcs = ['min', 'max', 'mean', 'std', 'count']
        df_plot = df[required_cols].groupby(agg_col).agg(agg_funcs)
        return df_plot
