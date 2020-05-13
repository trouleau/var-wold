import os
import glob
import json
import torch
import numpy as np
import pandas as pd

import tsvar


def _extract_exp_name(output_fname):
    """Return the experiment name for an output filename"""
    return output_fname.split(os.path.sep)[-2]


def _process_param_file(param_fname):
    """Process a parameter file"""
    with open(param_fname, 'r') as f:
        data = json.load(f)
    series = pd.concat((
        pd.Series(data['params']).apply(np.array),
        pd.Series(data)
    ))
    del series['params']
    series['expId'] = os.path.split(param_fname)[0]
    return series


def _extract_param_df(exp_dir):
    """Extract data from all parameter files in an experiment directory"""
    param_file_list = glob.glob(os.path.join(exp_dir, '*', 'params.json'))
    param_df = pd.DataFrame(list(map(_process_param_file, param_file_list)))
    return param_df


def _process_mle(data):
    """Process the output of a MLE run"""
    series = pd.Series(data['coeffs']).apply(np.array)
    series['conv'] = data['conv']
    series['history'] = data['history']
    series['num_iter'] = data['history']['iter'][-1]
    # Add prefix
    series = series.rename({col: 'mle_' + col for col in series.index})
    return series


def _process_bbvi(data, dim=None):
    """Process the output of a BBVI run"""
    series = pd.Series()
    series['loc'] = np.array(data['coeffs']['loc'])
    series['log_scale'] = np.array(data['coeffs']['log-scale'])
    post = tsvar.posteriors.LogNormalPosterior()
    coeffs_mean = post.mean(torch.tensor(series['loc']), torch.tensor(series['log_scale'])).numpy()
    coeffs_mode = post.mode(torch.tensor(series['loc']), torch.tensor(series['log_scale'])).numpy()
    series['coeffs_mean'] = coeffs_mean
    series['coeffs_mode'] = coeffs_mode
    if dim:
        series['adj_mean'] = np.reshape(coeffs_mean[dim+dim**2:], (dim, dim))
        series['adj_mode'] = np.reshape(coeffs_mean[dim+dim**2:], (dim, dim))
    series['conv'] = data['conv']
    series['history'] = data['history']
    # Add prefix
    series = series.rename({col: 'bbvi_' + col for col in series.index})
    return series


def _process_vi(data):
    series = pd.Series(data['coeffs']).apply(np.array)
    series['conv'] = data['conv']
    series['history'] = data['history']
    # Extract mean/model posterior for coeffs
    model = tsvar.models.WoldModelVariational()
    series['adj_mode'] = model.alpha_posterior_mean(as_po=series['as_po'], ar_po=series['ar_po'])[1:,:]
    series['adj_mean'] = model.alpha_posterior_mode(as_po=series['as_po'], ar_po=series['ar_po'])[1:,:]
    # Add prefix
    series = series.rename({col: 'vi_' + col for col in series.index})
    return series


def _process_vi_fixed_beta(data):
    series = pd.Series(data['coeffs']).apply(np.array)
    series['conv'] = data['conv']
    series['history'] = data['history']
    # Extract mean/model posterior for coeffs
    model = tsvar.models.WoldModelVariationalFixedBeta()
    series['adj_mode'] = model.alpha_posterior_mean(as_po=series['as_po'], ar_po=series['ar_po'])[1:,:]
    series['adj_mean'] = model.alpha_posterior_mode(as_po=series['as_po'], ar_po=series['ar_po'])[1:,:]
    # Add prefix
    series = series.rename({col: 'vi_fixed_beta_' + col for col in series.index})
    return series


def _process_gb(data):
    series = pd.Series(data['coeffs']).apply(np.array)
    series['conv'] = data['conv']
    series['history'] = data['history']
    # Extract normalized adjacency
    series['adj_normed'] = series['adjacency'] / series['adjacency'].sum(axis=1)[:, None]
    # Add prefix
    series = series.rename({col: 'gb_' + col for col in series.index})
    return series


def _process_output_file(output_fname, param_df):
    """Process a single output file"""
    with open(output_fname, 'r') as f:
        all_data = json.load(f)
    # Extract experiment Id and output Id
    exp_id, output_idx = os.path.split(output_fname)
    # Extract number of dimensions
    dim = int(param_df.loc[param_df.expId == exp_id].iloc[0].dim)
    # Extract the output for each inference method
    series_list = list()
    print(f"Process file: {output_fname}...", end='\r', flush=True)
    for method, data in all_data.items():
        if method == 'mle':
            s = _process_mle(data)
            series_list.append(s)
        if method == 'bbvi':
            s = _process_bbvi(data, dim=dim)
            series_list.append(s)
        if method == 'vi':
            s = _process_vi(data)
            series_list.append(s)
        if method == 'vi-fixed-beta':
            s = _process_vi_fixed_beta(data)
            series_list.append(s)
        if method == 'gb':
            s = _process_gb(data)
            series_list.append(s)
    # Add experiment name `expId` and output index `outputIdx`
    if len(series_list) > 0:
        series = pd.concat(series_list)
        series['expId'] = exp_id
        series['outputIdx'] = output_idx
        series['dim_o'] = dim
        return series
    else:
        print('Failed:', output_fname)
        return pd.Series()


def _extract_output_df(exp_dir, param_df):
    """Extract data from all output files in an experiment directory"""
    output_file_list = sorted(glob.glob(os.path.join(exp_dir, '*', 'output-*.json')))
    output_df = pd.DataFrame([_process_output_file(fp, param_df) for fp in output_file_list])
    print()
    return output_df


def load_outputs(exp_dir):
    """Load all the outputs for an experiment"""
    # Load parameters
    param_df = _extract_param_df(exp_dir)
    # Load outputs
    output_df = _extract_output_df(exp_dir, param_df=param_df)
    # Merge
    df = pd.merge(output_df, param_df, on='expId', suffixes=('', '_true'), validate='m:1')
    return df
