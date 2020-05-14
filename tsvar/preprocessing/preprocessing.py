from collections import defaultdict
from math import exp
import gzip
import pickle
import numpy as np
import networkx as nx


import tsvar


def get_graph_stamps(path, top=None):

    # ==== Find valid entities
    # ========================

    count = defaultdict(int)  # Count number of events in dst
    srcs = set()  # Set of sources
    with gzip.open(path, 'r') as in_file:
        for line in in_file:
            if b',' in line:
                spl = line.split(b',')
            else:
                spl = line.split()
            src, dst = spl[:2]
            count[dst] += 1
            srcs.add(src)

    # Filter valid users
    if top is None:
        valid = srcs
    else:
        valid = set()
        for v, k in sorted(((v, k) for k, v in count.items()), reverse=True):
            if k in srcs:
                valid.add(k)
                if len(valid) == top:
                    break

    # ==== Build point process
    # ========================

    graph = {}  # Adjacency links graph[src][dst]
    ids = {}  # Id mapping dst -> incremental integers
    with gzip.open(path, 'r') as in_file:
        timestamps = []
        for line in in_file:

            # Read line
            if b',' in line:
                spl = line.split(b',')
            else:
                spl = line.split()

            src, dst = spl[:2]
            stamp = float(spl[-1])

            # Ignore if src or dst not valid
            if src not in valid:
                continue
            if dst not in valid:
                continue

            # Add event
            if src not in graph:
                graph[src] = {}
            if dst not in graph[src]:
                graph[src][dst] = 0
            graph[src][dst] += 1
            if dst in ids:
                timestamps[ids[dst]].append(stamp)
            else:
                ids[dst] = len(timestamps)
                timestamps.append([stamp])

    # ==== Consolidate `graph` structure
    # ==================================

    # Delete graph src keys that are not part of the dst in ids
    for id_ in list(graph.keys()):
        if id_ not in ids:
            del graph[id_]
    # Add missing keys for all dst not there with an empty adjacency
    for id_ in ids:
        if id_ not in graph:
            graph[id_] = {}

    return timestamps, graph, ids


class Dataset:

    def __init__(self, path, top=None, timescale='median', verbose=False):
        self._from_raw_gz(path, top, timescale, verbose)

    def _from_raw_gz(self, path, top, timescale, verbose):
        # Build point process from raw file (using same preprocessing as in
        #   https://github.com/flaviovdf/granger-busca/
        # Find the valid top entities
        valid = self._find_valid_entities(path, top)
        # Extract timestamps and ground truth adjacency graph
        timestamps, sources, graph, ids = self._build_point_process(path, valid)

        # Set point process attributes
        self.timestamps = list(map(np.array, timestamps))
        self.sources = list(map(np.array, sources))
        # Delete the original objects to avoid bugs
        del timestamps
        del sources

        # Process the timestamps
        sargs = list(map(np.argsort, self.timestamps))  # argsort timestamps
        # Sort the timestamps
        self.timestamps = [ev[s] for ev, s in zip(self.timestamps, sargs)]
        # Sort the ground truth causal source of each event
        self.sources = [ev[s] for ev, s in zip(self.sources, sargs)]

        # Set dimension attribute
        self.dim = len(self.timestamps)

        # Rescale time
        busca_beta_ji = None
        if timescale == 'median':
            timescale, busca_beta_ji = self._compute_median_timescale(self.timestamps)
        elif not (isinstance(timescale, (int, float)) and (timescale > 0)):
            raise ValueError('`timescale should be a positive number`')
        self.timestamps = [ev / timescale for ev in self.timestamps]

        # Set end time attribute
        self.end_time = max(map(max, self.timestamps))

        # Compute the Busca estimators of beta_ji (if not already done with timescale)
        self.busca_beta_ji = busca_beta_ji if (busca_beta_ji is not None) else self._compute_busca_beta_ji(self.timestamps)

        # Set sparse adjacency matrix attribute
        graph = self._build_graph_with_indices(graph, ids)
        self.graph = nx.DiGraph(graph)
        assert self.graph.number_of_nodes() == self.dim, "Something went wrong with ground truth graph"

        # Set name <-> idx mapping attributes
        self.name_to_idx = ids
        self.idx_to_name = dict(zip(ids.values(), ids.keys()))

        # Keep track of top value
        self.top = top

    @classmethod
    def from_pickle(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _compute_median_timescale(self, timestamps, return_busca_betas=True):
        """Compute the time scale leading to unit median inter-arrival time"""
        # Compute the inter-arrival times
        wmod = tsvar.models.WoldModel()
        wmod.observe(timestamps)
        median_delta = np.median(np.hstack(map(np.ravel, wmod.delta_ikj)))
        # Busca estimator for beta in Wold processes
        scale = median_delta / exp(1)

        if return_busca_betas:
            busca_beta_ji = self._compute_busca_beta_ji(timestamps, wmod)
            busca_beta_ji /= scale
            return scale, busca_beta_ji

        return scale

    def _compute_busca_beta_ji(self, timestamps, wmod=None):
        if wmod is None:
            # Init WoldModel to get inter-arrival time deltas
            wmod = tsvar.models.WoldModel()
            wmod.observe(timestamps)
        # Compute the busca estimaotr of beta_ji
        busca_beta_ji = np.zeros((wmod.dim, wmod.dim))
        for i in range(wmod.dim):
            busca_beta_ji[:, i] = np.median(wmod.delta_ikj[i], axis=0) / exp(1)
        return busca_beta_ji

    def _find_valid_entities(self, path, top):
        count = defaultdict(int)  # Count number of events in dst
        srcs = set()  # Set of sources
        with gzip.open(path, 'r') as in_file:
            for line in in_file:
                if b',' in line:
                    spl = line.split(b',')
                else:
                    spl = line.split()
                src, dst = spl[:2]
                count[dst] += 1
                srcs.add(src)

        # Filter valid users
        if top is None:
            valid = srcs
        else:
            valid = set()
            for v, k in sorted(((v, k) for k, v in count.items()), reverse=True):
                if k in srcs:
                    valid.add(k)
                    if len(valid) == top:
                        break
        return valid

    def _build_point_process(self, path, valid):
        graph = {}  # Adjacency links graph[src][dst]
        ids = {}  # Id mapping dst -> incremental integers
        with gzip.open(path, 'r') as in_file:
            timestamps = []
            sources = []
            for line in in_file:

                # Read line
                if b',' in line:
                    spl = line.split(b',')
                else:
                    spl = line.split()
                src, dst = spl[:2]
                stamp = float(spl[-1])

                # Ignore if src or dst not valid
                if src not in valid:
                    continue
                if dst not in valid:
                    continue

                # Add event
                if src not in graph:
                    graph[src] = {}
                if dst not in graph[src]:
                    graph[src][dst] = 0
                graph[src][dst] += 1
                if dst in ids:
                    timestamps[ids[dst]].append(stamp)
                    sources[ids[dst]].append(src)
                else:
                    ids[dst] = len(timestamps)
                    timestamps.append([stamp])
                    sources.append([src])

        # Consolidate `graph` structure
        # Delete graph src keys that are not part of the dst in ids
        for id_ in list(graph.keys()):
            if id_ not in ids:
                del graph[id_]
        # Add missing keys for all dst not there with an empty adjacency
        for id_ in ids:
            if id_ not in graph:
                graph[id_] = {}

        return timestamps, sources, graph, ids

    def _build_graph_with_indices(self, graph, ids):
        # Compute the adjacency indexed by index instead of
        graph_new = {}
        for src in graph.keys():
            src_idx = ids[src]
            graph_new[src_idx] = {}
            for dst, value in graph[src].items():
                dst_idx = ids[dst]
                graph_new[src_idx][dst_idx] = {'weight': value}
        return graph_new
