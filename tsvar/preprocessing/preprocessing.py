from collections import defaultdict
import gzip
import pickle
import pandas as pd
import numpy as np


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

    def __init__(self, path, top=None):
        self._from_raw_gz(path, top)

    def _from_raw_gz(self, path, top):
        # Find the valid top entities
        valid = self._find_valid_entities(path, top)
        # Build point process from raw file
        timestamps, sources, graph, ids = self._build_point_process(path, valid)

        # Set sparse adjacency matrix attribute
        self.graph = self._build_graph_with_indices(graph, ids)

        # Set point process attributes
        self.timestamps = [np.array(sorted(ev)) for ev in timestamps]
        self.sources = sources

        # Set name <-> idx mapping attributes
        self.name_to_idx = ids
        self.idx_to_name = dict(zip(ids.values(), ids.keys()))

        # Keep track of top value
        self.top = top

    @classmethod
    def from_pickle(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

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
