from collections import defaultdict
from collections import Counter
from math import exp
import gzip
import pickle
import numpy as np
import pandas as pd
import networkx as nx

from ..models import WoldModel


class Dataset:

    def __init__(self, path=None, top=None, timescale='busca'):
        if path is not None:
            self._from_raw_gz(path, top, timescale)

    def _from_raw_gz(self, path, top, timescale):
        """
        Process a G-zipped csv file of events. Each row is assumed to be
        formatted as (`sender`, `receiver`, ..., `timestamp`), with possible
        extra meta-data between the `receiver` and `timestamp` fields.
        `timestamp` must be the last column.
        """
        # Build point process from raw file (using same preprocessing as in
        #   https://github.com/flaviovdf/granger-busca/
        # Find the valid top entities
        valid = self._find_valid_entities(path, top)
        # Extract timestamps and ground truth adjacency graph
        timestamps, sources, graph, ids = self._build_point_process(path, valid)
        # Set point process attributes as `np.ndarray`s
        self.timestamps = list(map(np.array, timestamps))
        self.sources = list(map(np.array, sources))
        # Delete the original objects
        del timestamps
        del sources
        # Sort the timestamps and their source (if necessary)
        sargs = list(map(np.argsort, self.timestamps))  # argsort timestamps
        self.timestamps = [ev[s] for ev, s in zip(self.timestamps, sargs)]
        self.sources = [ev[s] for ev, s in zip(self.sources, sargs)]
        # Translate time axis to origin
        min_time = min(map(min, self.timestamps))
        self.timestamps = [ev - min_time for ev in self.timestamps]
        # Set dimension attribute
        self.dim = len(self.timestamps)
        # Rescale time
        if timescale == 'busca':
            timescale = self._compute_median_timescale(self.timestamps)
        elif not (isinstance(timescale, (int, float)) and (timescale > 0)):
            raise ValueError('`timescale should be a positive number`')
        self.timestamps = [ev / timescale for ev in self.timestamps]
        self.time_scale = timescale
        # Set end time attribute
        self.end_time = max(map(max, self.timestamps))
        # Set sparse adjacency matrix attribute
        self.graph = self._build_graph_with_indices(graph, ids)
        self.graph_names = graph
        assert self.graph.number_of_nodes() == self.dim, "Something went wrong with ground truth graph"
        # Set name <-> idx mapping attributes
        self.name_to_idx = ids
        self.idx_to_name = dict(zip(ids.values(), ids.keys()))
        # Keep track of top value
        self.top = top

    @classmethod
    def from_data(cls, timestamps, idx_to_name, graph, timescale='busca'):
        dataset = cls()
        # Set timestamps attributes
        dataset.timestamps = timestamps
        dataset.dim = len(dataset.timestamps)
        dataset.end_time = max(map(max, dataset.timestamps))
        dataset.top = -1
        # Rescale time
        if timescale == 'busca':
            timescale, busca_beta_ji = dataset._compute_median_timescale(dataset.timestamps)
        elif not (isinstance(timescale, (int, float)) and (timescale > 0)):
            raise ValueError('`timescale should be a positive number`')
        dataset.timestamps = [ev / timescale for ev in dataset.timestamps]
        dataset.time_scale = timescale
        dataset.busca_beta_ji = busca_beta_ji
        # Set end time attribute
        dataset.end_time = max(map(max, dataset.timestamps))
        # Set names/idx mappings
        dataset.idx_to_name = idx_to_name
        dataset.name_to_idx = {v: k for k, v in idx_to_name.items()}
        # Set graph attribute
        dataset.graph = graph
        return dataset

    @classmethod
    def from_pickle(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _compute_median_timescale(self, timestamps, return_busca_betas=True):
        """Compute the time scale leading to unit median inter-arrival time"""
        # Compute the inter-arrival times
        wmod = WoldModel()
        wmod.observe(timestamps)
        # Compute the median inter-event time accross all events
        median_delta = np.median(np.hstack(map(np.ravel, wmod.delta_ikj)))
        # Busca estimator for beta in Wold processes
        scale = median_delta / exp(1)
        return scale

    def _find_valid_entities(self, path, top):
        # Count number of events in dst
        count = defaultdict(int)
        with gzip.open(path, 'r') as in_file:
            for line in in_file:
                if b',' in line:
                    spl = line.split(b',')
                else:
                    spl = line.split()
                src, dst = spl[:2]
                count[dst] += 1
        # Filter valid users
        if top is None:
            # All users with at least one received events
            valid = set(count.keys())
        else:
            # top users with most received events
            valid = set()
            for v, k in sorted(((v, k) for k, v in count.items()), reverse=True):
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
                if b',' in line:  # comma-separated
                    spl = line.split(b',')
                else:  # whitespace-separated
                    spl = line.split()
                # First two columns must be src & dst
                src, dst = spl[:2]
                # Last column must be the timestamp
                stamp = float(spl[-1])

                # Ignore if src or dst not valid
                if (src not in valid) or (src == b''):  # src can be null (in MemeTracker)
                    continue
                if dst not in valid:
                    continue

                # Add event
                if (src not in graph):
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
        digraph = nx.DiGraph()
        # Init all nodes
        for name, idx in ids.items():
            digraph.add_node(idx, name=name)
        num_nodes = digraph.number_of_nodes()

        for src in graph.keys():
            src_idx = ids[src]
            for dst, value in graph[src].items():
                dst_idx = ids[dst]
                digraph.add_edge(src_idx, dst_idx, weight=value)

        # Sanity check that no new node was added
        assert digraph.number_of_nodes() == num_nodes

        return digraph


class MemeTrackerDataset:

    def __init__(self, path):
        """
        Load the MemeTracker dataset

        Arguments:
        ----------
        path : str
            Path to the preprocessed MemeTracker dataset. Expects a pickled and
            gzipped `pandas` dataframe indexed by "Blog" name and with columns:
            * "Timestamp": the lists of events
            * "Hyperlink": the name of the corresponding source events
            * "Blog_idx": numerical index of the Blog
            * "Hyperlink_idx": list of numerical index of the source hyperlinks
        """
        # Read the preprocessed dateframe file
        self.data = pd.read_pickle(path)
        for col in ["Timestamp", "Hyperlink", "Blog_idx", "Hyperlink_idx"]:
            if col not in self.data.columns:
                raise ValueError(f"DataFrame column `{col}` is missing")
        # Rescale time to approx unit inter-event median
        self.data.Timestamp /= 426.3722723177017
        # Set dim
        self.dim = len(self.data)

    def sample(self, start_time, end_time):
        """
        Sample a subset of the the dataset and split in train and test sets.
        The training set contains all events in time window
            [`train_start`, `train_end`),
        and the test set contains all events in time window
            [`train_end`, `test_end`).

        Also compute the ground truth networks on both subsets of the data.

        Arguments:
        ----------
        start_time : float
            Start of the desired window
        end_time : float
            End of the desired window
        """
        mask = (self.data.Timestamp >= start_time) & (self.data.Timestamp < end_time)
        df = self.data.loc[mask].groupby('Blog').agg({'Timestamp': list, 'Hyperlink': list}).sort_index()

        # Format events as a dict of {blog_idx: array(events in blog)}
        events = df.Timestamp.apply(np.array).to_dict()
        events = {idx: ev - start_time for idx, ev in events.items()}

        edge_data = self.data.loc[mask].groupby(['Hyperlink', 'Blog']).agg({'Timestamp': 'count'})['Timestamp'].to_dict()
        edge_data = [(u, v, {'weight': w}) for (u, v), w in edge_data.items()]
        graph = nx.DiGraph()
        graph.add_nodes_from(events.keys())
        graph.add_edges_from(edge_data)

        return events, graph

    def filter_nodes(self, events, graph, nodes_to_keep):
        events = {idx: ev for idx, ev in events.items() if idx in nodes_to_keep}
        graph = graph.subgraph(nodes_to_keep)
        return events, graph

    def build_train_test(self, train_start, train_end, test_end):
        # Sample both observation windows
        train_events, train_graph = self.sample(train_start, train_end)
        test_events, test_graph = self.sample(train_end, test_end)
        # Keep only nodes with events in both sets
        nodes_to_keep = set(train_events.keys()).intersection(set(test_events.keys()))
        # Filter out nodes with no data
        train_events, train_graph = self.filter_nodes(train_events, train_graph, nodes_to_keep)
        test_events, test_graph = self.filter_nodes(test_events, test_graph, nodes_to_keep)
        return train_events, train_graph, test_events, test_graph
