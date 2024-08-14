import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_indices(reference, query):
    """
    get the indices of the query that are in reference.
    e.g.
    reference = torch.tensor([3, 10, 7])
    query = torch.tensor([7, 3, 3, 7, 10])
    find_indices(reference, query)
    >>> tensor([2, 0, 0, 2, 1])
    """
    return torch.max(reference[:, None] == query[None, :], axis=0).indices


def negative_sampling(source_ix, target_ix, pos_edges, size):
    """
    Negative sampling using GPU and batched impl.
    Create source_ix[i], target_ix[j] pairs that are not present in pos_edges.
    """
    size = size * 2
    neg_source_ix = torch.randperm(size) % len(source_ix)
    neg_source_ix = source_ix[neg_source_ix]
    neg_target_ix = torch.randperm(size) % len(target_ix)
    neg_target_ix = target_ix[neg_target_ix]
    samples = torch.stack([neg_source_ix, neg_target_ix]).T
    samples = torch.unique(samples, dim=0)

    y_true = torch.any(torch.all(samples[:, None] == pos_edges.T, axis=2), axis=1)
    samples = samples[~y_true]
    samples = samples[: size // 2].T
    return samples


def select_nodes_to_sample(data, split):
    """Select nodes to build negative samples based on the split"""
    source_ix = data["binds"].edge_label_index[0]
    target_ix = data["binds"].edge_label_index[1]
    if split != "source":
        source_ix = torch.cat((source_ix, data["binds"].edge_index[0]))
    if split != "target":
        target_ix = torch.cat((target_ix, data["binds"].edge_index[1]))
    return source_ix.unique(), target_ix.unique()


class SampleNegatives(BaseTransform):
    def __init__(self, edges, split, ratio):
        self.edges = torch.tensor(edges, device=DEVICE)
        self.split = split
        self.ratio = ratio

    def forward(self, data: HeteroData):
        data = data.to(DEVICE, non_blocking=True)

        num_pos = len(data["binds"].edge_label)
        # Select nodes
        subgraph_src, subgraph_tgt = select_nodes_to_sample(data, self.split)

        # map local (subgraph) edge indices to global indices
        global_src = data["source"].node_id[subgraph_src]
        global_tgt = data["target"].node_id[subgraph_tgt]

        size = num_pos * self.ratio
        neg_edges = negative_sampling(global_src, global_tgt, self.edges, size)

        # map global edge indices to local (subgraph) indices
        neg_src = find_indices(data["source"].node_id, neg_edges[0])
        neg_tgt = find_indices(data["target"].node_id, neg_edges[1])

        # concat current and new edges and labels
        neg_edges = torch.stack([neg_src, neg_tgt])
        new_edges = torch.cat((data["binds"].edge_label_index, neg_edges), axis=1)

        neg_label = torch.zeros(len(neg_src), device=DEVICE)
        new_label = torch.cat((data["binds"].edge_label, neg_label))

        # update data object
        data["binds"].edge_label = new_label
        data["binds"].edge_label_index = new_edges.contiguous()

        return data
