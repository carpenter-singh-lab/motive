import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform

SEED = 2024319

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SampleNegatives(BaseTransform):
    def __init__(self, edges, datasplit, ratio=1):
        self.edges = edges
        self.datasplit = datasplit
        self.ratio = ratio

    def forward(self, data: HeteroData):
        num_pos = len(data["binds"].edge_label)

        if self.datasplit == "source":
            subgraph_src = data["binds"].edge_label_index[0].unique()
            global_src = data["source"].node_id[subgraph_src]

            subgraph_tgt = torch.cat(
                (
                    data["binds"].edge_index[1].unique().cpu(),
                    data["binds"].edge_label_index[1].unique().cpu(),
                ),
                dim=0,
            ).unique()
            global_tgt = data["target"].node_id[subgraph_tgt]

        elif self.datasplit == "target":
            subgraph_src = torch.cat(
                (
                    data["binds"].edge_index[0].unique().cpu(),
                    data["binds"].edge_label_index[0].unique().cpu(),
                ),
                dim=0,
            ).unique()
            global_src = data["source"].node_id[subgraph_src]

            subgraph_tgt = data["binds"].edge_label_index[1].unique()
            global_tgt = data["target"].node_id[subgraph_tgt]

        elif self.datasplit == "random":
            subgraph_src = torch.cat(
                (
                    data["binds"].edge_index[0].unique().cpu(),
                    data["binds"].edge_label_index[0].unique().cpu(),
                ),
                dim=0,
            ).unique()
            global_src = data["source"].node_id[subgraph_src]

            subgraph_tgt = torch.cat(
                (
                    data["binds"].edge_index[1].unique().cpu(),
                    data["binds"].edge_label_index[1].unique().cpu(),
                ),
                dim=0,
            ).unique()
            global_tgt = data["target"].node_id[subgraph_tgt]

        subgraph_src = subgraph_src.cpu().numpy()
        global_src = global_src.cpu().numpy()
        subgraph_tgt = subgraph_tgt.cpu().numpy()
        global_tgt = global_tgt.cpu().numpy()

        pos_edges = pd.MultiIndex.from_arrays(self.edges)

        # 3 chances to sample negative edges
        rng = np.random.default_rng(SEED)
        for _ in range(3):
            rnd_srcs = rng.choice(global_src, size=(num_pos * self.ratio * 2))
            rnd_tgts = rng.choice(global_tgt, size=(num_pos * self.ratio * 2))

            rnd_pairs = np.stack((rnd_srcs, rnd_tgts))
            rnd_pairs = np.unique(rnd_pairs, axis=1)
            rnd_pairs = pd.MultiIndex.from_arrays(rnd_pairs)
            inter = rnd_pairs.intersection(pos_edges, sort=False)
            neg_pairs = rnd_pairs.difference(inter, sort=False)

            if len([*neg_pairs]) < (num_pos * self.ratio):
                continue
            neg_pairs = rng.choice([*neg_pairs], num_pos * self.ratio, replace=False).T
            break

        else:
            raise RuntimeError("Could not successfully sample negatives.")

        # build dictionaries to map global edge indices to local (subgraph) indices
        source_map = dict(zip(pd.Series(global_src), pd.Series(subgraph_src)))
        target_map = dict(zip(pd.Series(global_tgt), pd.Series(subgraph_tgt)))

        neg_edges_srcs = pd.Series(neg_pairs[0]).map(source_map).values
        neg_edges_tgts = pd.Series(neg_pairs[1]).map(target_map).values

        new_labels = torch.cat(
            (
                data["binds"].edge_label.cpu(),
                torch.Tensor(np.zeros(num_pos * self.ratio)),
            )
        ).to(DEVICE)
        new_edges = (
            torch.cat(
                (
                    data["binds"].edge_label_index.cpu(),
                    torch.Tensor(np.array([neg_edges_srcs, neg_edges_tgts])),
                ),
                axis=1,
            )
            .type(torch.int32)
            .to(DEVICE)
        )

        data["binds"].edge_label = new_labels
        data["binds"].edge_label_index = new_edges

        return data
