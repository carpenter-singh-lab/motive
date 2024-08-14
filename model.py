import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import MLP as tMLP
from torch_geometric.nn import GATv2Conv, GINConv, SAGEConv, to_hetero

from motive import get_counts
from utils.utils import PathLocator


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = SAGEConv(hidden_channels, hidden_channels, normalize=True)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, normalize=True)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # Define a 2-layer GNN computation graph.
        h1 = F.leaky_relu(self.conv1(x, edge_index))
        h2 = self.conv2(h1, edge_index)
        h3 = h1 + h2
        return h3


class GIN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.mlp1 = tMLP([hidden_channels, hidden_channels, hidden_channels])
        self.mlp2 = tMLP([hidden_channels, hidden_channels, hidden_channels])
        self.conv1 = GINConv(self.mlp1)
        self.conv2 = GINConv(self.mlp2)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # Define a 2-layer GNN computation graph.
        h1 = F.leaky_relu(self.conv1(x, edge_index))
        h2 = self.conv2(h1, edge_index)
        h3 = h1 + h2
        return h3


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = GATv2Conv(
            hidden_channels,
            hidden_channels // 2,
            heads=2,
            add_self_loops=False,
            dropout=0.3,
        )
        self.conv2 = GATv2Conv(
            hidden_channels,
            hidden_channels // 2,
            heads=2,
            add_self_loops=False,
            dropout=0.3,
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # Define a 2-layer GNN computation graph.
        h1 = F.leaky_relu(self.conv1(x, edge_index))
        h2 = self.conv2(h1, edge_index)
        h3 = h1 + h2
        return h3


# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(
        self, x_source: Tensor, x_target: Tensor, edge_label_index: Tensor
    ) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_source = x_source[edge_label_index[0]]
        edge_feat_target = x_target[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_source * edge_feat_target).sum(dim=-1)


class GraphSAGE_Embs(torch.nn.Module):
    def __init__(
        self, hidden_channels, num_source_nodes, num_target_nodes, data, GNNClass
    ):
        super().__init__()

        # embedding matrices for sources and targets:
        self.source_emb = torch.nn.Embedding(num_source_nodes, hidden_channels)
        self.target_emb = torch.nn.Embedding(num_target_nodes, hidden_channels)

        # Instantiate homogeneous GNN:
        self.gnn = GNNClass(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        metadata = data.metadata()
        self.gnn = to_hetero(self.gnn, metadata=metadata)

        # Instantiate one of the classifier classes
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "source": self.source_emb(data["source"].node_id),
            "target": self.target_emb(data["target"].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)

        pred = self.classifier(
            x_dict["source"],
            x_dict["target"],
            data["source", "binds", "target"].edge_label_index,
        )
        return pred


# Child of our GNN model that initializes embedding weights with
# cp features but freezes embeddings throughout training
class GraphSAGE_CP(GraphSAGE_Embs):
    def __init__(
        self, hidden_channels, num_source_nodes, num_target_nodes, data, GNNClass
    ):
        super().__init__(
            hidden_channels, num_source_nodes, num_target_nodes, data, GNNClass
        )
        src_weights = data["source"].x
        tgt_weights = data["target"].x
        source_size = data["source"].x.shape[1]
        target_size = data["target"].x.shape[1]

        self.source_emb = torch.nn.Sequential(
            torch.nn.Embedding(
                num_source_nodes, source_size, _weight=src_weights, _freeze=True
            ),
            torch.nn.Linear(source_size, hidden_channels),
            torch.nn.ReLU(),
        )

        self.target_emb = torch.nn.Sequential(
            torch.nn.Embedding(
                num_target_nodes, target_size, _weight=tgt_weights, _freeze=True
            ),
            torch.nn.Linear(target_size, hidden_channels),
            torch.nn.ReLU(),
        )


class MLP(torch.nn.Module):
    def __init__(self, source_size, target_size, hidden_size):
        super().__init__()

        self.dense_source = torch.nn.Linear(source_size, hidden_size)
        self.dense_target = torch.nn.Linear(target_size, hidden_size)
        source_size = target_size = hidden_size
        self.bilinear = torch.nn.Bilinear(source_size, target_size, 1)

    def forward(self, data: HeteroData) -> Tensor:
        source_ix = data["binds"]["edge_label_index"][0]
        target_ix = data["binds"]["edge_label_index"][1]
        x_source = data["source"].x[source_ix]
        x_target = data["target"].x[target_ix]
        h_source = F.relu(self.dense_source(x_source))
        h_target = F.relu(self.dense_target(x_target))
        logits = self.bilinear(h_source, h_target)
        return torch.squeeze(logits)


class Bilinear(torch.nn.Module):
    def __init__(self, source_size, target_size):
        super().__init__()
        self.bilinear = torch.nn.Bilinear(source_size, target_size, 1)

    def forward(self, data: HeteroData) -> Tensor:
        source_ix = data["binds"]["edge_label_index"][0]
        target_ix = data["binds"]["edge_label_index"][1]
        x_source = data["source"].x[source_ix]
        x_target = data["target"].x[target_ix]
        logits = self.bilinear(x_source, x_target)
        return torch.squeeze(logits)


class Cosine(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

    def forward(self, data: HeteroData) -> Tensor:
        source_ix = data["binds"]["edge_label_index"][0]
        target_ix = data["binds"]["edge_label_index"][1]
        x_source = data["source"].x[source_ix]
        x_target = data["target"].x[target_ix]
        logits = self.cos(x_source, x_target)
        return logits


def create_model(locator: PathLocator, data):
    model_name = locator.config["model_name"]
    num_sources, num_targets, num_features = get_counts(data)

    if model_name in ("gnn", "gat", "gin"):
        GNNClass = {"gnn": GNN, "gat": GAT, "gin": GIN}.get(model_name)
        initialization = locator.config["initialization"]
        if initialization == "cp":
            model = GraphSAGE_CP(
                int(locator.config["hidden_channels"]),
                num_sources,
                num_targets,
                data,
                GNNClass,
            )
        elif initialization == "embs":
            model = GraphSAGE_Embs(
                int(locator.config["hidden_channels"]),
                num_sources,
                num_targets,
                data,
                GNNClass,
            )
    elif model_name == "mlp":
        model = MLP(
            num_features["source"],
            num_features["target"],
            hidden_size=int(locator.config["hidden_channels"]),
        )
    elif model_name == "bilinear":
        model = Bilinear(
            num_features["source"],
            num_features["target"],
        )
    elif model_name == "cosine":
        model = Cosine()
    else:
        raise ValueError(f"Invalid model_name: {model_name}")
    return model
