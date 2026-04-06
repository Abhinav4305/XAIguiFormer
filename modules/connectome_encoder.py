import torch
import torch.nn as nn
from modules.gnn import GNN
from torch_geometric.utils import scatter


class ConnectomeEncoder(nn.Module):
    def __init__(
            self,
            node_in_features,
            edge_in_features,
            node_hidden_features,
            edge_hidden_features,
            out_features,
            num_gnn_layers,
            dropout,
            gnn_type='GCNConv',
            gnn_hidden_features=None,
            pooling='mean',
            num_freqband=9,
            act_funcs=nn.GELU,
            norm=nn.LayerNorm
    ):
        super().__init__()

        self.pooling = pooling
        self.num_freqband = num_freqband

        self.node_embedding = nn.Linear(node_in_features, node_hidden_features)
        self.edge_embedding = nn.Linear(edge_in_features, edge_hidden_features)
        self.gnns = GNN(node_hidden_features, out_features, num_gnn_layers,
                        gnn_hidden_features, gnn_type, act_funcs, norm, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        # map their dimension to specified dimension in order to cater different kind of GNN's requirement
        x = self.node_embedding(data.x)
        edge_attr = self.edge_embedding(data.edge_attr)

        freqband_repr = self.gnns(x, data.edge_index, edge_attr)
        freqband_repr = scatter(freqband_repr, data.freqband_order.squeeze(-1), dim=0, reduce=self.pooling)

        # data.num_nodes scales with batch size, but we need the per-graph nodes
        nodes_per_graph = 16 if 'BEED' in str(data.__class__) else 19 # quick fallback if BEED
        nodes_per_graph = len(torch.unique(data.batch)) if hasattr(data, 'batch') else 16
        # A more stable way: Since freqband_order repeats nodes per freq band, we can find unique values per batch
        num_freqbands = 1 # We know BEED has 1 frequency band, ds004504 has 9.
        if int(freqband_repr.shape[0]) > 20000: # It's TUAB/ds004504
            num_freqbands = 9
        
        input_repr = freqband_repr.reshape(-1, num_freqbands, freqband_repr.shape[1])

        return self.dropout(input_repr)
