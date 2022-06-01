"""
helper file to handle GNN model implementation in https://github.com/safe-graph/GNN-FakeNews
"""

import torch
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_max_pool
import torch.nn.functional as F

from GNNFakeNews.utils.enums import GNNModelTypeEnum
from GNNFakeNews.utils.helpers import GNNModelHelper


class GNNet(GNNModelHelper):
    """

    The GCN, GAT, and GraphSAGE implementation

    """

    def __init__(self, model_args, model_hparams, model_dataset_manager):
        super(GNNet, self).__init__(model_args, model_hparams, model_dataset_manager)
        num_features = self.m_dataset_manager.num_features
        n_hidden = self.m_hparams.n_hidden
        num_classes = self.m_dataset_manager.num_classes
        model_type = self.m_hparams.model_type

        if model_type == GNNModelTypeEnum.GCN_GNN:
            self.conv1 = GCNConv(num_features, n_hidden)
        elif model_type == GNNModelTypeEnum.SAGE_GNN:
            self.conv1 = SAGEConv(num_features, n_hidden)
        elif model_type == GNNModelTypeEnum.GAT_GNN:
            self.conv1 = GATConv(num_features, n_hidden)
        else:
            raise ValueError(f'Possible Values are {GNNModelTypeEnum.all_elements()}')

        if self.m_hparams.concat:
            self.lin0 = torch.nn.Linear(num_features, n_hidden)
            self.lin1 = torch.nn.Linear(n_hidden * 2, n_hidden)

        self.lin2 = torch.nn.Linear(n_hidden, num_classes)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        edge_attr = None

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = global_max_pool(x, batch)

        if self.m_hparams.concat:
            news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
            news = F.relu(self.lin0(news))
            x = torch.cat([x, news], dim=1)
            x = F.relu(self.lin1(x))

        x = F.log_softmax(self.lin2(x), dim=-1)

        return x
