"""
helper file to handle GCNFN model implementation in https://github.com/safe-graph/GNN-FakeNews
"""

import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool, GATConv

from GNNFakeNews.utils.data_loader import *
from GNNFakeNews.utils.helpers import GNNModelHelper


class GCNFNet(GNNModelHelper):
    """

    GCNFN is implemented using two GCN layers and one mean-pooling layer as the graph encoder;
    the 310-dimensional node feature (args.feature = content) is composed of 300-dimensional
    comment word2vec (spaCy) embeddings plus 10-dimensional profile features

    Paper: Fake News Detection on Social Media using Geometric Deep Learning
    Link: https://arxiv.org/pdf/1902.06673.pdf


    Model Configurations:

    Vanilla GCNFN: args.concat = False, args.feature = content
    UPFD-GCNFN: args.concat = True, args.feature = spacy

    """

    def __init__(self, model_args, model_hparams, model_dataset_manager, verbose):
        super(GCNFNet, self).__init__(model_args, model_hparams, model_dataset_manager, verbose)

        num_features = self.m_dataset_manager.num_features
        num_classes = self.m_dataset_manager.num_classes

        n_hidden = self.m_hparams.n_hidden

        self.conv1 = GATConv(num_features, n_hidden * 2)
        self.conv2 = GATConv(n_hidden * 2, n_hidden * 2)

        self.fc1 = Linear(n_hidden * 2, n_hidden)

        if self.m_hparams.concat:
            self.fc0 = Linear(num_features, n_hidden)
            self.fc1 = Linear(n_hidden * 2, n_hidden)

        self.fc2 = Linear(n_hidden, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.selu(self.conv1(x, edge_index))
        x = F.selu(self.conv2(x, edge_index))
        x = F.selu(global_mean_pool(x, batch))
        x = F.selu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)

        if self.m_hparams.concat:
            news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
            news = F.relu(self.fc0(news))
            x = torch.cat([x, news], dim=1)
            x = F.relu(self.fc1(x))

        x = F.log_softmax(self.fc2(x), dim=-1)

        return x
