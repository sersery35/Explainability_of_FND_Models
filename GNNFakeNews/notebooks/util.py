import matplotlib.pyplot as plt
import torch
import numpy as np
from torch_geometric.nn import GNNExplainer

from GNNFakeNews.utils.enums import GNNModelTypeEnum
from GNNFakeNews.models import gnn, gcnfn, bigcn, gnncl
from GNNFakeNews.utils.helpers import ModelArguments, HparamFactory, GNNDatasetManager


def run_model(model_type: GNNModelTypeEnum, test_mode=False, return_dataset_manager=True, local_load=True,
              hparams=None, verbose=False):
    """
    method is a convenient wrapper to initialize, train then evaluate the model
    """
    args = ModelArguments()
    model_hparams = HparamFactory(model_type, test_mode=test_mode) if hparams is None else hparams
    dataset_manager = GNNDatasetManager(local_load=local_load, hparam_manager=model_hparams, multi_gpu=args.multi_gpu)
    if model_type == GNNModelTypeEnum.BIGCN:
        model = bigcn.BiGCNet(model_args=args,
                              model_hparams=model_hparams,
                              model_dataset_manager=dataset_manager, verbose=verbose)
    elif model_type in [GNNModelTypeEnum.UPFD_GCNFN, GNNModelTypeEnum.VANILLA_GCNFN]:
        model = gcnfn.GCNFNet(model_args=args,
                              model_hparams=model_hparams,
                              model_dataset_manager=dataset_manager, verbose=verbose)
    elif model_type in [GNNModelTypeEnum.GCN_GNN, GNNModelTypeEnum.GAT_GNN, GNNModelTypeEnum.SAGE_GNN]:
        model = gnn.GNNet(model_args=args,
                          model_hparams=model_hparams,
                          model_dataset_manager=dataset_manager, verbose=verbose)
    elif model_type == GNNModelTypeEnum.GNNCL:
        model = gnncl.GNNCLNet(model_args=args,
                               model_hparams=model_hparams,
                               model_dataset_manager=dataset_manager, verbose=verbose)
    else:
        raise ValueError(f'Options are {GNNModelTypeEnum.all_elements()}')

    model.train_then_eval()

    if return_dataset_manager:
        return model, dataset_manager
    return model


def visualize_label_distribution(labels: torch.tensor):
    labels_np = labels.cpu().numpy()
    unique_labels = np.unique(labels_np)

    fig = plt.figure(figsize=(5, 5))

    ax = fig.add_axes([0, 0, 1, 1])
    for unique_label in unique_labels:
        occurrence_count = len(np.where(labels_np == unique_label)[0])
        ax.bar_label(ax.bar(unique_label, occurrence_count))

    plt.title('Label distribution')
    ax.set_xticks(unique_labels)

    plt.show()


def visualize_edge_mask(edge_mask: torch.tensor):
    edge_mask_np = edge_mask.cpu().numpy()
    plt.figure(figsize=(8, 15))

    indexes = np.arange(0, len(edge_mask_np))
    plt.scatter(x=indexes, y=edge_mask_np)
    plt.title('Edge mask distribution')
    plt.show()


class GNNModelExplainer:
    def __init__(self, model, sample_data, visualize_explaining_graph=True, visualize_label_dist=True,
                 visualize_edge_mask_dist=False):

        # pick the root node since it is the news itself, all leaf nodes are the users who shared this news
        self.node_idx = 0

        x, edge_index, batch, num_graphs = sample_data.x, sample_data.edge_index, sample_data.batch, sample_data.num_graphs
        self.gnn_explainer = GNNExplainer(model, epochs=200).to(model.m_args.device)
        self.node_feat_mask, self.edge_mask = self.gnn_explainer.explain_graph(x=x, edge_index=edge_index)
        # print(f'x.size: {x.size()}\nedge_index.size: {edge_index.size()}\nnum_graphs: {num_graphs}\ny.size: {sample_data.y.size()}')

        if visualize_explaining_graph:
            plt.figure(figsize=(20, 20))
            ax, self.subgraph = self.gnn_explainer.visualize_subgraph(node_idx=self.node_idx,
                                                                      edge_index=edge_index.cpu(),
                                                                      edge_mask=self.edge_mask.cpu(),
                                                                      # y=sample_data.y,
                                                                      node_size=600, font_size=15)
            plt.show()

        if visualize_label_dist:
            visualize_label_distribution(sample_data.y)

        if visualize_edge_mask_dist:
            visualize_edge_mask(self.edge_mask)
