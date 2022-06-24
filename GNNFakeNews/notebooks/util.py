import matplotlib.pyplot as plt
import torch
import numpy as np

from GNNFakeNews.utils.enums import GNNModelTypeEnum
from GNNFakeNews.models import gnn, gcnfn, bigcn, gnncl
from GNNFakeNews.utils.helpers import ModelArguments, HparamFactory, GNNDatasetManager


def run_model(model_type: GNNModelTypeEnum, test_mode=False, return_dataset_manager=False, local_load=True,
              hyperparams=None):
    """
    method is a convenient wrapper to initialize, train then evaluate the model
    """
    args = ModelArguments()
    hparams = HparamFactory(model_type, test_mode=test_mode) if hyperparams is None else hyperparams
    dataset_manager = GNNDatasetManager(local_load=local_load, hparam_manager=hparams, multi_gpu=args.multi_gpu)
    if model_type == GNNModelTypeEnum.BIGCN:
        model = bigcn.BiGCNet(model_args=args,
                              model_hparams=hparams,
                              model_dataset_manager=dataset_manager)
    elif model_type in [GNNModelTypeEnum.UPFD_GCNFN, GNNModelTypeEnum.VANILLA_GCNFN]:
        model = gcnfn.GCNFNet(model_args=args,
                              model_hparams=hparams,
                              model_dataset_manager=dataset_manager)
    elif model_type in [GNNModelTypeEnum.GCN_GNN, GNNModelTypeEnum.GAT_GNN, GNNModelTypeEnum.SAGE_GNN]:
        model = gnn.GNNet(model_args=args,
                          model_hparams=hparams,
                          model_dataset_manager=dataset_manager)
    elif model_type == GNNModelTypeEnum.GNNCL:
        model = gnncl.GNNCLNet(model_args=args,
                               model_hparams=hparams,
                               model_dataset_manager=dataset_manager)
    else:
        raise ValueError(f'Options are {GNNModelTypeEnum.all_elements()}')

    model.train_then_eval()

    if return_dataset_manager:
        return model, dataset_manager
    return model


def visualize_label_dist(labels: torch.tensor):
    labels_np = labels.cpu().numpy()
    unique_labels = np.unique(labels_np)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    for unique_label in unique_labels:
        occurence_count = len(np.where(labels_np == unique_label)[0])
        ax.bar_label(ax.bar(unique_label, occurence_count))

    ax.set_xticks(unique_labels)
    plt.show()


def visualize_edge_mask(edge_mask: torch.tensor):
    edge_mask_np = edge_mask.cpu().numpy()
    plt.figure(figsize=(8, 15))
    indexes = np.arange(0, len(edge_mask_np))
    plt.scatter(x=indexes, y=edge_mask_np)
    plt.show()
