"""
helper file for loading the models in https://github.com/safe-graph/GNN-FakeNews
"""

import math
from os import path
from typing import Union

from GNNFakeNews.utils.enums import *
from GNNFakeNews.utils.data_loader import FNNDataset, DropEdge, ToUndirected
from GNNFakeNews.utils.eval_helper import *

import torch.nn.functional as F
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader, DataListLoader, DenseDataLoader
from torch_geometric.datasets import UPFD
import torch_geometric.data
from torch_geometric.nn import GNNExplainer
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

PROJECT_DIR = path.dirname(path.dirname(path.abspath(__file__)))
DATA_DIR = PROJECT_DIR + '/data'

LOCAL_DATA_FOLDER = 'local'
REMOTE_DATA_FOLDER = 'remote'


class GNNModelHelper(torch.nn.Module):
    """
    helper class for GNN models. aims to reduce code repetition
    """
    m_args = None
    m_hparams = None
    m_dataset_manager = None

    def __init__(self, model_args, model_hparams, model_dataset_manager, verbose=True):
        super(GNNModelHelper, self).__init__()
        self.m_args = model_args
        self.m_hparams = model_hparams
        self.m_dataset_manager = model_dataset_manager
        self.verbose = verbose

    def get_optimizer(self):
        """
        extension method to handle initialization of optimizer easily
        """
        return torch.optim.Adam(self.parameters(), lr=self.m_hparams.lr, weight_decay=self.m_hparams.weight_decay)

    def m_handle_train(self, data):
        if not self.m_args.multi_gpu:
            data = data.to(self.m_args.device)
        if self.m_hparams.model_type == GNNModelTypeEnum.GNNCL:
            out, _, _ = self(data.x, data.adj, data.mask)
        elif self.m_hparams.model_type == GNNModelTypeEnum.BIGCN:
            out = self(data.x, data.edge_index, data.batch, data.BU_edge_index, data.root_index)
        else:
            out = self(data.x, data.edge_index, data.batch)
        if self.m_args.multi_gpu:
            y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
        else:
            y = data.y
        return out, y

    @torch.no_grad()
    def compute_test(self, loader, verbose=False):
        self.eval()
        loss_test = 0.0
        out_log = []
        for data in loader:
            out, y = self.m_handle_train(data)
            if verbose:
                print(F.softmax(out, dim=1).cpu().numpy())

            out_log.append([F.softmax(out, dim=1), y])

            if self.m_hparams.model_type == GNNModelTypeEnum.GNNCL:
                loss_test += y.size(0) * F.nll_loss(out, y.view(-1)).item()
            else:
                loss_test += F.nll_loss(out, y).item()
        return eval_deep(out_log, loader), loss_test

    def train_then_eval(self):
        """
        extension method to train()
        """
        self.to(self.m_args.device)
        optimizer = self.get_optimizer()
        self.train()
        for epoch in range(self.m_hparams.epochs):
            out_log = []
            loss_train = 0.0
            for i, data in enumerate(self.m_dataset_manager.train_loader):
                optimizer.zero_grad()
                out, y = self.m_handle_train(data)

                if self.m_hparams.model_type == GNNModelTypeEnum.GNNCL:
                    loss = F.nll_loss(out, y.view(-1))
                else:
                    loss = F.nll_loss(out, y)

                loss.backward()
                optimizer.step()

                if self.m_hparams.model_type == GNNModelTypeEnum.GNNCL:
                    loss_train += data.y.size(0) * loss.item()
                else:
                    loss_train += loss.item()

                out_log.append([F.softmax(out, dim=1), y])
            self.m_evaluate_train(out_log, epoch, loss_train)
        self.m_evaluate_test()

    def m_evaluate_train(self, out_log, epoch, loss_train):
        acc_train, _, _, _, recall_train, auc_train, _ = eval_deep(out_log, self.m_dataset_manager.train_loader)
        [acc_val, _, _, _, recall_val, auc_val, _], loss_val = self.compute_test(self.m_dataset_manager.val_loader)

        if self.verbose:
            print(f'\n************** epoch: {epoch} **************'
                  f'\nloss_train: {loss_train:.4f}, acc_train: {acc_train:.4f},'
                  f'\nrecall_train: {recall_train:.4f}, auc_train: {auc_train:.4f},'
                  f'\nloss_val: {loss_val:.4f}, acc_val: {acc_val:.4f},'
                  f'\nrecall_val: {recall_val:.4f}, auc_val: {auc_val:.4f}'
                  '\n***************************************')

    def m_evaluate_test(self):
        [acc, f1_macro, f1_micro, precision, recall, auc, ap], test_loss = self.compute_test(
            self.m_dataset_manager.test_loader, verbose=False)
        print(f'Test set results: acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, f1_micro: {f1_micro:.4f},'
              f'precision: {precision:.4f}, recall: {recall:.4f}, auc: {auc:.4f}, ap: {ap:.4f}')


class ModelArguments:
    """
    handy class to save the defaults and initial setup for PyTorch
    """
    seed = None
    device = None
    multi_gpu = None

    def __init__(self, seed=777, device=DeviceTypeEnum.GPU, multi_gpu=False):
        self.seed = seed
        if not torch.cuda.is_available() and device == DeviceTypeEnum.GPU:
            raise ValueError(f'device cannot be {DeviceTypeEnum.GPU}, because CUDA is not available.')
        self.device = torch.device(device.value)
        self.multi_gpu = multi_gpu
        self.setup()

    def setup(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)


class GNNDatasetManager:
    """
    Manager class that handles dataset pipeline for GNN models
    """
    train_set = None
    train_loader = None
    val_set = None
    val_loader = None
    test_set = None
    test_loader = None
    num_classes = None
    num_features = None
    batch_size = None

    def __init__(self, local_load=True, hparam_manager=None, multi_gpu=False, root=DATA_DIR, empty=False):
        # first determine the loader
        if multi_gpu:
            loader = DataListLoader
        else:
            loader = DataLoader
        # for GNNCL we use a different loader
        if hparam_manager.model_type == GNNModelTypeEnum.GNNCL:
            loader = DenseDataLoader

        # we can either manually download datasets under politifact and gossipcop folders
        if local_load:
            self.local_load(hparam_manager, root, empty)
        else:
            self.remote_load(hparam_manager, root, empty)

        self.batch_size = hparam_manager.batch_size

        self.train_loader = loader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader = loader(self.val_set, batch_size=self.batch_size, shuffle=False)
        self.test_loader = loader(self.test_set, batch_size=self.batch_size, shuffle=False)

    def local_load(self, hparam_manager, root, empty):
        """
        method loads dataset from a local file
        Parameters
        ----------
        hparam_manager: HparamFactory,
            The HParamFactory instance. Used to load the dataset with default hyperparameters
        root: str,
            the data directory
        empty: bool,
            the empty parameter to be passed to FNNDataset
        """
        root = path.join(root, LOCAL_DATA_FOLDER)
        print(f"Loading dataset '{hparam_manager.dataset.value}' from directory: {root}")
        dataset = FNNDataset(root=root, feature=hparam_manager.feature.value, name=hparam_manager.dataset.value,
                             transform=hparam_manager.transform, pre_transform=hparam_manager.pre_transform,
                             empty=empty)

        self.num_classes = dataset.num_classes
        self.num_features = dataset.num_features

        num_training = int(len(dataset) * 0.2)
        num_val = int(len(dataset) * 0.1)
        num_test = len(dataset) - (num_training + num_val)
        self.train_set, self.val_set, self.test_set = torch.utils.data.random_split(dataset,
                                                                                    [num_training, num_val, num_test])

    def remote_load(self, hparam_manager, root):
        """
        method loads dataset from the repository
        Parameters
        ----------
        hparam_manager: HparamFactory,
            The HParamFactory instance. Used to load the dataset with default hyperparameters
        root: str,
            the data directory
        """
        # load the dataset from torch_geometric.datasets and follow:
        # https://github.com/pyg-team/pytorch_geometric/blob/master/examples/upfd.py
        root = path.join(root, REMOTE_DATA_FOLDER)
        print(f"Loading data into the directory: {root}")

        self.train_set = UPFD(root=root, name=hparam_manager.dataset.value, feature=hparam_manager.feature,
                              split='train', transform=hparam_manager.transform,
                              pre_transform=hparam_manager.pre_transform)

        self.num_classes = self.train_set.num_classes
        self.num_features = self.train_set.num_features

        self.val_set = UPFD(root=root, name=hparam_manager.dataset.value, feature=hparam_manager.feature, split='val',
                            transform=hparam_manager.transform, pre_transform=hparam_manager.pre_transform)
        self.test_set = UPFD(root=root, name=hparam_manager.dataset.value, feature=hparam_manager.feature,
                             split='test', transform=hparam_manager.transform,
                             pre_transform=hparam_manager.pre_transform)

    @staticmethod
    def get_random_samples(loader: torch_geometric.data.DataLoader, device: torch.device, len_samples=1):
        """
        randomly select torch_geometric.data.Data instances from loader and return these instances as a list
        Parameters
        ----------
        loader: torch_geometric.data.DataLoader,
            the loader to be used for random sampling
        device: torch.device,
            the device to put data in
        len_samples: int,
            the length of samples to be returned
        """
        assert len_samples <= len(loader.dataset)
        samples = []
        indexes = np.random.random_integers(0, len(loader.dataset) - 1, len_samples)
        for i, data in enumerate(loader.dataset):
            if i in indexes:
                samples.append(data.to(device))
            # early stopping if necessary
            if len(samples) == len(indexes):
                break
        return samples

    def get_random_train_samples(self, device: torch.device, len_samples=1):
        """
        return a subset of the train_loader for model explanation
        Parameters
        ----------
        device: torch.device,
            the device to put data in
        len_samples: int,
            the length of samples to be returned
        """
        return self.get_random_samples(self.train_loader, device, len_samples)

    def get_random_val_samples(self, device: torch.device, len_samples=1):
        """
        return a subset of the val_loader for model explanation
        Parameters
        ----------
        device: torch.device,
            the device to put data in
        len_samples: int,
            the length of samples to be returned
        """
        return self.get_random_samples(self.val_loader, device, len_samples)

    def get_test_samples(self, device: torch.device, len_samples=1):
        """
        return a subset of the test_loader for model explanation
        Parameters
        ----------
        device: torch.device,
            the device to put data in
        len_samples: int,
            the length of samples to be returned
        """
        return self.get_random_samples(self.test_loader, device, len_samples)


class HparamFactory:
    """
    factory class that generates different hparams for different models
    """
    model_type = None
    dataset = None
    batch_size = None
    lr = None
    weight_decay = None
    n_hidden = None
    dropout_rates = None
    epochs = None
    feature = None
    transform = None
    pre_transform = None
    concat = None
    max_nodes = None

    def __init__(self, model_type: GNNModelTypeEnum, test_mode=False, **kwargs):
        self.model_type = model_type
        self._load_for_model(model_type)
        if test_mode:
            self._set_epochs_for_test()

        for key in self.__dict__.keys():
            if key in kwargs.keys():
                value = kwargs.pop(key, None)
                setattr(self, key, value)

        if self.dataset == GNNDatasetTypeEnum.GOSSIPCOP and model_type == GNNModelTypeEnum.GNNCL:
            self.max_nodes = 200

        print('#################################')
        print('-----> The hyperparameters are set!')
        for key in self.__dict__.keys():
            print(f'{key} = {getattr(self, key)}')
        print('#################################')

    def _set_epochs_for_test(self):
        self.epochs = math.ceil(self.epochs / 5)

    def _load_for_model(self, model_type: GNNModelTypeEnum):
        """
        Given a model type, method returns the initialized class instance
        """
        if model_type == GNNModelTypeEnum.BIGCN:
            self._load_for_bigcn()
        elif model_type == GNNModelTypeEnum.VANILLA_GCNFN:
            self._load_for_vanilla_gcnfn()
        elif model_type == GNNModelTypeEnum.UPFD_GCNFN:
            self._load_for_upfd_gcnfn()
        elif model_type in [GNNModelTypeEnum.GCN_GNN, GNNModelTypeEnum.GAT_GNN, GNNModelTypeEnum.SAGE_GNN]:
            self._load_for_gnn()
        elif model_type == GNNModelTypeEnum.GNNCL:
            self._load_for_gnncl()
        else:
            raise ValueError(f'Possible values are {GNNModelTypeEnum.all_elements()}')

    def _load_for_bigcn(self):
        self.dataset = GNNDatasetTypeEnum.POLITIFACT
        self.batch_size = 128
        self.lr = 0.01
        self.weight_decay = 0.001
        self.n_hidden = 128
        self.dropout_rates = {
            'TDdroprate': 0.2,
            'BUdroprate': 0.2,
        }
        self.epochs = 45
        self.feature = GNNFeatureTypeEnum.PROFILE
        self.transform = DropEdge(self.dropout_rates['TDdroprate'], self.dropout_rates['BUdroprate'])

    def _load_for_upfd_gcnfn(self):
        self._load_for_gcnfn()
        self.feature = GNNFeatureTypeEnum.SPACY
        self.concat = True

    def _load_for_vanilla_gcnfn(self):
        self._load_for_gcnfn()
        self.feature = GNNFeatureTypeEnum.CONTENT
        self.concat = False

    def _load_for_gcnfn(self):
        self.dataset = GNNDatasetTypeEnum.POLITIFACT
        self.batch_size = 128
        self.lr = 0.01
        self.weight_decay = 0.001
        self.n_hidden = 128
        self.epochs = 60
        self.transform = ToUndirected()

    def _load_for_gnn(self):
        self.dataset = GNNDatasetTypeEnum.POLITIFACT
        self.batch_size = 128
        self.lr = 0.01
        self.weight_decay = 0.01
        self.n_hidden = 128
        self.epochs = 35
        self.feature = GNNFeatureTypeEnum.BERT
        self.concat = True
        self.transform = ToUndirected()

    def _load_for_gnncl(self):
        self.dataset = GNNDatasetTypeEnum.POLITIFACT
        self.batch_size = 128
        self.lr = 0.001
        # self.weight_decay = 0.001 # set in args but never used in the original implementation
        self.n_hidden = 128
        self.epochs = 60
        self.feature = GNNFeatureTypeEnum.PROFILE
        self.max_nodes = 500
        self.transform = T.ToDense(self.max_nodes)
        self.pre_transform = ToUndirected()


class GNNModelExplainer:
    def __init__(self, model: torch.nn.Module,
                 sample_data: Union[torch_geometric.data.Data, torch_geometric.data.batch.Batch]):
        """
        class is a manager for explanation pipeline. When initialized, it will explain the model with sample_data
        Parameters
        ----------
        model: torch.nn.Module,
            the model to be explained
        sample_data:  Union[torch_geometric.data.Data, torch_geometric.data.batch.DataBatch]
            the graph data to be explained
        """
        self.subgraph = None
        self.adjacency_matrix = None
        # pick the root node since it is the news itself, all leaf nodes are the users who shared this news
        self.node_idx = 0
        self.sample_data = sample_data

        self.gnn_explainer = GNNExplainer(model, epochs=200).to(model.m_args.device)
        self.node_feat_mask, self.edge_mask = self.gnn_explainer.explain_graph(x=sample_data.x,
                                                                               edge_index=sample_data.edge_index)

    def visualize_explaining_graph(self, threshold=None):
        """
        visualize the subgraph obtained from the GNNExplainer using the edge mask.
        Parameters
        ----------
        threshold: float,
            the threshold value for which edge masks to use when visualizing. helps to visualize better
        """
        plt.figure(figsize=(15, 15))
        print(f'y: {self.sample_data.y.cpu()}')
        threshold = torch.mean(self.edge_mask).cpu() if threshold is None else threshold
        indexes = self.edge_mask > threshold
        ax, self.subgraph = self.gnn_explainer.visualize_subgraph(node_idx=self.node_idx,
                                                                  edge_index=self.sample_data.edge_index[:,
                                                                             indexes].cpu(),
                                                                  edge_mask=self.edge_mask[indexes].cpu(),
                                                                  # y=sample_data.y.cpu(),
                                                                  threshold=threshold,
                                                                  node_size=600, font_size=15)
        plt.show()

    def visualize_label_dist(self):
        """
        visualize the label distribution of the given batch
        """
        labels_np = self.sample_data.y.cpu().numpy()
        if len(labels_np) == 1:
            print(f'There only one sample with label: {labels_np}')
            return

        unique_labels = np.unique(labels_np)

        fig = plt.figure(figsize=(5, 5))

        ax = fig.add_axes([0, 0, 1, 1])
        for unique_label in unique_labels:
            occurrence_count = len(np.where(labels_np == unique_label)[0])
            ax.bar_label(ax.bar(unique_label, occurrence_count))

        plt.title('Label distribution')
        ax.set_xticks(unique_labels)

        plt.show()

    def visualize_edge_mask_dist(self):
        """
        scatter plot the edge mask obtained from GNNExplainer
        """
        edge_mask_np = self.edge_mask.cpu().numpy()
        plt.figure(figsize=(8, 15))

        indexes = np.arange(0, len(edge_mask_np))
        plt.scatter(x=indexes, y=edge_mask_np)
        plt.title('Edge mask distribution')
        plt.show()

    def visualize_adjacency_matrix(self):
        """
        show a grayscale image representing the adjacency matrix
        """
        self.adjacency_matrix = nx.to_pandas_adjacency(self.subgraph)
        fig, ax = plt.subplots(figsize=(12, 8))

        # Full adjacency
        ax.set_title('Full Adjacency mask')
        ax.imshow(self.adjacency_matrix, cmap='gray')
        plt.show()
