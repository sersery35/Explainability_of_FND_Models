import math
from os import path
from utils.gnn_utils.enums import *
from utils.gnn_utils.data_loader import FNNDataset, DropEdge, ToUndirected
from utils.gnn_utils.eval_helper import *
import torch.nn.functional as F
import numpy as np
import torch
import shap

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader, DataListLoader, DenseDataLoader

PROJECT_DIR = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
DATA_DIR = PROJECT_DIR + '/data'


class GNNModelHelper(torch.nn.Module):
    """
    helper class for GNN models. aims to reduce code repetition
    """
    m_args = None
    m_hparams = None
    m_dataset_manager = None

    def __init__(self, model_args, model_hparams, model_dataset_manager):
        super(GNNModelHelper, self).__init__()
        self.m_args = model_args
        self.m_hparams = model_hparams
        self.m_dataset_manager = model_dataset_manager

    def get_optimizer(self):
        """
        extension method to handle initialization of optimizer easily
        """
        return torch.optim.Adam(self.parameters(), lr=self.m_hparams.lr, weight_decay=self.m_hparams.weight_decay)

    @torch.no_grad()
    def compute_test(self, loader, verbose=False):
        self.eval()
        loss_test = 0.0
        out_log = []
        for data in loader:
            if not self.m_args.multi_gpu:
                data = data.to(self.m_args.device)
            if self.m_hparams.model_type == GNNModelTypeEnum.GNNCL:
                out, _, _ = self(data.x, data.adj, data.mask)
            else:
                out = self(data)
            if self.m_args.multi_gpu:
                y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
            else:
                y = data.y
            if verbose:
                print(F.softmax(out, dim=1).cpu().numpy())
            out_log.append([F.softmax(out, dim=1), y])
            if self.m_hparams.model_type == GNNModelTypeEnum.GNNCL:
                loss_test += y.size(0) * F.nll_loss(out, y.view(-1)).item()
            else:
                loss_test += F.nll_loss(out, y).item()
        return eval_deep(out_log, loader), loss_test

    def train(self, mode=True):
        """
        extension method to train()
        """
        optimizer = self.get_optimizer()
        super(GNNModelHelper, self).train(mode=mode)
        if mode:
            for epoch in range(self.m_hparams.epochs):
                out_log = []
                loss_train = 0.0
                for i, data in enumerate(self.m_dataset_manager.train_loader):
                    optimizer.zero_grad()
                    if not self.m_args.multi_gpu:
                        data = data.to(self.m_args.device)

                    if self.m_hparams.model_type == GNNModelTypeEnum.GNNCL:
                        out, _, _ = self(data.x, data.adj, data.mask)
                    else:
                        out = self(data)

                    if self.m_args.multi_gpu:
                        y = torch.cat([d.y for d in data]).to(out.device)
                    else:
                        y = data.y

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
        [acc_val, _, _, _, recall_val, auc_val, _], loss_val = self.compute_test(
            self.m_dataset_manager.val_loader)
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

    def explain(self, background_data=None, test_data=None):
        background = background_data if background_data is not None else self.m_dataset_manager.get_train_samples()
        test = test_data if test_data is not None else self.m_dataset_manager.get_test_samples()
        explainer = shap.DeepExplainer(self, background)
        shap_values = explainer.shap_values(test)
        shap.image_plot(shap_values, test)


class ModelArguments:
    """
    handy class to save the defaults and initial setup for PyTorch
    """
    seed = None
    device = None
    multi_gpu = None

    def __init__(self, seed=777, device=DeviceTypeEnum.CPU, multi_gpu=False):
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
    _loader = None

    def __init__(self, hparam_manager, multi_gpu, root=DATA_DIR, empty=False):
        print(f"Loading data from directory: {root}")
        dataset = FNNDataset(root=root,
                             feature=hparam_manager.feature,
                             name=hparam_manager.dataset.value,
                             transform=hparam_manager.transform,
                             pre_transform=hparam_manager.pre_transform,
                             empty=empty)
        num_training = int(len(dataset) * 0.2)
        num_val = int(len(dataset) * 0.1)
        num_test = len(dataset) - (num_training + num_val)
        self.num_classes = dataset.num_classes
        self.num_features = dataset.num_features
        self.train_set, self.val_set, self.test_set = torch.utils.data.random_split(dataset,
                                                                                    [num_training, num_val, num_test])
        if multi_gpu:
            self._loader = DataListLoader
        else:
            self._loader = DataLoader
        if hparam_manager.model_type == GNNModelTypeEnum.GNNCL:
            self._loader = DenseDataLoader

        self.batch_size = hparam_manager.batch_size
        self.train_loader = self._loader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader = self._loader(self.val_set, batch_size=self.batch_size, shuffle=False)
        self.test_loader = self._loader(self.test_set, batch_size=self.batch_size, shuffle=False)

    def get_train_samples(self, len_samples=30):
        """
        return a subset of the train_loader for model explanation
        """
        train_samples = []
        for data in enumerate(self.train_loader.dataset):
            train_samples.append(data)
            if len(train_samples) == len_samples:
                break
        print(f"{len(train_samples)} training samples are fetched.")
        return train_samples

    def get_test_samples(self, len_samples=10):
        """
        return a subset of the test_loader for model explanation
        """
        test_samples = []
        for data in enumerate(self.test_loader.dataset):
            test_samples.append(data)
            if len(test_samples) == len_samples:
                break
        print(f"{len(test_samples)} test samples are fetched.")
        return test_samples
        # return self._loader(self.test_set[0:len_samples], batch_size=self.batch_size, shuffle=False)


class HparamManager:
    """
    manager class that generates different hparams for different models
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

    def __init__(self, model_type: GNNModelTypeEnum, test_mode=False):
        self.model_type = model_type
        self._load_for_model(model_type)
        if test_mode:
            self._set_epochs_for_test()

    def set_custom(self, dataset, batch_size, lr, weight_decay, n_hidden, dropout_rates, epochs, feature,
                   transform, concat):
        """
        set custom values regardless of the preset values
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_hidden = n_hidden
        self.dropout_rates = dropout_rates
        self.epochs = epochs
        self.feature = feature
        self.transform = transform
        self.concat = concat

    def _set_epochs_for_test(self):
        self.epochs = math.ceil(self.epochs / 3)

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
        self.feature = GNNFeatureTypeEnum.PROFILE.value
        self.transform = DropEdge(self.dropout_rates['TDdroprate'], self.dropout_rates['BUdroprate'])

    def _load_for_upfd_gcnfn(self):
        self._load_for_gcnfn()
        self.feature = GNNFeatureTypeEnum.SPACY.value
        self.concat = False

    def _load_for_vanilla_gcnfn(self):
        self._load_for_gcnfn()
        self.feature = GNNFeatureTypeEnum.CONTENT.value
        self.concat = True

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
        self.feature = GNNFeatureTypeEnum.BERT.value
        self.concat = True
        self.transform = ToUndirected()

    def _load_for_gnncl(self):
        self.dataset = GNNDatasetTypeEnum.POLITIFACT
        self.batch_size = 128
        self.lr = 0.001
        # self.weight_decay = 0.001 # set in args but never used in the original implementation
        self.n_hidden = 128
        self.epochs = 60
        self.feature = GNNFeatureTypeEnum.PROFILE.value
        self.max_nodes = 500  # 200 if self.dataset == GNNDatasetTypeEnum.GOSSIPCOP
        self.transform = T.ToDense(self.max_nodes)
        self.pre_transform = ToUndirected()
