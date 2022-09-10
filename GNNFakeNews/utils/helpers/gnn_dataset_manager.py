from os import path
from typing import Union

import torch
from torch_geometric.loader import DataLoader, DataListLoader, DenseDataLoader
from torch_geometric.datasets import UPFD
import torch_geometric.data
import numpy as np

from GNNFakeNews.utils.data_loader import FNNDataset
from GNNFakeNews.utils.enums import GNNModelTypeEnum

PROJECT_DIR = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
DATA_DIR = PROJECT_DIR + '/data'

LOCAL_DATA_FOLDER = 'local'
REMOTE_DATA_FOLDER = 'remote'


class GNNDatasetManager:
    """
    Manager class that handles dataset pipeline for GNN deprecated
    """

    def __init__(self, local_load=True, hparam_manager=None, multi_gpu=False, root=DATA_DIR, empty=False):
        """
        Parameters
        ----------
        local_load: bool,
            whether to load from local file or load from a remote file. defaults to True
        hparam_manager: HparamFactory,
            The HParamFactory instance. Used to load the dataset with default hyperparameters. defaults to None
        multi_gpu: bool,
            whether to use multi gpu on the device. defaults to False.
        root: str,
            the data directory, defaults to DATA_DIR which is defined in this file according to the project tree.
        empty: bool,
            the empty parameter to be passed to FNNDataset
        """
        # first determine the loader
        self.num_classes = None
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.train_loader = None
        self.num_features = None

        if multi_gpu:
            self.loader = DataListLoader
        else:
            self.loader = DataLoader
        # for GNNCL we use a different loader
        if hparam_manager.model_type == GNNModelTypeEnum.GNNCL:
            self.loader = DenseDataLoader

        # we can either manually download datasets under politifact and gossipcop folders
        if local_load:
            self.local_load(hparam_manager, root, empty)
        else:
            self.remote_load(hparam_manager, root, empty)

        self.batch_size = hparam_manager.batch_size

        self.train_loader = self.loader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader = self.loader(self.val_set, batch_size=self.batch_size, shuffle=False)
        self.test_loader = self.loader(self.test_set, batch_size=self.batch_size, shuffle=False)

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

    def get_random_samples(self, loader: torch_geometric.data.DataLoader, device: torch.device, label: Union[None, int],
                           len_samples=1, return_indexes=False):
        """
        randomly select torch_geometric.data.Data instances from loader and return these instances as a list
        Parameters
        ----------
        loader: torch_geometric.data.DataLoader,
            the loader to be used for random sampling
        device: torch.device,
            the device to put data in
        label: Union[None, int],
            if None then all labels are considered when randomly sampling, if 0 only include fake news,
            if 1 only include real news
        len_samples: int,
            the length of samples to be returned
        return_indexes: bool,
            if set to True returns indexes (in the respective dataset) with the samples.
        """
        assert len_samples <= len(loader.dataset)
        samples = []
        ds_indexes = []
        # collect the current dataset's indices in the whole dataset
        idxs_set = loader.dataset.indices
        if label is not None:
            # collect the instances from the whole dataset
            idxs_label = np.where(loader.dataset.dataset.data.y == label)
            # get the intersection of the two arrays to get the desired set
            idxs = np.intersect1d(idxs_set, idxs_label)
        else:
            idxs = idxs_set
        # get the location of these indexes in the given set
        ds_indexes = [np.where(idxs_set == i)[0][0] for i in idxs]
        # randomly select from prepared indexes
        indexes = np.random.choice(ds_indexes, len_samples, replace=False)
        print(f'Choosing indexes: {indexes}')
        dataset = torch.utils.data.Subset(loader.dataset, indexes)
        loader = self.loader(dataset, batch_size=self.batch_size, shuffle=True)

        for data in loader:
            samples.append(data.to(device))
        if return_indexes:
            return samples, indexes
        return samples

    def get_random_train_samples(self, device: torch.device, label=None, len_samples=1, return_indexes=False):
        """
        return a subset of the train_loader for model explanation
        Parameters
        ----------
        device: torch.device,
            the device to put data in
        label: Union[None, int],
            if None then all labels are considered when randomly sampling, if 0 only include fake news,
            if 1 only include real news
        len_samples: int,
            the length of samples to be returned
        return_indexes: bool,
            if set to True returns indexes with the samples.
        """

        return self.get_random_samples(self.train_loader, device, label, len_samples, return_indexes)

    def get_random_val_samples(self, device: torch.device, label=None, len_samples=1, return_indexes=False):
        """
        return a subset of the val_loader for model explanation
        Parameters
        ----------
        device: torch.device,
            the device to put data in
        label: Union[None, int],
            if None then all labels are considered when randomly sampling, if 0 only include fake news,
            if 1 only include real news
        len_samples: int,
            the length of samples to be returned
        return_indexes: bool,
            if set to True returns indexes with the samples.
        """
        return self.get_random_samples(self.val_laoder, device, label, len_samples, return_indexes)

    def get_test_samples(self, device: torch.device, label=None, len_samples=1, return_indexes=False):
        """
        return a subset of the test_loader for model explanation
        Parameters
        ----------
        device: torch.device,
            the device to put data in
        label: Union[None, int],
            if None then all labels are considered when randomly sampling, if 0 only include fake news,
            if 1 only include real news
        len_samples: int,
            the length of samples to be returned
        return_indexes: bool,
            if set to True returns indexes with the samples.
        """
        return self.get_random_samples(self.test_loader, device, label, len_samples, return_indexes)

    def fetch_all_news(self, label=None):
        """
        Parameters
        ----------
        label: int,
            0 or 1 indicating the label of the to be fetched.
        """
        ds = self.train_loader.dataset.dataset
        if label is not None:
            # collect the instances from the whole dataset
            idxs_label = np.where(ds.data.y == 0)
            ds = torch.utils.data.Subset(ds, idxs_label)
        return ds
