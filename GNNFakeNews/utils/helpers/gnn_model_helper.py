import torch
import torch.nn.functional as F
import numpy as np

from GNNFakeNews.utils.eval_helper import *
from GNNFakeNews.utils.enums import GNNModelTypeEnum

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class GNNModelHelper(torch.nn.Module):
    """
    helper class for GNN models. reduces code repetition
    """

    def __init__(self, model_args, model_hparams, model_dataset_manager, verbose=True):
        super(GNNModelHelper, self).__init__()
        self.m_args = model_args
        self.m_hparams = model_hparams
        self.m_dataset_manager = model_dataset_manager
        self.verbose = verbose
        self.last_layer = None
        self.last_conv_layer = None
        self.last_layers = {}
        self.last_conv_layers = {}

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
        elif self.m_hparams.model_type == GNNModelTypeEnum.UPFD_GCNFN:
            out = self(data.x, data.edge_index, data.batch, data.num_graphs)
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
            self.last_layers['y'] = y
            self.last_layers['last_layer_val'] = self.last_layer
            self.last_conv_layers['y'] = y
            self.last_conv_layers['last_layer_val'] = self.last_conv_layer
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
            for data in self.m_dataset_manager.train_loader:
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

    def m_visualize_tsne_of_last_conv_layer(self):
        """
        this method should run after train_then_eval is called.
        """
        assert self.last_conv_layers != {}, 'Run train_then_eval() first'

        self.last_conv_layers['y'] = self.last_conv_layers['y'].cpu().numpy()
        self.last_conv_layers['last_layer_val'] = self.last_conv_layers['last_layer_val'].cpu().detach().numpy()
        last_layer_transformed = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(
            X=self.last_conv_layers['last_layer_val'], y=self.last_conv_layers['y'])
        last_layer_transformed_real = last_layer_transformed[np.where(self.last_conv_layers['y'] == 1)[0]]
        last_layer_transformed_fake = last_layer_transformed[np.where(self.last_conv_layers['y'] == 0)[0]]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(last_layer_transformed_real[:, 0], last_layer_transformed_real[:, 1], c='green', label='Real')
        ax.scatter(last_layer_transformed_fake[:, 0], last_layer_transformed_fake[:, 1], c='red', label='Fake')

        ax.set_title('TSNE of the last convolutional layer')
        ax.legend(title='Classes')
        plt.show()

    def m_visualize_tsne_of_last_layer_before_classification(self):
        """
        this method should run after train_then_eval is called.
        """
        assert self.last_layers != {}, 'Run train_then_eval() first'

        self.last_layers['y'] = self.last_layers['y'].cpu().numpy()
        self.last_layers['last_layer_val'] = self.last_layers['last_layer_val'].cpu().detach().numpy()
        last_layer_transformed = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(
            X=self.last_layers['last_layer_val'], y=self.last_layers['y'])
        last_layer_transformed_real = last_layer_transformed[np.where(self.last_layers['y'] == 1)[0]]
        last_layer_transformed_fake = last_layer_transformed[np.where(self.last_layers['y'] == 0)[0]]

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(last_layer_transformed_real[:, 0], last_layer_transformed_real[:, 1], c='green', label='Real')
        ax.scatter(last_layer_transformed_fake[:, 0], last_layer_transformed_fake[:, 1], c='red', label='Fake')

        ax.set_title('TSNE of the last layer')
        ax.legend(title='Classes')
        plt.show()
