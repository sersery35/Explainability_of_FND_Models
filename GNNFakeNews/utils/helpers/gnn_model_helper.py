import torch
import torch.nn.functional as F

from GNNFakeNews.utils.eval_helper import *
from GNNFakeNews.utils.enums import GNNModelTypeEnum


class GNNModelHelper(torch.nn.Module):
    """
    helper class for GNN models. reduces code repetition
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
