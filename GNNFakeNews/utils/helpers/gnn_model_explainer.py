import torch
from torch_geometric.nn import GNNExplainer
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Union
import torch_geometric.data

from GNNFakeNews.utils.helpers.gnn_model_helper import GNNModelHelper


class GNNModelExplainer:
    def __init__(self, model: GNNModelHelper,
                 sample_data: Union[torch_geometric.data.Data, torch_geometric.data.batch.Batch], epochs=200):
        """
        class is a manager for explanation pipeline. When initialized, it will explain the model with sample_data
        Parameters
        ----------
        model: GNNModelHelper,
            the model to be explained
        sample_data:  Union[torch_geometric.data.Data, torch_geometric.data.batch.DataBatch]
            the graph data to be explained
        epochs: int,
            epochs that GNNExplainer should run.
        """
        self.subgraph = None
        self.adjacency_matrix = None
        # pick the root node since it is the news itself, all leaf nodes are the users who shared this news
        self.node_idx = 0
        self.sample_data = sample_data

        self.gnn_explainer = GNNExplainer(model, epochs=epochs).to(model.m_args.device)
        self.node_feat_mask, self.edge_mask = self.gnn_explainer.explain_graph(x=sample_data.x,
                                                                               edge_index=sample_data.edge_index)

    @staticmethod
    def convert_label_to_text(label: torch.Tensor):
        """
        Parameters
        ----------
        label: torch.Tensor,
            torch tensor with possible values 0 and 1.
        """
        if label == 0:
            return 'Fake'
        else:
            return 'Real'

    def visualize_explaining_graph(self, threshold=None):
        """
        visualize the subgraph obtained from the GNNExplainer using the edge mask.
        Parameters
        ----------
        threshold: float,
            the threshold value for which edge masks to use when visualizing. helps to visualize better. defaults to
            the median of self.edge_mask
        """
        plt.figure(figsize=(8, 8))

        print(f'y: {self.convert_label_to_text(self.sample_data.y.cpu())}')
        threshold = torch.median(self.edge_mask).cpu() if threshold is None else threshold
        print(f'Removing edges with score less than {threshold} with '
              f'min {torch.min(self.edge_mask.cpu(), axis=-1)} and '
              f'max {torch.max(self.edge_mask.cpu(), axis=-1)}')

        indexes = self.edge_mask > threshold
        # print(f'Continuing with edges with following indexes: {indexes.cpu().numpy()}')
        print(f'Dropping {len(np.where(indexes.cpu().numpy() == False)[0])} edges out of {len(indexes)}')
        ax, self.subgraph = self.gnn_explainer.visualize_subgraph(node_idx=self.node_idx,
                                                                  edge_index=self.sample_data.edge_index[:, indexes]
                                                                  .cpu(),
                                                                  edge_mask=self.edge_mask[indexes].cpu(),
                                                                  # y=sample_data.y.cpu(),
                                                                  threshold=threshold,
                                                                  node_size=1000, font_size=15)
        plt.axis('off')

        plt.show()

    def get_node_ids_of_explaining_subgraph(self):
        """
        return the node_ids in self.subgraph
        """
        node_ids = []
        for entry in self.subgraph.nodes.items():
            if entry[0] != 0:
                node_ids.append(entry[0])
        return node_ids

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
