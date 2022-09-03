from GNNFakeNews.utils.enums import GNNModelTypeEnum
from GNNFakeNews.models import gcnfn, gnncl
from GNNFakeNews.models.deprecated import bigcn, gnn
from GNNFakeNews.utils.helpers import ModelArguments, HparamFactory, GNNDatasetManager, DATA_DIR
import pickle
from os import path


# from ipywidgets import interact, interactive, fixed, interact_manual
# import ipywidgets as widgets
# import os
# import json


def run_model(model_type: GNNModelTypeEnum, test_mode=False, return_dataset_manager=True, local_load=True,
              hparams=None, verbose=False):
    """
    method is a convenient wrapper to initialize, train then evaluate the model
    Parameters
    ----------
    model_type: GNNModelTypeEnum,
        the model type to be run
    test_mode: bool,
        when set to true, runs 1/5 of the original epochs in the hyperparameter settings
    return_dataset_manager: bool,
        when true, returns the respective instance of GNNDatasetManager with the model
    local_load: bool,
        when true, loads the dataset from local resources, when false downloads the UPFD dataset.
    hparams: HparamFactory,
        the hyperparameters of the model, when None, the default hyperparameters are used.
    verbose: bool,
        when true, outputs more information about the process.

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


def load_pkl_file(file_name: str):
    """
    read and return a .pkl file
    """
    with open(path.join(DATA_DIR, file_name), 'rb') as f:
        data = pickle.load(f)
    return data


'''
from networkx.readwrite import json_graph


def save_mask(G, fname, logdir, expdir, fmt='json', suffix=''):
    pth = os.path.join(logdir, expdir, fname + '-filt-' + suffix + '.' + fmt)
    if fmt == 'json':
        dt = json_graph.node_link_data(G)
        with open(pth, 'w') as f:
            json.dump(dt, f)
    elif fmt == 'pdf':
        plt.savefig(pth)
    elif fmt == 'npy':
        np.save(pth, nx.to_numpy_array(G))


def show_adjacency_full(logdir, expdir, mask, ax=None):
    adj = np.load(os.path.join(logdir, expdir, mask), allow_pickle=True)
    if ax is None:
        plt.figure()
        plt.imshow(adj);
    else:
        ax.imshow(adj)
    return adj


def read_adjacency_full(logdir, expdir, mask, ax=None):
    adj = np.load(os.path.join(logdir, expdir, mask), allow_pickle=True)
    return adj


@interact
def filter_adj(mask, thresh=0.5):
    filt_adj = read_adjacency_full(mask)
    filt_adj[filt_adj < thresh] = 0
    return filt_adj


# EDIT THIS INDEX
MASK_IDX = 0


# EDIT THIS INDEX

# m = masks[MASK_IDX]
# adj = read_adjacency_full(m)


@interact(thresh=widgets.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.01))
def plot_interactive(m, thresh=0.5):
    filt_adj = read_adjacency_full(m)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    plt.title(str(m));

    # Full adjacency
    ax1.set_title('Full Adjacency mask')
    adj = show_adjacency_full(m, ax=ax1);

    # Filtered adjacency
    filt_adj[filt_adj < thresh] = 0
    ax2.set_title('Filtered Adjacency mask');
    ax2.imshow(filt_adj);

    # Plot subgraph
    ax3.set_title("Subgraph")
    G_ = nx.from_numpy_array(adj)
    G = nx.from_numpy_array(filt_adj)
    G.remove_nodes_from(list(nx.isolates(G)))
    nx.draw(G, ax=ax3)
    save_mask(G, fname=m, fmt='json')

    print("Removed {} edges -- K = {} remain.".format(G_.number_of_edges() - G.number_of_edges(), G.number_of_edges()))
    print("Removed {} nodes -- K = {} remain.".format(G_.number_of_nodes() - G.number_of_nodes(), G.number_of_nodes()))'''
