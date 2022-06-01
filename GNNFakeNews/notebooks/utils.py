from GNNFakeNews.utils.enums import GNNModelTypeEnum
from GNNFakeNews.models import gnn, gcnfn, bigcn, gnncl
from GNNFakeNews.utils.helpers import ModelArguments, HparamManager, GNNDatasetManager


def run_model(model_type: GNNModelTypeEnum, test_mode=False):
    """
    method is a convenient wrapper to initialize, train then evaluate the model
    """
    args = ModelArguments()
    hparams = HparamManager(model_type, test_mode=test_mode)
    dataset_manager = GNNDatasetManager(hparams, multi_gpu=args.multi_gpu)
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

    return model
