from os import path
from enum import Enum
import pandas as pd

PROJECT_DIR = path.abspath(path.abspath(path.dirname(__file__)))
TRUE_INSTANCE_DIR = 'True.csv'
FAKE_INSTANCE_DIR = 'Fake.csv'


class DatasetEnum(Enum):
    ROBERTA_FAKE_NEWS = 'RobertaFakeNews'  # https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?resource=download


def dataset_loader(dataset: DatasetEnum) -> (pd.DataFrame, pd.DataFrame):
    """
    convenience wrapper to load data for Huggingface models
    returns a tuple of true and false instances in the form of pandas dataframes
    """
    true_instance_loading_dir = path.join(PROJECT_DIR, dataset.value, TRUE_INSTANCE_DIR)
    print(f"Loading true instances from dir: {true_instance_loading_dir}")
    fake_instance_loading_dir = path.join(PROJECT_DIR, dataset.value, FAKE_INSTANCE_DIR)
    print(f"Loading fake instances from dir: {fake_instance_loading_dir}")
    return pd.read_csv(true_instance_loading_dir), pd.read_csv(fake_instance_loading_dir)
