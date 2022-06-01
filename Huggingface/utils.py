import shap
from os import path
from enum import Enum
import pandas as pd
import torch
import transformers.pipelines
import numpy as np

PROJECT_DIR = path.abspath(path.dirname(__file__))
DATA_DIR = path.join(PROJECT_DIR, 'data')


class DatasetTypeEnum(Enum):
    # https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?resource=download
    ROBERTA_FAKE_NEWS = 'RobertaFakeNews'
    # https://github.com/chuachinhon/transformers_state_trolls_cch
    CHINHON_FAKE_TWEET_DETECT = 'Chinhon_FakeTweetDetect'


FILE_DIRS = {
    DatasetTypeEnum.ROBERTA_FAKE_NEWS: {'TRUE_NEWS_FILE': 'True.csv', 'FAKE_NEWS_FILE': 'Fake.csv'},
    DatasetTypeEnum.CHINHON_FAKE_TWEET_DETECT: {'TRUE_NEWS_FILE': 'real_50k.csv', 'FAKE_NEWS_FILE': 'troll_50k.csv'}
}

TEXT_COLNAMES = {
    DatasetTypeEnum.ROBERTA_FAKE_NEWS: 'text',
    DatasetTypeEnum.CHINHON_FAKE_TWEET_DETECT: 'clean_text'
}

LABEL_MAPPINGS = {
    DatasetTypeEnum.ROBERTA_FAKE_NEWS: {'LABEL_0': 'Fake', 'LABEL_1': 'True'},
    DatasetTypeEnum.CHINHON_FAKE_TWEET_DETECT: {'LABEL_0': 'True', 'LABEL_1': 'Fake'}
}


class HuggingfaceDatasetManager:
    """
    handles loading, sampling, preparing of the data for model explanation
    evaluate real as True and troll as fake when working with DatasetTypeEnum.CHINHON_FAKE_TWEET_DETECT
    """

    def __init__(self, dataset_type: DatasetTypeEnum, embedding_dim: int):
        self.text_colname = TEXT_COLNAMES[dataset_type]
        self.embedding_dim = embedding_dim
        true_news_file_dir = path.join(DATA_DIR, dataset_type.value, FILE_DIRS[dataset_type]["TRUE_NEWS_FILE"])
        self.true_news_df = self._load_dataframe(true_news_file_dir)
        fake_news_file_dir = path.join(DATA_DIR, dataset_type.value, FILE_DIRS[dataset_type]["FAKE_NEWS_FILE"])
        self.fake_news_df = self._load_dataframe(fake_news_file_dir)

    @staticmethod
    def _load_dataframe(dir: str):
        print(f"Loading instances from dir: {dir}")
        df = pd.read_csv(dir)
        print(f"Loaded {len(df)} instances from {dir}")
        return df

    def _fetch_rows_with_text(self, text: str, from_true_news: bool):
        """
        method filters the dataset according to the existence of the value of text parameter in the "text" column of the
        dataframe
        """
        df = self.true_news_df if from_true_news else self.fake_news_df
        idxs = df[self.text_colname].map(lambda x: text in x)
        return df[idxs]

    def _fetch_samples(self, dataframe, sample_count: int, sample_random: bool):
        df_len = len(dataframe)
        # if the indexes are not randomized the first "sample_count" rows will be selected.
        idxs = np.random.randint(low=0, high=df_len - 1, size=sample_count) if sample_random else \
            list(range(0, sample_count))
        print(f"Getting the following indexes: {idxs}")
        # we need to trim the original text before feeding it to the Explainer
        return dataframe[self.text_colname].map(lambda x: x[:self.embedding_dim]).iloc[idxs].values

    def _fetch_samples_with_text(self, from_true_news: bool, text: str, sample_count: int, sample_random: bool):
        df = self._fetch_rows_with_text(text, from_true_news=from_true_news)
        return self._fetch_samples(df, sample_count, sample_random)

    def fetch_true_samples_with_text(self, text: str, sample_count=10, sample_random=True):
        return self._fetch_samples_with_text(True, text, sample_count, sample_random)

    def fetch_fake_samples_with_text(self, text: str, sample_count=10, sample_random=True):
        return self._fetch_samples_with_text(False, text, sample_count, sample_random)

    def fetch_true_samples(self, sample_count=10, sample_random=True):
        return self._fetch_samples(self.true_news_df, sample_count, sample_random)

    def fetch_fake_samples(self, sample_count=10, sample_random=True):
        return self._fetch_samples(self.fake_news_df, sample_count, sample_random)


def explain_text(dataset_type: DatasetTypeEnum, pipeline: transformers.pipelines.Pipeline, text: str, algorithm="auto",
                 output_names=None):
    """
    method returns the shapley values for the given pipeline
    only works with transformers pipeline for now. not tested in other models.
    """
    output_names = output_names if output_names is not None else list(
        LABEL_MAPPINGS[DatasetTypeEnum.CHINHON_FAKE_TWEET_DETECT].values())
    print(
        'Explaining the following text: \n'
        '<--------------------------------------------------------------------------------------------->'
        f'\n{text}\n'
        '<--------------------------------------------------------------------------------------------->')
    predict_with_correct_labels(dataset_type, pipeline, text)
    explainer = shap.Explainer(pipeline, output_names=output_names, algorithm=algorithm)
    shap_values = explainer([text])
    shap.text_plot(shap_values)
    return shap_values, explainer


def predict_with_correct_labels(dataset_type: DatasetTypeEnum, pipeline: transformers.pipelines.Pipeline, text: str):
    label_mapping = LABEL_MAPPINGS[dataset_type]
    raw_predictions = pipeline([text])[0]
    for label_score_map in raw_predictions:
        print(f"Predicted {label_mapping[label_score_map['label']]} with score: {label_score_map['score']}")


# This will not work with string data, must pass tokenized data to use this class.
class _HuggingfaceDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, label):
        possible_labels = {
            "Fake": 0,
            "True": 1,
        }
        assert label in list(possible_labels.keys()), f"label must be one of these values: {possible_labels.keys()}"
        self.label = torch.tensor(possible_labels[label], dtype=torch.float32)
        self.tensor_data = torch.tensor(dataframe.values, dtype=torch.float32)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return self.tensor_data[idx], self.label
