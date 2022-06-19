import types
from platform import python_version
from os import path
from enum import Enum
import pandas as pd
import torch
import transformers.pipelines
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import shap

print(f'This project is written and tested in Python {python_version()}')
PROJECT_DIR = path.abspath(path.dirname(__file__))
DATA_DIR = path.join(PROJECT_DIR, 'data')


class DatasetTypeEnum(Enum):
    ROBERTA_FAKE_NEWS = {
        'TRUE_NEWS_DIR': 'RobertaFakeNews/True.csv',
        'FAKE_NEWS_DIR': 'RobertaFakeNews/Fake.csv',
        'SOURCE': 'https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?resource=download',
        'TEXT_COLNAME': 'text',
        'TITLE_COLNAME': 'title',
    }
    CHINHON_FAKE_TWEET_DETECT = {
        'TRUE_NEWS_DIR': 'Chinhon_FakeTweetDetect/real_50k.csv',
        'FAKE_NEWS_DIR': 'Chinhon_FakeTweetDetect/troll_50k.csv',
        'DIR': 'Chinhon_FakeTweetDetect',
        'SOURCE': 'https://github.com/chuachinhon/transformers_state_trolls_cch',
        'TEXT_COLNAME': 'clean_text',
    }
    # if the dataset is unknown we set it to the most common dataset for now
    UNKNOWN = ROBERTA_FAKE_NEWS


class TransformersModelTypeEnum(Enum):
    CH_FAKE_TWEET_DETECT = {
        'NAME': 'chinhon/fake_tweet_detect',
        'LABEL_MAPPINGS': {'LABEL_0': 'True', 'LABEL_1': 'Fake'},
        'TRAIN_DATASET': DatasetTypeEnum.CHINHON_FAKE_TWEET_DETECT,
        'EXTERNAL_LINKS': [
            'https://towardsdatascience.com/detecting-state-backed-twitter-trolls-with-transformers-5d7825945938',
            'https://github.com/chuachinhon/transformers_state_trolls_cch']
    }
    EZ_BERT_BASE_CASED_FAKE_NEWS = {
        'NAME': 'elozano/bert-base-cased-fake-news',
        'LABEL_MAPPINGS': {'Fake': 'Fake', 'Real': 'True'},
        'TRAIN_DATASET': DatasetTypeEnum.UNKNOWN,
        'EXTERNAL_LINKS': [],
    }
    GA_DISTIL_ROBERTA_BASE_FINETUNED_FAKE_NEWS = {
        'NAME': 'GonzaloA/distilroberta-base-finetuned-fakeNews',
        'LABEL_MAPPINGS': {'LABEL_0': 'Fake', 'LABEL_1': 'True'},
        'TRAIN_DATASET': DatasetTypeEnum.UNKNOWN,
        'EXTERNAL_LINKS': [],
    }
    GS_ROBERTA_FAKE_NEWS = {
        'NAME': 'ghanashyamvtatti/roberta-fake-news',
        'LABEL_MAPPINGS': {'LABEL_0': 'Fake', 'LABEL_1': 'True'},
        'TRAIN_DATASET': DatasetTypeEnum.ROBERTA_FAKE_NEWS,
        'EXTERNAL_LINKS': [],
    }
    HB_ROBERTA_FAKE_NEWS = {
        'NAME': 'hamzab/roberta-fake-news-classification',
        'LABEL_MAPPINGS': {'FAKE': 'Fake', 'TRUE': 'True'},
        'TRAIN_DATASET': DatasetTypeEnum.ROBERTA_FAKE_NEWS,
        'EXTERNAL_LINKS': []
    }
    JY_FAKE_NEWS_BERT_DETECT = {
        'NAME': 'jy46604790/Fake-News-Bert-Detect',
        'LABEL_MAPPINGS': {'LABEL_0': 'Fake', 'LABEL_1': 'True'},
        'TRAIN_DATASET': DatasetTypeEnum.UNKNOWN,
        'EXTERNAL_LINKS': [],
    }


class HuggingfaceDatasetManager:
    """
    handles loading, sampling, preparing of the data for model explanation
    evaluate real as True and troll as fake when working with DatasetTypeEnum.CHINHON_FAKE_TWEET_DETECT
    """

    def __init__(self, model_type: TransformersModelTypeEnum):
        dataset_type = model_type.value['TRAIN_DATASET']
        self.model_type = model_type
        true_news_file_dir = path.join(DATA_DIR, dataset_type.value['TRUE_NEWS_DIR'])
        self.true_news_df = self._load_dataframe(true_news_file_dir)
        fake_news_file_dir = path.join(DATA_DIR, dataset_type.value['FAKE_NEWS_DIR'])
        self.fake_news_df = self._load_dataframe(fake_news_file_dir)

        self.text_colname = dataset_type.value['TEXT_COLNAME']
        self.text_col_idx = np.where(self.true_news_df.columns.values == self.text_colname)[0][0]
        if dataset_type == DatasetTypeEnum.ROBERTA_FAKE_NEWS:
            self.title_colname = dataset_type.value['TITLE_COLNAME']
            self.title_col_idx = np.where(self.true_news_df.columns.values == self.title_colname)[0][0]

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

    def _fetch_samples(self, dataframe, sample_count: int, sample_random: bool) -> list:
        df_len = len(dataframe)
        # if the indexes are not randomized the first "sample_count" rows will be selected.
        idxs = np.random.randint(low=0, high=df_len - 1, size=sample_count).tolist() if sample_random else \
            list(range(0, sample_count))
        print(f"Getting the following indexes: {idxs}")

        return dataframe.iloc[idxs].apply(self._transform, axis=1)[self.text_colname].values.tolist()

    def _transform(self, row):
        if self.model_type == TransformersModelTypeEnum.HB_ROBERTA_FAKE_NEWS:
            title_token = '<title> '
            content_token = ' <content> '
            end_token = ' <end>'
            text_title_part = title_token + row[self.title_col_idx]
            text_content_part = content_token + row[self.text_col_idx]
            row[self.text_col_idx] = text_title_part + text_content_part + end_token
        return row

    def _fetch_samples_with_text(self, from_true_news: bool, text: str, sample_count: int, sample_random: bool):
        df = self._fetch_rows_with_text(text, from_true_news=from_true_news)
        return self._fetch_samples(df, sample_count, sample_random)

    def fetch_true_samples_with_text(self, text: str, sample_count=3, sample_random=True):
        return self._fetch_samples_with_text(True, text, sample_count, sample_random)

    def fetch_fake_samples_with_text(self, text: str, sample_count=3, sample_random=True):
        return self._fetch_samples_with_text(False, text, sample_count, sample_random)

    def fetch_true_samples(self, sample_count=3, sample_random=True):
        return self._fetch_samples(self.true_news_df, sample_count, sample_random)

    def fetch_fake_samples(self, sample_count=3, sample_random=True):
        return self._fetch_samples(self.fake_news_df, sample_count, sample_random)


class ModelManager:
    """
    class that loads all models from Huggingface and manages various tasks
    """

    def __init__(self, model_type: TransformersModelTypeEnum):
        self.model_type = model_type
        model_name = model_type.value['NAME']
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        print(f'Using device: {device}')
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_type == TransformersModelTypeEnum.HB_ROBERTA_FAKE_NEWS:
            self.pipeline = FakeNewsPipelineForHamzaB(model=model, tokenizer=tokenizer, return_all_scores=True,
                                                      device=0)
        else:
            self.pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True,
                                                       device=0)
        # force the pipeline preprocess to truncate the outputs for convenience
        self.pipeline.preprocess = types.MethodType(custom_preprocess, self.pipeline)


def explain_texts(model_manager: ModelManager, texts: list,
                  algorithm="auto", output_names=None):
    """
    method plots the text_plot for the given texts based on their SHAP values.
    only works with transformers pipeline for now. not tested in other models.
    """
    output_names = output_names if output_names is not None else list(
        model_manager.model_type.value['LABEL_MAPPINGS'].values())
    predict_multiple_with_correct_labels(model_manager, texts)
    explainer = shap.Explainer(model=model_manager.pipeline, output_names=output_names, algorithm=algorithm)
    shap_values = explainer(texts)
    shap.text_plot(shap_values)
    # return shap_values, explainer


def predict_multiple_with_correct_labels(model_manager: ModelManager, texts: list):
    label_mapping = model_manager.model_type.value['LABEL_MAPPINGS']
    raw_predictions = model_manager.pipeline(texts)
    for i, raw_pred in enumerate(raw_predictions):
        for label_score_map in raw_pred:
            print(f"Sample {i} is predicted {label_mapping[label_score_map['label']]} "
                  f"with score: {label_score_map['score']}")
        print("###################################################################")


def custom_preprocess(self, inputs, **tokenizer_kwargs):
    return_tensors = self.framework
    model_inputs = self.tokenizer(inputs, truncation=True, return_tensors=return_tensors, **tokenizer_kwargs)
    return model_inputs


class FakeNewsPipelineForHamzaB(transformers.TextClassificationPipeline):
    """
    custom pipeline for the model TransformersModelTypeEnum.HB_ROBERTA_FAKE_NEWS
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def postprocess(self, model_outputs, function_to_apply=None, return_all_scores=True):
        assert function_to_apply is None, 'If you want to use another function, ' \
                                          'please use TextClassificationPipeline instead.'
        outputs = model_outputs["logits"][0].numpy()
        scores = transformers.pipelines.text_classification.softmax(outputs)

        if self.model.config.label2id is None:
            # there is no label2id in the model config so we create it
            label2id = {}
            for id, label in self.model.config.id2label.items():
                label2id[label] = id
            self.model.config.label2id = label2id
            assert self.model.config.label2id is not None, 'Updating label2id in model config failed.'
        if return_all_scores:
            return [{"label": self.model.config.id2label[i], "score": score.item()} for i, score in
                    enumerate(scores)]
        else:
            return {"label": self.model.config.id2label[scores.argmax().item()], "score": scores.max().item()}


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
