import types
from platform import python_version
from os import path
from enum import Enum

import datasets
import pandas as pd
import torch
import transformers.pipelines
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import shap
import matplotlib.pyplot as plot
from datasets import Dataset

print(f'This project is written and tested in Python {python_version()}')
PROJECT_DIR = path.abspath(path.dirname(__file__))
DATA_DIR = path.join(PROJECT_DIR, 'data')


class DatasetTypeEnum(Enum):
    KAGGLE_FAKE_NEWS = {
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
    UNKNOWN = KAGGLE_FAKE_NEWS


class TransformersModelTypeEnum(Enum):
    CH_FAKE_TWEET_DETECT = {
        'NAME': 'chinhon/fake_tweet_detect',
        'LABEL_MAPPINGS': {'LABEL_0': 'True', 'LABEL_1': 'Fake'},
        'TRAIN_DATASET': DatasetTypeEnum.CHINHON_FAKE_TWEET_DETECT,
        'EXTERNAL_LINKS': [
            'https://towardsdatascience.com/detecting-state-backed-twitter-trolls-with-transformers-5d7825945938',
            'https://github.com/chuachinhon/transformers_state_trolls_cch']
    }
    EZ_BERT_BASE_UNCASED_FAKE_NEWS = {
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
        'TRAIN_DATASET': DatasetTypeEnum.KAGGLE_FAKE_NEWS,
        'EXTERNAL_LINKS': [],
    }
    HB_ROBERTA_FAKE_NEWS = {
        'NAME': 'hamzab/roberta-fake-news-classification',
        'LABEL_MAPPINGS': {'FAKE': 'Fake', 'TRUE': 'True'},
        'TRAIN_DATASET': DatasetTypeEnum.KAGGLE_FAKE_NEWS,
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

    def __init__(self, dataset_type: DatasetTypeEnum):

        self.shuffled_df = None
        true_news_file_dir = path.join(DATA_DIR, dataset_type.value['TRUE_NEWS_DIR'])
        self.true_news_df = self._load_dataframe(true_news_file_dir)

        fake_news_file_dir = path.join(DATA_DIR, dataset_type.value['FAKE_NEWS_DIR'])
        self.fake_news_df = self._load_dataframe(fake_news_file_dir)

        self.text_colname = dataset_type.value['TEXT_COLNAME']
        self.text_col_idx = np.where(self.true_news_df.columns.values == self.text_colname)[0][0]
        if dataset_type == DatasetTypeEnum.KAGGLE_FAKE_NEWS:
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

    def _fetch_samples(self, dataframe: pd.DataFrame, sample_count: int, sample_random: bool,
                       model_type: TransformersModelTypeEnum):
        # if the indexes are not randomized the first "sample_count" rows will be selected.
        self.idxs = np.random.randint(low=0, high=len(dataframe) - 1,
                                      size=sample_count).tolist() if sample_random else list(range(0, sample_count))
        # print(f"Getting the following indexes: {idxs}")
        return self.fetch_latest_samples(model_type)
        # if model_type == TransformersModelTypeEnum.HB_ROBERTA_FAKE_NEWS:
        #    return dataframe.iloc[self.idxs].apply(self._transform, axis=1)[self.text_colname].values.tolist()
        # return dataframe[self.text_colname].iloc[self.idxs].values.tolist()

    def _transform(self, row):
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

    def fetch_random_samples(self, model_type: TransformersModelTypeEnum, sample_count=200):
        # add labels for explanation
        true_samples_with_labels = self.true_news_df.copy().sample(int(sample_count / 2))
        true_samples_with_labels['label'] = 1
        fake_samples_with_labels = self.fake_news_df.copy().sample(int(sample_count / 2))
        fake_samples_with_labels['label'] = 0

        # concatenate dataframes then shuffle
        self.shuffled_df = pd.concat([true_samples_with_labels, fake_samples_with_labels]).sample(frac=1)

        return self.shuffled_df['label'], self._fetch_samples(self.shuffled_df, sample_count, False, model_type)

    def fetch_latest_samples(self, model_type: TransformersModelTypeEnum):
        """
        fetch_random_samples should run before this method so that this method can recognize the last sampled indexes.
        """
        assert self.idxs is not None, 'self.idxs is None, you probably did not run fetch_random_samples.'
        assert self.shuffled_df is not None, 'self.shuffled_df is None, you probably did not run fetch_random_samples.'

        if model_type == TransformersModelTypeEnum.HB_ROBERTA_FAKE_NEWS:
            return self.shuffled_df.iloc[self.idxs].apply(self._transform, axis=1)[self.text_colname].values.tolist()
        return self.shuffled_df['label'], self.shuffled_df[self.text_colname].iloc[self.idxs].values.tolist()


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

    def explain_texts(self, texts: list, algorithm="auto", output_names=None, visualize=True, verbose=False):
        """
        method plots the text_plot for the given texts based on their SHAP values.
        only works with transformers pipeline for now. not tested in other models.
        """
        output_names = output_names if output_names is not None else list(
            self.model_type.value['LABEL_MAPPINGS'].values())
        predict_multiple_with_correct_labels(self, texts, verbose)
        explainer = shap.Explainer(model=self.pipeline, output_names=output_names, algorithm=algorithm)
        shap_values = explainer(texts)
        if visualize:
            shap.text_plot(shap_values)
        return shap_values, explainer


def custom_preprocess(self, inputs, **tokenizer_kwargs):
    return_tensors = self.framework
    model_inputs = self.tokenizer(inputs, truncation=True, return_tensors=return_tensors, **tokenizer_kwargs)
    return model_inputs


def barplot_token_shap_values(tokens, shap_values):
    fig, ax = plot.subplots(figsize=(20, 6))
    ax.barh(tokens, shap_values)
    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)

    # Add x, y gridlines
    ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)

    # Show top values
    ax.invert_yaxis()

    # Add annotation to bars
    for i in ax.patches:
        plot.text(i.get_width() + 0.001, i.get_y() + 0.5, str(round((i.get_width()), 5)), fontsize=10,
                  fontweight='bold', color='grey')

    # Add Plot Title
    ax.set_title(f'First {len(tokens)} important tokens', loc='left', )
    # Show Plot
    plot.show()


def textplot_shap_values(shap_values_dict):
    shap.text_plot(shap_values_dict)


def get_most_important_n_tokens(shap_values_dict, label, n=5, verbose=False):
    """
    collect n most important tokens from the shap_values_dict
    """
    # we keep this sum to understand how much of the overall is made up by the important ones.
    tokenized_input = shap_values_dict.data
    base_values = shap_values_dict.base_values
    if verbose:
        print(f'Base values are: {base_values[0]} and  {base_values[1]}')

    shap_values = shap_values_dict.values[:, label]

    first_n_important_indexes = np.argsort(shap_values)[-n:]
    first_n_important_shap_values = shap_values[first_n_important_indexes]
    if verbose:
        for idx in first_n_important_indexes:
            print(f'\n--> "{tokenized_input[idx]}" with shap value: {shap_values[idx]}')

    n_cumulative_importance = np.sum(first_n_important_shap_values, axis=0)
    # we take only the positive values for comparison. This value is very close to base_values[0]
    positive_shap_values_sum = np.sum(shap_values[np.where(shap_values > 0)], axis=0)
    if verbose:
        print(f'\nAll {n} tokens are {n * 100 / shap_values.shape[0]}% of all tokens.\nTheir cumulative importance is '
              f'{n_cumulative_importance}.\nThis value shows that '
              f'{n_cumulative_importance * 100 / positive_shap_values_sum}% of cumulative importance of all positive '
              f'contributing tokens ({positive_shap_values_sum}) are these {n} tokens.\n')
        print("###################################################################")

    return tokenized_input[first_n_important_indexes], first_n_important_shap_values


def predict_multiple_with_correct_labels(model_manager: ModelManager, texts: list, verbose=False):
    label_mapping = model_manager.model_type.value['LABEL_MAPPINGS']
    raw_predictions = model_manager.pipeline(texts)
    if verbose:
        for i, raw_pred in enumerate(raw_predictions):
            for label_score_map in raw_pred:
                print(f"Sample {i} is predicted {label_mapping[label_score_map['label']]} "
                      f"with score: {label_score_map['score']}")
            print("###################################################################")


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
