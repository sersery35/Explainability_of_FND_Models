import re
import types
from platform import python_version
from os import path
from enum import Enum

import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import shap
from datasets import load_dataset

from Huggingface.noteboooks.deprecated.pipeline_for_hamzab_model import FakeNewsPipelineForHamzaB
from Huggingface.noteboooks.visualization_utils import barplot_first_n_largest_shap_values

print(f'This project is written and tested in Python {python_version()}')
PROJECT_DIR = path.abspath(path.dirname(__file__))
DATA_DIR = path.join(PROJECT_DIR, '../data')


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
        'TRAIN_DATASET': 'GonzaloA/fake_news',
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
    class that loads all deprecated from Huggingface and manages various tasks
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
        only works with transformers pipeline for now. not tested in other deprecated.
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


def predict_multiple_with_correct_labels(model_manager: ModelManager, texts: list, verbose=False):
    label_mapping = model_manager.model_type.value['LABEL_MAPPINGS']
    raw_predictions = model_manager.pipeline(texts)
    if verbose:
        for i, raw_pred in enumerate(raw_predictions):
            for label_score_map in raw_pred:
                print(f"Sample {i} is predicted {label_mapping[label_score_map['label']]} "
                      f"with score: {label_score_map['score']}")
            print("###################################################################")


class FakeNewsExplainer:
    def __init__(self, model: dict):
        """Class handles all code heavy tasks and returns meaningful data and visualizations for convenience.
        Parameters
        ----------
        model: dict
            a dict with following keys: 'NAME', 'LABEL_MAPPINGS', 'DATASET'
        """
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.label_mappings = model['LABEL_MAPPINGS']
        self.dataset = load_dataset(model['DATASET'])
        self.dataset.cache_files

        self.load_model(model['NAME'])
        self.explainer = shap.Explainer(self.pipeline)

    def load_model(self, model_name: str):
        """load model to the device and create the pipeline that will be used in the explanation
        Parameters
        ----------
        model_name: str
            the model name in huggingface transformers repository
        """
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        print(f'Using device: {device}')
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        pipeline = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, return_all_scores=True,
                                              device=0)
        pipeline.preprocess = types.MethodType(custom_preprocess, pipeline)

        self.pipeline = shap.models.TransformersPipeline(pipeline)

    def get_random_samples(self, n=10, split='train', label=None, should_contain_word='') -> (list, list):
        """returns the samples and their labels as builtin python lists: (list, list)
        Parameters
        ----------
        n : int
            the number of samples to be retrieved, The default is 10
        split : str
            can be 'train', 'validation', or 'test'.  which split to get samples from. The default is 'train'
        label: int,
            0 or 1, indicating the label of to be returned samples, defaults to None, if set to 0 or 1 the random
            samples will be sampled from the rows with that label.
        should_contain_word : str
            the word required to be in the samples. If the required word is not in any of the entries, then we return an
            empty list, The default is ''
        """
        ds = self.dataset.get(split)
        if should_contain_word != '':
            ds = ds.filter(lambda row: should_contain_word in row['text'])

        if label is not None:
            assert label in [0, 1], 'please use only 0 or 1 as labels'
            ds = ds.filter(lambda row: row['label'] == label)

        indexes = range(0, len(ds) - 1)
        indexes = np.random.choice(indexes, size=n, replace=False)
        print(f'Getting the indexes: {indexes}')
        # can use shuffle and pick the first 200 as well, i.e., samples.shuffle()[:200]
        random_samples = ds.select(indices=indexes)
        # now we need to transform Huggingface dataset to a python list
        random_samples_pd = random_samples.to_pandas()
        return random_samples_pd['text'].values.tolist(), random_samples_pd['label'].values.tolist()

    def explain_samples(self, samples: list, labels: list, text_plot=True, bar_plot=True, n_most_important_tokens=10,
                        verbose=False):
        """returns the shap values of random samples
        Parameters
        ----------
        samples: list(str)
            strings to be explained
        labels: list(int)
            labels of samples
        text_plot: bool
            whether to show the shap.text_plot() of the samples. The default is True
        bar_plot: bool
            whether to show the most important n_most_important_tokens tokens. The default is True
        n_most_important_tokens: int
            number of tokens to display in the bar plot. The default is 10
        verbose: whether to output predictions
        """
        shap_values = self.explainer(samples)
        print(f'labels: {labels}')
        for i, val in enumerate(shap_values):
            pred = self.predict_sample(samples[i], labels[i], verbose=verbose)
            if bar_plot:
                barplot_first_n_largest_shap_values(val, pred, n=n_most_important_tokens)
            if verbose:
                print(
                    '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            if text_plot:
                shap.plots.text(val[:, pred])
            if verbose:
                print(
                    '###################################################################################################')
        return shap_values

    def predict_sample(self, sample: str, label=None, verbose=False) -> int:
        """convenience method for printing out the prediction probabilities
        Parameters
        ----------
        sample: str
            the sample to be predicted by the model
        label: int
            the actual label of the sample. leave empty when predicting a sample without a label, i.e., test data
        verbose: bool,
            whether to print out the prediction vs actual information, the default is True
        """
        pred = self.pipeline([sample])
        fake_prob = pred[:, 0]
        real_prob = pred[:, 1]
        if verbose:
            print('###################################################################################################')
            print(f'Predicted fake with {fake_prob}')
            print(f'Predicted real with {real_prob}')
            if label is not None:
                print(f'The actual value is {self.label_mappings[label]}')
            print('---------------------------------------------------------------------------------------------------')
        return 1 if real_prob > fake_prob else 0

    @staticmethod
    def perturb_sample(sample: str, perturbation_type='add', position=0, target_string=None,
                       new_string=None, replace_all=True, replace_until_position=0):
        """
        when perturbation_type is 'add' then the method adds new_string to the given perturbation_location
        when perturbation_type is 'delete' then the method removes target_string from the given sample
        when perturbation_type is 'replace' then the method replaces the target_string with new_string.
        Parameters
        ----------
        sample: str
            string to be perturbed
        perturbation_type: str
            one of 'add', 'delete', 'replace', type of perturbation method
        position: int
            index of which occurrence to remove, if 0 then add to the beginning, if -1 add to the end
            note that this position is the index of the characters not tokenized words.
        target_string: str
            the target string to be replaced
        new_string: str
            the new string that will either
            i. be added to perturbation_location of the sample
            ii. replace the target_string
        replace_all: bool
            if True, and if perturbation_type is 'replace' or 'delete' then replaces/deletes all occurrences.
            if False, replaces/deletes the first occurrence
        replace_until_position: int
            starting from the first, until how many occurrences should the target_string in sample be replaced/deleted.
        """
        perturbation_types = ['add', 'delete', 'replace']
        assert perturbation_type in perturbation_types, f'parameter perturbation_types can only take values: ' \
                                                        f'{perturbation_types}'
        assert position.__abs__() < len(
            sample), f'the given parameter position is an out of range index: {position} <! {len(sample)}'
        if perturbation_type == 'add':
            if position == 0:
                return new_string + sample
            elif position == -1:
                return sample + new_string
            else:
                return sample[0:position] + new_string + sample[position:]
        elif perturbation_type == 'delete':
            matches = re.findall(fr'{target_string}', sample)
            assert len(matches) > 0, 'target_string can not be found in the sample.'
            return sample.replace(target_string, '') if replace_until_position <= 0 or replace_all \
                else sample.replace(target_string, '', replace_until_position)
        else:
            assert target_string is not None and target_string != '', 'target_string should have a value, can not be ' \
                                                                      'None or empty string ("")'
            matches = re.findall(fr'{target_string}', sample)
            assert len(matches) > 0, 'target_string can not be found in the sample.'
            return sample.replace(target_string, new_string) if replace_until_position <= 0 or replace_all \
                else sample.replace(target_string, new_string, replace_until_position)


def custom_preprocess(self, inputs, **tokenizer_kwargs):
    """
    convenience method to force transformers pipeline to truncate the texts to 512 words
    """
    return_tensors = self.framework
    model_inputs = self.tokenizer(inputs, truncation=True, return_tensors=return_tensors, **tokenizer_kwargs)
    return model_inputs
