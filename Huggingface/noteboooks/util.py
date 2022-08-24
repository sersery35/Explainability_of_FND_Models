import types
import matplotlib.pyplot as plot
import numpy as np
import shap.models
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TextClassificationPipeline, AutoModelForSequenceClassification
import shap
import re

# for interactive plots
shap.initjs()


def barplot_token_shap_values(tokens, shap_values, label):
    fig, ax = plot.subplots(figsize=(12, 8))
    if label == 0:
        label_color = 'red'
    else:
        label_color = 'green'
    ax.barh(tokens, shap_values, color=label_color)
    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)

    # Add x, y gridlines
    ax.grid(visible=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)

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

    print(f'Max shap value is {np.max(shap_values)}')

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


def barplot_first_n_largest_shap_values(shap_values, label, n=10):
    """
    wrapper method for plotting the most important n tokens
    shap_values: dict, the shap values of the sample
    label: int, the actual label of the sample
    n: int, how many tokens to barplot
    """
    tokens, s_values = get_most_important_n_tokens(shap_values, label=label, n=n)
    barplot_token_shap_values(tokens, s_values, label)


def custom_preprocess(self, inputs, **tokenizer_kwargs):
    """
    convenience method to force transformers pipeline to truncate the texts to 512 words
    """
    return_tensors = self.framework
    model_inputs = self.tokenizer(inputs, truncation=True, return_tensors=return_tensors, **tokenizer_kwargs)
    return model_inputs


class FakeNewsExplainer:
    def __init__(self, model):
        self.pipeline = None
        self.label_mappings = model['LABEL_MAPPINGS']
        self.dataset = load_dataset(model['DATASET'])
        self.dataset.cache_files

        self.load_model(model['NAME'])
        self.explainer = shap.Explainer(self.pipeline)

    def load_model(self, model_name):
        """
        load model to the device and create the pipeline that will be used in the explanation
        """
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        print(f'Using device: {device}')
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True,
                                              device=0)
        pipeline.preprocess = types.MethodType(custom_preprocess, pipeline)

        self.pipeline = shap.models.TransformersPipeline(pipeline)

    def get_random_samples(self, n=10, split='train', should_contain_word='') -> (list, list):
        """
        n: int, the number of samples to be retrieved
        split: str, can be 'train', 'validation', or 'test'.  which split to get samples from.
        should_contain_word: str, the word required to be in the samples. If the required word is not in any of the
        entries, then we return an empty list
        method returns the samples and their labels as builtin python lists: (list, list)
        """
        ds = self.dataset.get(split)
        if should_contain_word != '':
            ds = ds.filter(lambda row: should_contain_word in row['text'])
        indexes = np.random.randint(low=0, high=len(ds) - 1, size=n)
        # can use shuffle and pick the first 200 as well, i.e., samples.shuffle().select(indices=indexes)
        random_samples = ds.select(indices=indexes)
        # now we need to transform Huggingface dataset to a python list
        random_samples_pd = random_samples.to_pandas()
        return random_samples_pd['text'].values.tolist(), random_samples_pd['label'].values.tolist()

    def explain_samples(self, samples: list, labels: list, split='train', should_contain_word='',
                        text_plot=False, bar_plot=False, n_most_important_tokens=10):
        """
        samples: list of strings, strings to be explained
        labels: list of integers, labels of samples
        split: str, can be 'train', 'validation', or 'test'.  which split to get samples from.
        should_contain_word: str, the word required to be in the samples. If the required word is not in any of the
        entries, then we return an empty list
        text_plot: bool, whether to show the shap.text_plot() of the samples.
        bar_plot: bool, whether to show the most important n_most_important_tokens tokens
        n_most_important_tokens: int, number of tokens to display in the bar plot
        method returns the shap values of random samples
        """
        shap_values = self.explainer(samples)
        for i, val in enumerate(shap_values):
            self.print_predictions_for_sample(samples[i], labels[i])
            if bar_plot:
                barplot_first_n_largest_shap_values(val, labels[i], n=n_most_important_tokens)
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            if text_plot:
                shap.plots.text(val[:, labels[i]])
            print('###################################################################################################')
        return shap_values

    def print_predictions_for_sample(self, sample, label=None):
        """
        convenience method for printing out the prediction probabilities
        sample: str, the sample to be predicted by the model
        label: int, the actual label of the sample. leave empty when predicting a sample without a label, i.e., test
        data
        """
        pred = self.pipeline([sample])
        fake_prob = pred[:, 0]
        real_prob = pred[:, 1]
        print('###################################################################################################')
        print(f'Predicted fake with {fake_prob}')
        print(f'Predicted real with {real_prob}')
        if label is not None:
            print(f'The actual value is {self.label_mappings[label]}')
        print('---------------------------------------------------------------------------------------------------')

    @staticmethod
    def perturb_sample(sample: str, perturbation_type='add', position=0, target_string=None,
                       new_string=None, replace_all=True, replace_until_position=0):
        """
        sample: str, string to be perturbed
        perturbation_type: str, one of 'add', 'delete', 'replace', type of perturbation method
        position: int, index of which occurrence to remove, if 0 then add to the beginning, if -1 add to the end
        note that this position is the index of the characters not tokenized words.
        target_string: str, the target string to be replaced
        new_string: str, the new string that will either
        i. be added to perturbation_location of the sample
        ii. replace the target_string
        replace_all: bool, if True, and if perturbation_type is 'replace' or 'delete' then replaces/deletes all
        occurrences.
        if False, replaces/deletes the first occurrence
        replace_until_position: int, starting from the first, until how many occurrences should the target_string in
        sample be replaced/deleted.
        when perturbation_type is 'add' then the method adds new_string to the given perturbation_location
        when perturbation_type is 'delete' then the method removes target_string from the given sample
        when perturbation_type is 'replace' then the method replaces the target_string with new_string.
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
