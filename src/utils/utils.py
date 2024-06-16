import json
import os
from scipy.stats import pointbiserialr

import numpy as np
import pandas as pd


def get_drop_colums(settings):
    if settings.dataset.lower() == 'verbmem':
        drop_columns = ['id', 'old_new', 'decision', 'subject_file']
    elif settings.dataset.lower() == 'pilot01':
        drop_columns = ['id', 'subject_file', 'block_number', 'block_type', 'stim_indicator', 'go_nogo',
                        'is_experienced',
                        'is_resp', 'is_correct', 'stim']
        drop_columns = ['go_nogo', 'is_experienced', 'Wrd_Img', 'TargetValue', 'is_correct', 'TrialIndex',
                        'target_trial_index_asc']
    elif settings.dataset.lower() == 'clear':
        drop_columns = ['id', 'subject_file', 'ColorLev']
    else:
        raise ValueError("dataset in model_settings should be verbmem or pilot01")

    return drop_columns


def get_labels_pilot1(features_df, model_settings):
    if model_settings.binary_column == 'old_new':
        labels_array = features_df['old_new'].values
    elif model_settings.binary_column == 'decision':
        if len(np.unique(features_df['decision'].values)) > 2:
            labels_array = features_df['decision'].values * 2
        else:
            labels_array = features_df['decision'].values
    elif model_settings.binary_column == 'is_experienced':
        mapping = {'ctrl': 2, 'exp': 1, 'noexp': 0}

        data = {
            'block': [],
            'type': [],
            'num_ctl': [],
            'num_experienced': [],
            'num_not_experienced': [],
            'num_ctl_correct': [],
            'num_experienced_correct': [],
            'num_not_experienced_correct': []
        }

        for block in features_df['block_number'].unique():
            type = features_df[features_df['block_number'] == block]['block_type'].unique()
            num_trials = len(features_df[features_df['block_number'] == block])
            ctl = features_df[(features_df['block_number'] == block) & (features_df['stim'] == 'ctl')]
            experienced = features_df[(features_df['block_number'] == block) & (features_df['stim'] != 'ctl') & (
                    features_df['is_experienced'] != True)]
            notexperienced = features_df[(features_df['block_number'] == block) & (features_df['stim'] != 'ctl') & (
                    features_df['is_experienced'] != False)]

            ctl_correct = features_df[(features_df['block_number'] == block) & (features_df['stim'] == 'ctl') & (
                    features_df['is_correct'] == True)]
            experienced_correct = features_df[
                (features_df['block_number'] == block) & (features_df['stim'] != 'ctl') & (
                        features_df['is_experienced'] != True) & (features_df['is_correct'] == True)]
            notexperienced_correct = features_df[
                (features_df['block_number'] == block) & (features_df['stim'] != 'ctl') & (
                        features_df['is_experienced'] != False) & (features_df['is_correct'] == True)]

            data['block'].append(block)
            data['type'].append(type)
            data['num_ctl'].append(len(ctl))
            data['num_experienced'].append(len(experienced))
            data['num_not_experienced'].append(len(notexperienced))
            data['num_ctl_correct'].append(len(ctl_correct))
            data['num_experienced_correct'].append(len(experienced_correct))
            data['num_not_experienced_correct'].append(len(notexperienced_correct))

            print(f"Block {block}: type {type} "
                  f"\n \t Number of control stims : {len(ctl)} ({100 * len(ctl) / num_trials}%) "
                  f"(Correct answers: {len(ctl_correct)} ({100 * len(ctl_correct) / len(ctl)}%))"
                  f"\n \t Number of experienced stims : {len(experienced)} ({100 * len(experienced) / num_trials}%) "
                  f"(Correct answers: {len(experienced_correct)} ({100 * len(experienced_correct) / len(experienced)}%))"
                  f"\n \t Number of not experienced stims : {len(notexperienced)} ({100 * len(notexperienced) / num_trials}%) "
                  f"(Correct answers: {len(notexperienced_correct)} ({100 * len(notexperienced_correct) / len(notexperienced)}%))")

        # plot_datablocks_histogram(data)
        # Apply the mapping to the DataFrame column
        features_df = features_df[features_df['is_correct'] == True]
        features_df = features_df[((features_df['block_type'] == f"{model_settings['stim']}+e") |
                                   (features_df['block_type'] == f"{model_settings['stim']}-e") |
                                   (features_df['block_type'] == f"{model_settings['stim']}+e+x") |
                                   (features_df['block_type'] == f"{model_settings['stim']}-e+x")) & (
                                          features_df['stim'] != 'ctl')]
        # features_df['exp_label'] = features_df['exp_label'].map(mapping)

        labels_array = features_df['is_experienced'].values
    elif model_settings.binary_column == 'go_nogo':
        mapping = {'go': 1, 'nogo': 0}

        # Apply the mapping to the DataFrame column
        # features_df = features_df[features_df['is_correct'] == True]
        # features_df = features_df[(features_df['block_type'] == 'i+e') | (features_df['block_type'] == 'i-e')]
        features_df['exp_label'] = features_df['go_nogo'].map(mapping)

        labels_array = features_df['is_experienced'].values
    elif model_settings.binary_column == 'h5':
        labels_array = features_df['TargetValue'].values
    else:
        labels_array = features_df[model_settings['binary_column']].values
    return features_df, labels_array


def get_labels_verbmem(features_df, settings):
    labels_array = features_df[settings.binary_column].values
    return labels_array


def get_labels_clear(features_df, settings):
    labels_array = features_df[settings.target_column].values
    return labels_array


def get_labels(features_df, settings):
    patients_ids = features_df['id'].values
    patients_files = features_df['subject_file'].values[0]

    if settings.dataset.lower() == 'verbmem':
        labels_array = get_labels_verbmem(features_df, settings)
    elif settings.dataset.lower() == 'pilot01':
        features_df, labels_array = get_labels_pilot1(features_df, settings)
    elif settings.dataset.lower() == 'clear':
        labels_array = get_labels_clear(features_df, settings)
    else:
        raise ValueError("dataset in model_settings should be verbmem or pilot01 or clear")

    # ###################### Train test split  ######################
    y_one_hot = np.zeros((labels_array.size, 2))
    y_one_hot[np.arange(labels_array.size), labels_array.astype(int)] = 1
    y_one_hot = y_one_hot.astype(int)
    unique_pids = np.unique(patients_ids)

    return y_one_hot, labels_array, unique_pids, patients_files, features_df


def get_correlation(features_df, binary_column, paths):
    correlations = {}
    for column in features_df.columns:
        if column != binary_column and features_df[column].dtype in ['float64', 'int64']:
            # Calculate Point Biserial correlation for numerical columns
            corr, _ = pointbiserialr(features_df[column], features_df[binary_column])
            correlations[column] = corr

    # Convert correlations to a DataFrame for visualization
    corr_df = pd.DataFrame(list(correlations.items()), columns=['Column', 'Correlation'])

    # filtered_columns = corr_df['Column'][corr_df['Correlation'].abs() > 0.27]

    absolute_correlations = corr_df['Correlation'].abs()

    # Get the indices of the top 25 absolute correlations
    top_25_indices = absolute_correlations.nlargest(25).index

    # Use these indices to select the corresponding columns
    top_25_features = corr_df['Column'].iloc[top_25_indices]

    # Display the filtered columns
    selected_features = top_25_features.tolist()

    corr_df.to_csv(paths.path_result + 'features.csv')

    return selected_features


def get_selected_features(features_df, settings, paths, fold_idx, train_index,
                          pre_selected_features=None,
                          target_columns_drop=['id', 'old_new', 'decision', 'subject_file']):
    if pre_selected_features is None:
        pre_selected_features = ['reaction_time', 'D17-time_post_p300', 'A18-time_post_p300', 'D27-time_post_p300',
                                 'C8-time_post_p300', 'A6-time_post_p300', 'A5-time_post_p300',
                                 'A17-time_post_p300',
                                 'A19-time_post_p300', 'D16-time_post_p300',
                                 'C6-time_post_p300', 'C10-time_post_p300', 'C7-time_post_p300',
                                 'C31-time_post_p300',
                                 'C5-time_p300', 'A7-time_post_p300', 'D28-time_post_p300', 'D30-time_post_p300',
                                 'C15-time_post_p300', 'D7-time_post_p300', 'B19-time_post_p300',
                                 'A16-time_post_p300',
                                 'C14-time_post_p300', 'D19-time_post_p300', 'B10-time_post_p300']

    # select the feature extraction method
    if settings.features_selection_method.lower() == 'all':
        selected_features = features_df.columns.to_list()
    elif settings.features_selection_method.lower() == 'corr':
        selected_features = get_correlation(features_df.copy().reset_index(drop=True).loc[train_index, :].copy(),
                                            settings.target_column,
                                            paths)
    elif settings.features_selection_method.lower() == 'pre_selected':
        selected_features = pre_selected_features
    else:
        raise ValueError("Not a valid feature")

    patients_ids = features_df['id'].values
    patients_files = features_df['subject_file'].values[0]

    # save feature list
    with open(paths.path_result + f'features_fold{fold_idx + 1}.json', "w") as file:
        json.dump(selected_features, file, indent=2)

    # remove labels from dataframe and use selected features

    features_df = features_df.copy()
    features_df.drop(target_columns_drop, axis=1, inplace=True)
    selected_features = [feature for feature in selected_features if
                         feature not in target_columns_drop]

    print(
        f"Feature selection mode: {settings.features_selection_method.lower()}, Number of features {len(selected_features)}")

    # data transormation : "Normalization", "standardize", None
    if settings.feature_transformation is not None:
        if settings.feature_transformation.lower() == 'normalize':
            features_df = features_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        elif settings.feature_transformation.lower() == 'standardize':
            features_df = features_df.apply(lambda x: (x - x.mean()) / x.std())
        else:
            raise ValueError("The transformation is not defined")

    features_matrix = features_df[selected_features].values

    return features_matrix, selected_features, patients_ids, patients_files


class ResultList():
    def __init__(self, metric_list, method_list):
        self.result_list = {
            'subject id': [],
            'filename': []
        }

        for method in method_list:
            for metric in metric_list:
                self.result_list[method + ' avg_' + metric] = []
                self.result_list[method + ' std_' + metric] = []

    def add_result(self, method, metric, avg, std):
        self.result_list[method + ' avg_' + metric].append(avg)
        self.result_list[method + ' std_' + metric].append(std)

    def add_subject(self, unique_pids, patients_files):
        self.result_list['subject id'].append(unique_pids)
        self.result_list['filename'].append(patients_files.split('_')[0])

    def update_result(self, method, metric, avg_score, std_score):
        self.result_list[method + ' avg_' + metric].append(avg_score)
        self.result_list[method + ' std_' + metric].append(std_score)

    def to_dataframe(self):
        return pd.DataFrame(self.result_list)
