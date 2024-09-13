import json
import os
from scipy.stats import pointbiserialr
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from src.utils.verbmem_utils import get_labels_verbmem, get_labels_pilot1


def print_directory_tree(startpath, level=0):
    """
    Print the directory tree of a folder and its subfolders.

    Args:
        startpath (str): The root directory from which to start printing the tree.
        level (int, optional): The current depth level (used internally for recursion). Defaults to 0.

    Returns:
        None: The function prints the directory structure directly.
    """
    prefix = '|   ' * level  # Creates the indentation for the current level
    if level == 0:
        print(os.path.basename(startpath))
    else:
        print(prefix + '|-- ' + os.path.basename(startpath))

    for index, (root, dirs, files) in enumerate(os.walk(startpath)):
        for sub_dir in dirs:
            sub_dir_path = os.path.join(root, sub_dir)
            print_directory_tree(sub_dir_path, level + 1)
        # Print files in the current directory
        for file in files:
            print(prefix + '|   ' + '|-- ' + file)
        break  # Prevents descending into subdirectories by os.walk


def get_drop_columns(settings):
    dataset = settings.dataset.lower()
    drop_columns = []

    if dataset == 'verbmem':
        drop_columns = ['id', 'old_new', 'decision', 'subject_file']
    elif dataset == 'pilot01':
        drop_columns = [
            'go_nogo', 'is_experienced', 'Wrd_Img', 'TargetValue',
            'is_correct', 'TrialIndex', 'target_trial_index_asc',
            'id', 'subject_file'
        ]
    elif dataset == 'clear':
        if settings.dataset_task == 'flicker':
            drop_columns = ['id', 'subject_file', 'ColorLev']
        elif settings.dataset_task == 'm_sequence':
            target_columns = [f"target_{i}_{j}" for i in range(10) for j in range(10)]
            drop_columns = ['id', 'subject_file'] + target_columns
        else:
            raise ValueError(f"Undefined task: {settings.dataset_task}")
    else:
        raise ValueError("Dataset should be 'verbmem', 'pilot01', or 'clear'")

    return drop_columns


def get_labels_clear(features_df, settings):
    if settings.dataset_task == 'flicker':
        labels_array = features_df[settings.target_column].values
        target_columns = settings.target_column
    elif settings.dataset_task == 'm_sequence':
        target_columns = [f"target_{i}_{j}" for i in range(10) for j in range(10)]
        if settings.correlation_mode == 'single':
            labels_array = features_df[settings.single_event_target].values
        else:
            labels_array = features_df[target_columns].values
    else:
        raise ValueError(f"Undefined task: {settings.dataset_task}")

    return labels_array, target_columns


def get_labels(features_df, settings):
    dataset = settings.dataset.lower()

    if dataset == 'verbmem':
        labels_array = get_labels_verbmem(features_df, settings)
        target_columns = settings.target_column
    elif dataset == 'pilot01':
        features_df, labels_array = get_labels_pilot1(features_df, settings)
        target_columns = settings.target_column
    elif dataset == 'clear':
        labels_array, target_columns = get_labels_clear(features_df, settings)
    else:
        raise ValueError("Dataset should be 'verbmem', 'pilot01', or 'clear'")

    y_one_hot = generate_one_hot_labels(labels_array)
    unique_pids = np.unique(features_df['id'].values)
    patients_files = features_df['subject_file'].values[0]

    return y_one_hot, labels_array, unique_pids, patients_files, features_df, target_columns


def generate_one_hot_labels(labels_array):
    if len(labels_array.shape) > 1 and labels_array.shape[1] == 100:
        return labels_array


    encoder = OneHotEncoder(sparse_output=False)
    # Fit and transform the data
    y_one_hot = encoder.fit_transform(labels_array)

    return y_one_hot.astype(int)


def extract_top_correlations(correlation_df, column, nlargest):
    top_corr = correlation_df[column].abs().nlargest(nlargest)
    return {column: top_corr.index.tolist()}


def get_correlation_multi_event(features_df, target_columns, drop_columns, paths, nlargest=25):
    target_columns = target_columns if isinstance(target_columns, list) else [target_columns]
    correlation_df = calculate_correlations_multi(features_df, target_columns=target_columns, drop_columns=drop_columns)
    top_correlations = {col: extract_top_correlations(correlation_df, col, nlargest)[col] for col in target_columns}
    correlation_df.to_csv(os.path.join(paths.path_result, 'features.csv'))
    return top_correlations


def get_correlation_single_event(features_df, single_event_target, drop_columns, paths, nlargest=25):
    single_event_target = single_event_target[0] if isinstance(single_event_target, list) else single_event_target
    correlation_df = calculate_correlations_single(features_df, drop_columns=drop_columns,
                                                   single_event_target_column=single_event_target)
    top_correlations = extract_top_correlations(correlation_df, single_event_target, nlargest)
    correlation_df.to_csv(os.path.join(paths.path_result, 'features.csv'))
    return top_correlations


def calculate_correlations_multi(features_df, target_columns, drop_columns):
    correlation_df = pd.DataFrame(index=[feature for feature in features_df.columns if feature not in drop_columns],
                                  columns=target_columns)

    for binary_column in target_columns:
        for column in features_df.columns:
            if column not in drop_columns and features_df[column].dtype in ['float64', 'int64']:
                corr, _ = pointbiserialr(features_df[column], features_df[binary_column])
                correlation_df.loc[column, binary_column] = corr

    return correlation_df.apply(pd.to_numeric, errors='coerce').fillna(0)


def calculate_correlations_single(features_df, single_event_target_column, drop_columns):
    correlation_df = pd.DataFrame(index=[feature for feature in features_df.columns if feature not in drop_columns],
                                  columns=[single_event_target_column])

    for column in features_df.columns:
        if column not in drop_columns:
            corr, _ = pointbiserialr(features_df[column], features_df[single_event_target_column])
            correlation_df.loc[column, single_event_target_column] = corr

    return correlation_df.apply(pd.to_numeric, errors='coerce').fillna(0)


def remove_duplicates(input_list):
    return list(dict.fromkeys(input_list))


def get_selected_features(features_df, settings, paths, fold_idx, train_index,
                          pre_selected_features=None,
                          target_columns_drop=None, num_important_features=25):
    target_columns_drop = target_columns_drop or ['id', 'old_new', 'decision', 'subject_file']
    pre_selected_features = pre_selected_features or [
        'reaction_time', 'D17-time_post_p300', 'A18-time_post_p300', 'D27-time_post_p300',
        'C8-time_post_p300', 'A6-time_post_p300', 'A5-time_post_p300',
        'A17-time_post_p300', 'A19-time_post_p300', 'D16-time_post_p300',
        'C6-time_post_p300', 'C10-time_post_p300', 'C7-time_post_p300',
        'C31-time_post_p300', 'C5-time_p300', 'A7-time_post_p300', 'D28-time_post_p300',
        'D30-time_post_p300', 'C15-time_post_p300', 'D7-time_post_p300',
        'B19-time_post_p300', 'A16-time_post_p300', 'C14-time_post_p300',
        'D19-time_post_p300', 'B10-time_post_p300'
    ]

    selected_features = select_features(features_df, settings=settings, paths=paths, fold_idx=fold_idx,
                                        train_index=train_index,
                                        pre_selected_features=pre_selected_features,
                                        drop_columns=target_columns_drop,
                                        nlargest=num_important_features)

    with open(os.path.join(paths.path_result, f'features_fold{fold_idx + 1}.json'), "w") as file:
        json.dump(selected_features, file, indent=2)

    patients_ids = features_df['id'].values
    patients_files = features_df['subject_file'].values[0]
    features_df.drop(target_columns_drop, axis=1, inplace=True)
    selected_features_all = list({feature for key in selected_features for feature in selected_features[key]})
    features_df = transform_features(features_df, settings)
    features_matrix = features_df[selected_features_all].values

    return features_matrix, selected_features, patients_ids, patients_files


def select_features(features_df, settings, paths, fold_idx, train_index, pre_selected_features, drop_columns, nlargest=25):
    method = settings.features_selection_method.lower()
    correlation_mode = settings.correlation_mode  # Default to 'multi'

    if method == 'all':
        features_list = [feature for feature in features_df.columns.to_list() if feature not in drop_columns]
        return {'all': features_list}
    elif method == 'corr':
        if correlation_mode == 'multi':
            return get_correlation_multi_event(features_df.reset_index(drop=True).loc[train_index].copy(),
                                               target_columns=settings.target_column,
                                               drop_columns=drop_columns,
                                               paths=paths, nlargest=nlargest)
        elif correlation_mode == 'single':
            return get_correlation_single_event(features_df.reset_index(drop=True).loc[train_index].copy(),
                                                drop_columns=drop_columns,
                                                single_event_target=settings.single_event_target,
                                                paths=paths,
                                                nlargest=nlargest)
        else:
            raise ValueError("Invalid correlation mode")
    elif method == 'pre_selected':
        return {'all': pre_selected_features}
    else:
        raise ValueError("Invalid feature selection method")


def transform_features(features_df, settings):
    transformation = settings.feature_transformation

    if transformation is None:
        return features_df

    if transformation.lower() == 'normalize':
        return features_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    elif transformation.lower() == 'standardize':
        return features_df.apply(lambda x: (x - x.mean()) / x.std())
    else:
        raise ValueError("Undefined transformation method")
