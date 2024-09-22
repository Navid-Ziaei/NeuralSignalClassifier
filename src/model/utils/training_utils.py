import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold

from src.evaluation.evaluation import ResultList
from src.utils import get_labels, get_correlation_single_event, get_drop_columns, get_selected_features
from src.visualization.visualization_utils import plot_metric_heatmaps
from src.model import train_xgb, train_ldgd, train_fast_ldgd


def train_model_with_folds(features_raw_df_dict, settings, paths, random_seed=42):
    """
    Train models using either k-fold cross-validation or a single train/test split for each patient's data.

    This function performs the following steps for each patient's data:
    1. Prepares the data by removing certain columns.
    2. Extracts labels and features from the raw data.
    3. Depending on the mode, either generates k-folds or a single train/test split.
    4. Trains models specified in the settings on the training data for each fold or split.
    5. Evaluates the models on the test data and logs the results.
    6. Computes and saves average scores for each model across all folds (if in k-fold mode).

    Args:
        settings (Settings): The settings object containing configurations.
        paths (Paths): The paths object containing file paths.

    Returns:
        None: The function saves results to CSV files and updates the results logger.
    """
    results_logger = ResultList(method_list=settings.method_list, metric_list=settings.metric_list)

    for patient_id, (patient_file_name, features_raw_df) in enumerate(features_raw_df_dict.items()):
        print(f"====== Subject {patient_id} ({patient_file_name.split('.')[0]}) ====== \n")
        if patient_file_name == 'p05_task1_block2_CLEAR_Flicker_flicker':
            continue

        # Prepare data by removing specific columns
        columns_to_remove = [col for col in features_raw_df.columns if "EX" in col]
        features_df = features_raw_df.drop(columns=columns_to_remove)

        # Extract labels and features
        y_one_hot, labels_array, unique_pids, patients_files, features_df, target_columns = get_labels(features_df,
                                                                                                       settings)
        results_logger.add_subject(unique_pids=patient_id, patients_files=patients_files)
        paths.create_subject_paths(patients_files.split('.')[0])

        if settings.cross_validation_mode in ['k-fold', 'order']:
            # Generate k-folds for cross-validation
            cv_mode = setup_cross_validation(settings.cross_validation_mode, num_folds=settings.num_fold,
                                             random_state=random_seed)
            folds = generate_folds(features_df, labels_array, cv_mode, n_splits=settings.num_fold)
        else:  # Single train/test split
            folds = [(train_test_split(np.arange(len(labels_array)), test_size=0.2, stratify=labels_array))]

        fold_results = {method: [] for method in settings.method_list}
        fold_report_results = {method: [] for method in settings.method_list}

        for fold_idx, (train_index, test_index) in enumerate(folds):
            paths.create_fold_path(fold_idx)

            # Select features and prepare the training and testing datasets
            features_matrix, selected_features, patients_ids, _ = get_selected_features(
                features_df=features_df.copy(), settings=settings, paths=paths,
                fold_idx=fold_idx, train_index=train_index, target_columns_drop=get_drop_columns(settings),
                num_important_features=settings.num_important_features)

            data_train, data_test = features_matrix[train_index], features_matrix[test_index]
            labels_train, labels_test = labels_array[train_index], labels_array[test_index]
            y_train, y_test = y_one_hot[train_index], y_one_hot[test_index]

            # Train and evaluate models for each specified method
            for method in settings.method_list:
                print(f"=========== Train Subject {patient_id} Fold {fold_idx} Model {method} =========== \n")
                results, report_results = train_model(method,
                                                      data_train=data_train,
                                                      labels_train=labels_train,
                                                      data_test=data_test,
                                                      labels_test=labels_test,
                                                      settings=settings,
                                                      paths=paths,
                                                      y_train=y_train,
                                                      y_test=y_test,
                                                      selected_features=selected_features,
                                                      target_columns=target_columns)
                fold_results[method].append(results)
                fold_report_results[method].append(report_results)

            plt.close('all')

        # Aggregate the results for fold reports
        aggregated_results = aggregate_fold_report_results(fold_report_results)

        if settings.dataset_task == 'm_sequence':
            mean_grid_precision, _ = plot_metric_heatmaps(aggregated_results, metric='precision', grid_size=(10, 10),
                                                          save_dir=paths.results_base_path)
            mean_grid_recall, _ = plot_metric_heatmaps(aggregated_results, metric='recall', grid_size=(10, 10),
                                                       save_dir=paths.results_base_path)
            mean_grid_f1, _ = plot_metric_heatmaps(aggregated_results, metric='f1-score', grid_size=(10, 10),
                                                   save_dir=paths.results_base_path)

        # Save or print the DataFrames
        for method, df in aggregated_results.items():
            print(f"Aggregated results for {method}:\n", df)
            df.to_csv(os.path.join(paths.results_base_path, f'{method}_fold_report_results.csv'))

        # Compute and save average scores across all folds (or the single split)
        for method in settings.method_list:
            for metric in settings.metric_list:
                avg_score = np.mean([result[metric] for result in fold_results[method]])
                std_score = np.std([result[metric] for result in fold_results[method]])
                results_logger.update_result(method, metric, avg_score, std_score)
                print(f"Method {method}: {metric}: {avg_score} ± {std_score}")

    # Save overall results to CSV
    result_df = results_logger.to_dataframe()
    result_df.to_csv(os.path.join(paths.base_path, paths.folder_name, f'{settings.cross_validation_mode}_results.csv'))


def setup_cross_validation(cv_mode, num_folds=5, random_state=42):
    if isinstance(cv_mode, str):
        if cv_mode == 'k-fold':
            return StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
        elif cv_mode == 'block':
            return 'block'
        elif cv_mode == 'order':
            return 'order'
        else:
            raise ValueError("cross_validation_mode should be a number or 'block'/'order'")
    else:
        raise ValueError("Invalid cross_validation_mode")


def generate_folds(features_df, labels_array, cv_mode, n_splits=5):
    if cv_mode == 'block':
        block_nums = features_df['block_number'].unique()
        return [(np.where(features_df['block_number'] != block)[0],
                 np.where(features_df['block_number'] == block)[0]) for block in block_nums]
    elif cv_mode == 'order':
        num_trials = labels_array.shape[0]
        trial_idx = np.arange(num_trials)
        fold_idx = np.int16(n_splits * trial_idx / num_trials)
        return [(np.where(fold_idx != fold)[0], np.where(fold_idx == fold)[0]) for fold in np.unique(fold_idx)]
    else:
        return cv_mode.split(features_df, labels_array)


def train_model(method, data_train, labels_train, data_test, labels_test, settings, paths, y_train=None, y_test=None,
                selected_features=None, target_columns=None):
    if method.lower() == 'xgboost':
        return train_xgb(data_train, labels_train, data_test, labels_test, paths,
                         selected_features=selected_features, target_columns=target_columns)
    elif method.lower() == 'ldgd':
        return train_ldgd(data_train, labels_train, data_test, labels_test, y_train, y_test, settings, paths)
    elif method.lower() == 'fast_ldgd':
        return train_fast_ldgd(data_train, labels_train, data_test, labels_test, y_train, y_test, settings, paths,
                               use_validation=False)
    else:
        raise ValueError("Method should be 'xgboost', 'ldgd', or 'fast_ldgd'")


def aggregate_fold_report_results(fold_report_results):
    """
    Aggregates the fold report results to calculate the mean and standard deviation
    for precision, recall, f1-score, and support for each class across multiple folds.

    Args:
        fold_report_results (dict): A dictionary where keys are method names and values
                                    are lists of classification report dictionaries (one per fold).

    Returns:
        dict: A dictionary where keys are method names and values are DataFrames with the mean and std
              for precision, recall, f1-score, and support across folds.
    """
    aggregated_results = {}

    for method, reports in fold_report_results.items():
        metrics = ['precision', 'recall', 'f1-score', 'support']
        all_targets = list(reports[0].keys())

        # Initialize dictionary to store aggregated metrics
        aggregated_data = {target: {metric: [] for metric in metrics} for target in all_targets}

        # Collect data across all folds
        for report in reports:
            for target, values in report.items():
                if target not in ['accuracy']:
                    for metric in metrics:
                        aggregated_data[target][metric].append(values[metric])

        # Calculate mean and std for each metric
        result_dict = {}
        for target, metrics_dict in aggregated_data.items():
            result_dict[target] = {}
            for metric, values in metrics_dict.items():
                result_dict[target][f'{metric}_mean'] = np.mean(values)
                result_dict[target][f'{metric}_std'] = np.std(values)

        # Convert to DataFrame
        aggregated_results[method] = pd.DataFrame(result_dict).T

    return aggregated_results
