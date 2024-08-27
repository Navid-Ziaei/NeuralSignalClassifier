import os
import pandas as pd
from sklearn.metrics import f1_score, classification_report, accuracy_score, precision_score, recall_score, \
    roc_auc_score
import numpy as np
import torch
import xgboost as xgb
import gpytorch
import json
from LDGD.model.utils.kernels import ARDRBFKernel
from LDGD.model import LDGD, FastLDGD
from gpytorch.likelihoods import GaussianLikelihood, BernoulliLikelihood
from LDGD import visualization
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold

from src.evaluation.evaluation import ResultList
from src.utils import get_labels, get_correlation_single_event, get_drop_columns, get_selected_features
from src.visualization.visualization_utils import plot_metric_heatmaps


def train_model_with_folds(features_raw_df_dict, settings, paths, enable_group_level_result=False):
    """
    Train models using either k-fold cross-validation or a single train/test split for each patient's data.

    This function performs the following steps for each patient's data:
    1. Prepares the data by removing certain columns.
    2. Extracts labels and features from the raw data.
    3. Depending on the mode, either generates k-folds or a single train/test split.
    4. Trains models specified in the settings on the training data for each fold or split.
    5. Evaluates the models on the test data and logs the results.
    6. Computes and saves average scores for each model across all folds (if in k-fold mode).
    7. Optionally saves group-level results and overall results to CSV files.

    Args:
        settings (Settings): The settings object containing configurations.
        paths (Paths): The paths object containing file paths.
        enable_group_level_result (bool): Whether to save group-level results. Defaults to False.

    Returns:
        None: The function saves results to CSV files and updates the results logger.
    """
    results_logger = ResultList(method_list=settings.method_list, metric_list=settings.metric_list)
    all_patient_group_results = {method: [] for method in settings.method_list}

    for patient_id, (patient_file_name, features_raw_df) in enumerate(features_raw_df_dict.items()):
        print(f"====== Subject {patient_id} ({patient_file_name.split('.')[0]}) ====== \n")

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
            cv_mode = setup_cross_validation(settings.cross_validation_mode, num_folds=settings.num_fold)
            folds = generate_folds(features_df, labels_array, cv_mode, n_splits=settings.num_fold)
        else:  # Single train/test split
            folds = [(train_test_split(np.arange(len(labels_array)), test_size=0.2, stratify=labels_array))]

        fold_results = {method: [] for method in settings.method_list}
        fold_results_group = {method: [] for method in settings.method_list}
        fold_report_results = {method: [] for method in settings.method_list}

        for fold_idx, (train_index, test_index) in enumerate(folds):
            paths.create_fold_path(fold_idx)

            # Select features and prepare the training and testing datasets
            features_matrix, selected_features, patients_ids, _ = get_selected_features(
                features_df=features_df.copy(), settings=settings, paths=paths,
                fold_idx=fold_idx, train_index=train_index, target_columns_drop=get_drop_columns(settings))

            data_train, data_test = features_matrix[train_index], features_matrix[test_index]
            labels_train, labels_test = labels_array[train_index], labels_array[test_index]
            y_train, y_test = y_one_hot[train_index], y_one_hot[test_index]

            # Train and evaluate models for each specified method
            for method in settings.method_list:
                print(f"=========== Train Subject {patient_id} Fold {fold_idx} Model {method} =========== \n")
                results, group_result, report_results = train_model(method, data_train, labels_train, data_test,
                                                                    labels_test, settings,
                                                                    paths, y_train, y_test,
                                                                    selected_features=selected_features,
                                                                    target_columns=target_columns)
                fold_results[method].append(results)
                fold_results_group[method].append(group_result)
                fold_report_results[method].append(report_results)

            plt.close('all')

        # Aggregate the results for fold reports
        aggregated_results = aggregate_fold_report_results(fold_report_results)

        mean_grid_precision, _ = plot_metric_heatmaps(aggregated_results, metric='precision', grid_size=(10, 10), save_dir=paths.results_base_path)
        mean_grid_recall, _ = plot_metric_heatmaps(aggregated_results, metric='recall', grid_size=(10, 10), save_dir=paths.results_base_path)
        mean_grid_f1, _ = plot_metric_heatmaps(aggregated_results, metric='f1-score', grid_size=(10, 10), save_dir=paths.results_base_path)

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

            # Save group-level results
            # save_group_results(fold_results_group, method, paths.results_base_path)

        # Update results across all patients
        if enable_group_level_result:
            all_patient_group_results.update(
                {key: all_patient_group_results.get(key, []) + fold_results_group[key] for key in
                 fold_results_group.keys()})

    # Save overall results to CSV
    result_df = results_logger.to_dataframe()
    result_df.to_csv(os.path.join(paths.base_path, paths.folder_name, f'{settings.cross_validation_mode}_results.csv'))

    # Save group results for each method
    if enable_group_level_result:
        for key in all_patient_group_results.keys():
            df_gp = pd.DataFrame(all_patient_group_results[key], index=settings.patient)
            df_gp.to_csv(os.path.join(paths.base_path, paths.folder_name, f'group_results_{key}.csv'))


def train_xgb(data_train, labels_train, data_test, labels_test, paths, balance_method='weighting',
              selected_features=None, target_columns=None):
    # Create and train the XGBoost model with class weights

    if len(np.unique(labels_train)) > 2:
        model = xgb.XGBClassifier(objective="multi:softmax", num_class=2)
    else:
        scale_pos_weight = 1
        if balance_method == 'smote':
            smote = SMOTE(random_state=42)
            data_train, labels_train = smote.fit_resample(data_train, labels_train)
        elif balance_method == 'weighting':
            scale_pos_weight = 2 * np.sum(1 - labels_train) / np.sum(labels_train)

        model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight,
                                  max_depth=6,
                                  num_parallel_tree=2)

    model.fit(data_train, labels_train)
    if selected_features is not None:
        key_list = list(selected_features.keys())
        feature_list = selected_features[key_list[0]]
        feature_importance = {feature_list[i]: model.feature_importances_[i] for i in
                              range(len(feature_list))}
        print("top 10 important features are: ",
              sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])

    # Make predictions
    predictions = model.predict(data_test)
    group_result = {}
    for uvalue in np.unique(labels_test):
        print(f"Class {uvalue} has {np.sum(labels_test == uvalue)} samples")
        # use majority voting
        group_result[uvalue] = np.mean(predictions[labels_test == uvalue])

    report = classification_report(y_true=labels_test, y_pred=predictions, target_names=target_columns)
    print(report)

    report_results = classification_report(y_true=labels_test, y_pred=predictions, output_dict=True,
                                           target_names=target_columns)

    metrics = {
        'accuracy': accuracy_score(labels_test, predictions),
        'precision': precision_score(labels_test, predictions, average='weighted'),
        'recall': recall_score(labels_test, predictions, average='weighted'),
        'f1_score': f1_score(labels_test, predictions, average='weighted')
    }

    with open(paths.path_result + 'xgb_classification_report.txt', "w") as file:
        file.write(report)

    with open(paths.path_result + 'xgb_classification_result.json', "w") as file:
        json.dump(metrics, file, indent=2)

    return metrics, group_result, report_results


def train_ldgd(data_train, labels_train, data_test, labels_test, y_train, y_test,
               settings, paths):
    model_settings = {'data_dim': data_train.shape[-1],
                      'latent_dim': settings.latent_dim,
                      'num_inducing_points': settings.num_inducing_points,
                      'cls_weight': settings.cls_weight,
                      'reg_weight': 1.0,
                      'use_gpytorch': settings.use_gpytorch,
                      'use_shared_kernel': False,
                      'shared_inducing_points': settings.shared_inducing_points,
                      'early_stop': None,
                      'load_trained_model': False}

    batch_shape = torch.Size([model_settings['data_dim']])

    likelihood_reg = GaussianLikelihood(batch_shape=batch_shape)
    likelihood_cls = BernoulliLikelihood()

    data_train = torch.tensor(data_train, dtype=torch.float32)
    data_test = torch.tensor(data_test, dtype=torch.float32)
    y_train_onehot = torch.tensor(y_train)
    y_test_onehot = torch.tensor(y_test)

    if model_settings['use_gpytorch'] is False:
        kernel_cls = ARDRBFKernel(input_dim=model_settings['latent_dim'])
        kernel_reg = ARDRBFKernel(input_dim=model_settings['latent_dim'])
    else:
        kernel_reg = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=model_settings['latent_dim']))
        kernel_cls = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=model_settings['latent_dim']))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LDGD(torch.tensor(data_train, dtype=torch.float32),
                 kernel_reg=kernel_reg,
                 kernel_cls=kernel_cls,
                 num_classes=y_train_onehot.shape[-1],
                 latent_dim=model_settings['latent_dim'],
                 num_inducing_points_reg=model_settings['num_inducing_points'],
                 num_inducing_points_cls=model_settings['num_inducing_points'],
                 likelihood_reg=likelihood_reg,
                 likelihood_cls=likelihood_cls,
                 use_gpytorch=model_settings['use_gpytorch'],
                 shared_inducing_points=model_settings['shared_inducing_points'],
                 cls_weight=model_settings['cls_weight'],
                 reg_weight=model_settings['reg_weight'],
                 device=device)

    if settings.load_trained_model is False:
        losses, loss_dict, history_train = model.train_model(yn=data_train, ys=y_train_onehot,
                                                             epochs=settings.num_epochs_train,
                                                             batch_size=settings.batch_size,
                                                             save_best_result=True,
                                                             path_save=paths.path_model)
        model.load_weights(paths.path_model)
        # model.save_wights(path_save=paths.path_model)

        num_figures = len(loss_dict)

        fig, axs = plt.subplots(num_figures, 1, figsize=(10, 5 * num_figures))

        for i, (key, value) in enumerate(loss_dict.items()):
            axs[i].plot(value)
            axs[i].set_title(key)
            axs[i].set_xlabel('Epoch')
            axs[i].set_ylabel(key)
        plt.tight_layout()
        plt.savefig(paths.path_result + 'losses_train_ldgd.png')
        plt.savefig(paths.path_result + 'losses_train_ldgd.svg')
        plt.cla()
        plt.close()
        #plt.show()

        with open(paths.path_model + 'model_settings_ldgd.json', 'w') as f:
            json.dump(model_settings, f, indent=2)
    else:
        losses = []
        model.load_weights(paths.path_model)

    predictions, metrics, history_test, loss_terms_test, report_results = model.evaluate(yn_test=data_test,
                                                                                         ys_test=labels_test,
                                                                                         epochs=settings.num_epochs_test,
                                                                                         save_path=paths.path_result)
    group_result = {}
    for uvalue in np.unique(labels_test):
        print(f"Class {uvalue} has {np.sum(labels_test == uvalue)} samples")
        group_result[uvalue] = np.mean(predictions[labels_test == uvalue])

    with open(paths.path_result + 'ldgd_classification_result.json', "w") as file:
        json.dump(metrics, file, indent=2)
    num_figures = len(loss_terms_test)
    fig, axs = plt.subplots(num_figures, 1, figsize=(10, 5 * num_figures))
    for i, (key, value) in enumerate(loss_terms_test.items()):
        axs[i].plot(value)
        axs[i].set_title(key)
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(key)
    plt.tight_layout()
    plt.savefig(paths.path_result + 'losses_test_ldgd.png')
    plt.savefig(paths.path_result + 'losses_test_ldgd.svg')
    #plt.show()
    plt.cla()
    plt.close()

    if model_settings['use_gpytorch'] is False:
        alpha_reg = model.kernel_reg.alpha.detach().numpy()
        alpha_cls = model.kernel_cls.alpha.detach().numpy()
        X = model.x.q_mu.detach().numpy()
        std = model.x.q_sigma.detach().numpy()
    else:
        alpha_reg = 1 / model.kernel_reg.base_kernel.lengthscale.cpu().detach().numpy()
        alpha_cls = 1 / model.kernel_cls.base_kernel.lengthscale.cpu().detach().numpy()
        X = model.x.q_mu.detach().cpu().numpy()
        std = torch.nn.functional.softplus(model.x.q_log_sigma).cpu().detach().numpy()

    visualization.plot_results_gplvm(X, np.sqrt(std), labels=labels_train, losses=losses,
                                     inverse_length_scale=alpha_reg,
                                     latent_dim=model_settings['latent_dim'],
                                     save_path=paths.path_result,
                                     file_name=f'gplvm_train_reg_result_all_ldgd',
                                     show_errorbars=True)
    visualization.plot_results_gplvm(X, np.sqrt(std), labels=labels_train, losses=losses,
                                     inverse_length_scale=alpha_cls,
                                     latent_dim=model_settings['latent_dim'],
                                     save_path=paths.path_result,
                                     file_name=f'gplvm_train_cls_result_all_ldgd',
                                     show_errorbars=True)

    if model_settings['use_gpytorch'] is False:
        X_test = model.x_test.q_mu.detach().cpu().numpy()
        std_test = model.x_test.q_sigma.detach().numpy()
    else:
        X_test = model.x_test.q_mu.detach().cpu().numpy()
        std_test = torch.nn.functional.softplus(model.x_test.q_log_sigma).detach().cpu().numpy()

    # plot the heatmap of the latent space
    inducing_points = (history_test['z_list_reg'][-1], history_test['z_list_cls'][-1])

    visualization.plot_heatmap(X, labels_train, model, alpha_cls, cmap='binary', range_scale=1.2,
                               file_name='latent_heatmap_train_ldgd', inducing_points=inducing_points,
                               save_path=paths.path_result,
                               device=device,
                               heat_map_mode='std', show_legend=False)

    visualization.plot_heatmap(X_test, labels_test, model, alpha_cls, cmap='binary', range_scale=1.2,
                               file_name='latent_heatmap_test_ldgd', inducing_points=inducing_points,
                               save_path=paths.path_result,
                               device=device,
                               heat_map_mode='std', show_legend=False)

    visualization.animate_train(point_history=history_train['x_mu_list'],
                                labels=labels_train,
                                file_name='train_animation_with_inducing_ldgd',
                                save_path=paths.path_result,
                                inverse_length_scale=alpha_cls,
                                inducing_points_history=(history_train['z_list_reg'], history_train['z_list_cls']))

    visualization.animate_train(point_history=history_test['x_mu_list'],
                                labels=labels_test,
                                file_name='test_animation_with_inducing_ldgd',
                                save_path=paths.path_result,
                                inverse_length_scale=alpha_cls,
                                inducing_points_history=(history_test['z_list_reg'], history_test['z_list_cls']))

    visualization.plot_results_gplvm(X_test, std_test, labels=labels_test, losses=losses,
                                     inverse_length_scale=alpha_cls,
                                     latent_dim=model_settings['latent_dim'],
                                     save_path=paths.path_result, file_name=f'gplvm_test_result_all_ldgd',
                                     show_errorbars=True)

    """
    inducing_points = (history_test['z_list_reg'][-1], history_test['z_list_cls'][-1])

    plot_heatmap(X, labels_train, model, alpha_cls, cmap='winter', range_scale=1.2,
                 file_name='latent_heatmap_train', inducing_points=inducing_points, save_path=paths.path_result[0])
    plot_heatmap(X_test, labels_test, model, alpha_cls, cmap='winter', range_scale=1.2,
                 file_name='latent_heatmap_test', inducing_points=inducing_points, save_path=paths.path_result[0])
    """

    return metrics, group_result, report_results


def train_fast_ldgd(data_train, labels_train, data_test, labels_test, y_train, y_test,
                    settings, paths, use_validation=True):
    model_settings = {'data_dim': data_train.shape[-1], 'latent_dim': 10,
                      'num_inducing_points': settings.num_inducing_points, 'cls_weight': settings.cls_weight,
                      'reg_weight': 1.0, 'use_gpytorch': settings.use_gpytorch, 'use_shared_kernel': False,
                      'shared_inducing_points': False, 'early_stop': None}
    batch_shape = torch.Size([model_settings['data_dim']])

    likelihood_reg = GaussianLikelihood(batch_shape=batch_shape)
    likelihood_cls = BernoulliLikelihood()

    data_train = torch.tensor(data_train, dtype=torch.float32)
    data_test = torch.tensor(data_test, dtype=torch.float32)
    y_train_onehot = torch.tensor(y_train)
    y_test_onehot = torch.tensor(y_test)

    if model_settings['use_gpytorch'] is False:
        kernel_cls = ARDRBFKernel(input_dim=model_settings['latent_dim'])
        kernel_reg = ARDRBFKernel(input_dim=model_settings['latent_dim'])
    else:
        kernel_reg = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=model_settings['latent_dim']))
        kernel_cls = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=model_settings['latent_dim']))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_dim1 = 100
    hidden_dim2 = 50
    encoder = nn.Sequential(
        nn.LazyLinear(300),
        nn.ReLU(),
        nn.LazyLinear(hidden_dim1),
        nn.ReLU(),
        nn.LazyLinear(hidden_dim1),
        nn.ReLU(),
        nn.LazyLinear(hidden_dim2),
        nn.ReLU()
    )

    if use_validation is True:
        data_train, data_val, labels_train, labels_val, y_train_onehot, y_val = train_test_split(data_train,
                                                                                                 labels_train,
                                                                                                 y_train_onehot,
                                                                                                 test_size=0.2,
                                                                                                 random_state=42,
                                                                                                 stratify=labels_train)
    else:
        data_val = torch.Tensor(data_test)
        labels_val = labels_test
        y_val = torch.Tensor(y_test_onehot)

    model = FastLDGD(torch.tensor(data_train, dtype=torch.float32),
                     kernel_reg=kernel_reg,
                     kernel_cls=kernel_cls,
                     num_classes=y_train_onehot.shape[-1],
                     latent_dim=model_settings['latent_dim'],
                     num_inducing_points_reg=model_settings['num_inducing_points'],
                     num_inducing_points_cls=model_settings['num_inducing_points'],
                     likelihood_reg=likelihood_reg,
                     likelihood_cls=likelihood_cls,
                     use_gpytorch=model_settings['use_gpytorch'],
                     shared_inducing_points=model_settings['shared_inducing_points'],
                     cls_weight=model_settings['cls_weight'],
                     reg_weight=model_settings['reg_weight'],
                     device=device,
                     nn_encoder=encoder)

    if settings.load_trained_model is False:
        # spilit train to train and validation using 90% for training and 10% for validation skleren train_test_split

        losses, loss_dict, *_ = model.train_model(yn=data_train, ys=y_train_onehot,
                                                  epochs=settings.num_epochs_train,
                                                  batch_size=settings.batch_size,
                                                  yn_test=data_val,
                                                  ys_test=labels_val,
                                                  save_best_result=True,
                                                  path_save=paths.path_model)
        model.load_weights(paths.path_model)
        num_figures = len(loss_dict)

        fig, axs = plt.subplots(num_figures, 1, figsize=(10, 5 * num_figures))
        # find maximum of test loss
        max_test_loss_arg = np.argmax(loss_dict['accuracy_test'])

        for i, (key, value) in enumerate(loss_dict.items()):
            axs[i].plot(value)
            axs[i].set_title(key)
            axs[i].set_xlabel('Epoch')
            axs[i].set_ylabel(key)
            axs[i].axvline(x=max_test_loss_arg, color='r', linestyle='--')
        plt.tight_layout()
        plt.savefig(paths.path_result + 'losses_fast_ldgd.png')
        plt.savefig(paths.path_result + 'losses_fast_ldgd.svg')
        # plt.show()
        plt.close()
        plt.cla()
        # early_stop=early_stop)
        model.save_wights(path_save=paths.path_model)

        with open(paths.path_model + 'model_settings_fast_ldgd.json', 'w') as f:
            json.dump(model_settings, f, indent=2)
    else:
        losses = []
        model.load_weights(paths.path_model)

    predictions, metrics, *_, report_results = model.evaluate(yn_test=data_test, ys_test=labels_test,
                                                              epochs=settings.num_epochs_test,
                                                              save_path=paths.path_result)

    group_result = {}
    for uvalue in np.unique(labels_test):
        print(f"Class {uvalue} has {np.sum(labels_test == uvalue)} samples")
        group_result[uvalue] = np.mean(predictions[labels_test == uvalue])

    with open(paths.path_result + 'fast_ldgd_classification_result.json', "w") as file:
        json.dump(metrics, file, indent=2)

    return metrics, group_result


def save_group_results(fold_results_group, method, results_base_path):
    """
    Calculate and save group-level metrics based on the results from k-fold cross-validation.

    This function aggregates the true values and predictions across all folds for a specific
    method, calculates several evaluation metrics (accuracy, precision, recall, F1 score, and AUC),
    and saves these metrics to a text file.

    Args:
        fold_results_group (dict): A dictionary containing the group results for each method,
                                   where the keys are method names and the values are lists of
                                   dictionaries with true and predicted values.
        method (str): The name of the method for which to calculate and save the metrics.
        results_base_path (str): The base directory where the results file should be saved.

    Returns:
        None: The function saves the calculated metrics to a text file and does not return any value.
    """
    true_values, predictions = [], []

    # Aggregate true values and predictions from all folds
    for gp_result in fold_results_group[method]:
        for true_value, predicted_value in gp_result.items():
            true_values.append(true_value)
            predictions.append(predicted_value)

    # Convert lists to numpy arrays
    true_values, predictions = np.array(true_values), np.array(predictions)

    # Calculate group-level metrics
    group_result = {
        'accuracy': accuracy_score(true_values, predictions.round()),
        'precision': precision_score(true_values, predictions.round(), average='binary'),
        'recall': recall_score(true_values, predictions.round(), average='binary'),
        'f1': f1_score(true_values, predictions.round(), average='binary'),
        'auc': roc_auc_score(true_values, predictions)
    }

    # Save the metrics to a text file
    with open(os.path.join(results_base_path, f'group_results_{method}.txt'), 'w') as f:
        f.write(str(group_result))


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
                               use_validation=True)
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