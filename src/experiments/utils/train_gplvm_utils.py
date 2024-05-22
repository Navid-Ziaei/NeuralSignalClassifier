import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy.stats import pointbiserialr
from LDGD.model.utils.kernels import ARDRBFKernel
from LDGD.model import LDGD, FastLDGD
from LDGD.visualization import plot_results_gplvm, plot_heatmap
from gpytorch.likelihoods import GaussianLikelihood, BernoulliLikelihood
# from src.model.Bayesian_JointGPLVM import *
from LDGD import visualization
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE


def train_xgb(data_train, labels_train, data_test, labels_test, paths, balance_method='weighting',
              selected_features=None):
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
        feature_importance = {selected_features[i]: model.feature_importances_[i] for i in
                              range(len(selected_features))}
        print("top 10 important features are: ",
              sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])

    # Make predictions
    predictions = model.predict(data_test)

    # Calculate the F1-score
    f1 = f1_score(labels_test, predictions, average='macro')
    print(f"F1-score: {f1 * 100:.2f}%")

    report = classification_report(y_true=labels_test, y_pred=predictions)
    print(report)
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

    return metrics


def train_ldgd(data_train, labels_train, data_test, labels_test, y_train, y_test,
               model_settings, paths,
               shared_inducing_points=False,
               use_shared_kernel=False,
               cls_weight=1.0,
               reg_weight=1.0,
               early_stop=None):
    model_settings['data_dim'] = data_train.shape[-1]
    batch_shape = torch.Size([model_settings['data_dim']])


    likelihood_reg = GaussianLikelihood(batch_shape=batch_shape)
    likelihood_cls = BernoulliLikelihood()

    data_train = torch.tensor(data_train, dtype=torch.float32)
    data_test = torch.tensor(data_test, dtype=torch.float32)
    y_train_onehot = torch.tensor(y_train)
    y_test_onehot = torch.tensor(y_test)

    """model = JointGPLVM_Bayesian(torch.tensor(data_train, dtype=torch.float32),
                                kernel_reg=kernel_reg,
                                kernel_cls=kernel_cls,
                                num_classes=y_train_onehot.shape[-1],
                                latent_dim=model_settings['latent_dim'],
                                num_inducing_points=model_settings['num_inducing_points'],
                                likelihood_reg=likelihood_reg,
                                likelihood_cls=likelihood_cls,
                                use_gpytorch=model_settings['use_gpytorch'])"""

    """model = LDGD(torch.tensor(data_train, dtype=torch.float32),
                 kernel_reg=kernel_reg,
                 kernel_cls=kernel_cls,
                 num_classes=y_train_onehot.shape[-1],
                 latent_dim=model_settings['latent_dim'],
                 num_inducing_points_reg=model_settings['num_inducing_points'],
                 num_inducing_points_cls=model_settings['num_inducing_points'],
                 likelihood_reg=likelihood_reg,
                 likelihood_cls=likelihood_cls,
                 use_gpytorch=model_settings['use_gpytorch'],
                 shared_inducing_points=False,
                 cls_weight=cls_weight,
                 reg_weight=reg_weight,
                 random_state=42,
                 x_init=None)"""

    model_settings['latent_dim'] = 2
    model_settings['num_inducing_points'] = 25
    if model_settings['use_gpytorch'] is False:
        kernel_cls = ARDRBFKernel(input_dim=model_settings['latent_dim'])
        kernel_reg = ARDRBFKernel(input_dim=model_settings['latent_dim'])
    else:
        kernel_reg = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=model_settings['latent_dim']))
        kernel_cls = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=model_settings['latent_dim']))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                     shared_inducing_points=False,
                     cls_weight=cls_weight,
                     reg_weight=reg_weight,
                     device=device)
    # shared_inducing_points=shared_inducing_points,
    # use_shared_kernel=use_shared_kernel,
    # cls_weight=cls_weight,
    # reg_weight=reg_weight)

    if model_settings['load_trained_model'] is False:
        losses, *_ = model.train_model(yn=data_train, ys=y_train_onehot,
                                       epochs=model_settings['num_epochs_train'],
                                       batch_size=model_settings['batch_size'], yn_test=data_test, ys_test=labels_test)
        # early_stop=early_stop)
        model.save_wights(path_save=paths.path_model)

        with open(paths.path_model + 'model_settings.json', 'w') as f:
            json.dump(model_settings, f, indent=2)
    else:
        losses = []
        model.load_weights(paths.path_model)

    predictions, metrics, *_ = model.evaluate(yn_test=data_test, ys_test=labels_test,
                                              epochs=model_settings['num_epochs_test'] * 2,
                                              save_path=paths.path_result)

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
                                     save_path=paths.path_result, file_name=f'gplvm_train_reg_result_all',
                                     show_errorbars=True)
    visualization.plot_results_gplvm(X, np.sqrt(std), labels=labels_train, losses=losses,
                                     inverse_length_scale=alpha_cls,
                                     latent_dim=model_settings['latent_dim'],
                                     save_path=paths.path_result, file_name=f'gplvm_train_cls_result_all',
                                     show_errorbars=True)

    if model_settings['use_gpytorch'] is False:
        X_test = model.x_test.q_mu.detach().numpy()
        std_test = model.x_test.q_sigma.detach().numpy()
    else:
        X_test = model.x_test.q_mu.detach().numpy()
        std_test = torch.nn.functional.softplus(model.x_test.q_log_sigma).detach().numpy()

    visualization.plot_results_gplvm(X_test, std_test, labels=labels_test, losses=losses,
                                     inverse_length_scale=alpha_cls,
                                     latent_dim=model_settings['latent_dim'],
                                     save_path=paths.path_result, file_name=f'gplvm_test_result_all',
                                     show_errorbars=True)

    """inducing_points = (history_test['z_list_reg'][-1], history_test['z_list_cls'][-1])

    plot_heatmap(X, labels_train, model, alpha_cls, cmap='winter', range_scale=1.2,
                 file_name='latent_heatmap_train', inducing_points=inducing_points, save_path=paths.path_result[0])
    plot_heatmap(X_test, labels_test, model, alpha_cls, cmap='winter', range_scale=1.2,
                 file_name='latent_heatmap_test', inducing_points=inducing_points, save_path=paths.path_result[0])"""

    return metrics


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


def get_selected_features(features_df, model_settings, paths, fold_idx, train_index,
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
    if model_settings['features'].lower() == 'all':
        selected_features = features_df.columns.to_list()
    elif model_settings['features'].lower() == 'corr':
        selected_features = get_correlation(features_df.copy().reset_index(drop=True).loc[train_index, :].copy(),
                                            model_settings['binary_column'],
                                            paths)
    elif model_settings['features'].lower() == 'pre_selected':
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

    print(f"Feature selection mode: {model_settings['features'].lower()}, Number of features {len(selected_features)}")

    # data transormation : "Normalization", "standardize", None
    if model_settings['feature_transformation'] is not None:
        if model_settings['feature_transformation'].lower() == 'normalize':
            features_df = features_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        elif model_settings['feature_transformation'].lower() == 'standardize':
            features_df = features_df.apply(lambda x: (x - x.mean()) / x.std())
        else:
            raise ValueError("The transformation is not defined")

    features_matrix = features_df[selected_features].values

    return features_matrix, selected_features, patients_ids, patients_files


def plot_datablocks_histogram(data):
    n_blocks = len(data['block'])

    fig, ax = plt.subplots(figsize=(14, 8))

    # Set position of bar on X axis
    barWidth = 0.25
    r1 = np.arange(n_blocks)
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    # Function to add value labels
    def add_labels(bars, data):
        for bar, value in zip(bars, data):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{value}', ha='center', va='bottom')

    # Make the plot for total numbers
    ax.bar(r1, data['num_ctl'], color='b', width=barWidth, edgecolor='grey', label='Control')
    ax.bar(r2, data['num_experienced'], color='g', width=barWidth, edgecolor='grey', label='Experienced')
    ax.bar(r3, data['num_not_experienced'], color='r', width=barWidth, edgecolor='grey', label='Not Experienced')

    # Overlay correct answers with hatching
    bars_ctl_correct = ax.bar(r1, data['num_ctl_correct'], color='b', width=barWidth, edgecolor='black', hatch='//',
                              alpha=1,
                              label='Correct Control (hatched)')
    bars_exp_correct = ax.bar(r2, data['num_experienced_correct'], color='g', width=barWidth, edgecolor='black',
                              hatch='//', alpha=1,
                              label='Correct Experienced (hatched)')
    bars_not_exp_correct = ax.bar(r3, data['num_not_experienced_correct'], color='r', width=barWidth,
                                  edgecolor='black', hatch='//',
                                  alpha=1, label='Correct Not Experienced (hatched)')

    # Add value labels
    add_labels(bars_ctl_correct, np.round(np.array(data['num_ctl_correct']) / np.array(data['num_ctl']), 2))
    add_labels(bars_exp_correct,
               np.round(np.array(data['num_experienced_correct']) / np.array(data['num_experienced']), 2))
    add_labels(bars_not_exp_correct,
               np.round(np.array(data['num_not_experienced_correct']) / np.array(data['num_not_experienced']), 2))

    # Add xticks on the middle of the group bars
    ax.set_xlabel('Block', fontweight='bold')
    ax.set_ylabel('Number of trials', fontweight='bold')
    ax.set_xticks([r + barWidth for r in range(n_blocks)])
    ax.set_xticklabels([type[0] for type in data['type']], fontsize=16)

    # Simplify legend (Optional: Comment out the 'not in' condition to show all legends)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handle for i, handle in enumerate(handles) if "hatched" in labels[i]],
              [label for i, label in enumerate(labels) if "hatched" in labels[i]])

    plt.title('Number of Trials and Correct Answers by Stimuli Type and Experience Level per Block')
    plt.show()


def get_labels(features_df, model_settings):
    patients_ids = features_df['id'].values
    patients_files = features_df['subject_file'].values[0]

    if model_settings['binary_column'] == 'old_new':
        labels_array = features_df['old_new'].values
    elif model_settings['binary_column'] == 'decision':
        if len(np.unique(features_df['decision'].values)) > 2:
            labels_array = features_df['decision'].values * 2
        else:
            labels_array = features_df['decision'].values
    elif model_settings['binary_column'] == 'is_experienced':
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
    elif model_settings['binary_column'] == 'go_nogo':
        mapping = {'go': 1, 'nogo': 0}

        # Apply the mapping to the DataFrame column
        # features_df = features_df[features_df['is_correct'] == True]
        # features_df = features_df[(features_df['block_type'] == 'i+e') | (features_df['block_type'] == 'i-e')]
        features_df['exp_label'] = features_df['go_nogo'].map(mapping)

        labels_array = features_df['is_experienced'].values
    else:
        labels_array = features_df[model_settings['binary_column']].values

    # ###################### Train test split  ######################
    y_one_hot = np.zeros((labels_array.shape[0], len(np.unique(labels_array))))
    y_one_hot[np.arange(labels_array.shape[0]), np.uint(labels_array)] = 1
    unique_pids = np.unique(patients_ids)

    return y_one_hot, labels_array, unique_pids, patients_files, features_df
