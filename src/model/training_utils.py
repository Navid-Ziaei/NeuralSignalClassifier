import numpy as np
from sklearn.metrics import f1_score, classification_report, accuracy_score, precision_score, recall_score
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
from sklearn.model_selection import train_test_split


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
                                                             batch_size=settings.batch_size)
        model.save_wights(path_save=paths.path_model)

        num_figures = len(loss_dict)

        fig, axs = plt.subplots(num_figures, 1, figsize=(10, 5 * num_figures))

        for i, (key, value) in enumerate(loss_dict.items()):
            axs[i].plot(value)
            axs[i].set_title(key)
            axs[i].set_xlabel('Epoch')
            axs[i].set_ylabel(key)
        plt.tight_layout()
        plt.show()

        with open(paths.path_model + 'model_settings.json', 'w') as f:
            json.dump(model_settings, f, indent=2)
    else:
        losses = []
        model.load_weights(paths.path_model)

    predictions, metrics, history_test = model.evaluate(yn_test=data_test, ys_test=labels_test,
                                                        epochs=settings.num_epochs_test,
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
        X_test = model.x_test.q_mu.detach().cpu().numpy()
        std_test = model.x_test.q_sigma.detach().numpy()
    else:
        X_test = model.x_test.q_mu.detach().cpu().numpy()
        std_test = torch.nn.functional.softplus(model.x_test.q_log_sigma).detach().cpu().numpy()

    # plot the heatmap of the latent space
    inducing_points = (history_test['z_list_reg'][-1], history_test['z_list_cls'][-1])

    visualization.plot_heatmap(X, labels_train, model, alpha_cls, cmap='binary', range_scale=1.2,
                               file_name='latent_heatmap_train', inducing_points=inducing_points,
                               save_path=paths.path_result,
                               device=device,
                               heat_map_mode='std', show_legend=False)

    visualization.plot_heatmap(X_test, labels_test, model, alpha_cls, cmap='binary', range_scale=1.2,
                               file_name='latent_heatmap_test', inducing_points=inducing_points,
                               save_path=paths.path_result,
                               device=device,
                               heat_map_mode='std', show_legend=False)


    visualization.animate_train(point_history=history_train['x_mu_list'],
                                labels=labels_train,
                                file_name='train_animation_with_inducing',
                                save_path=paths.path_result,
                                inverse_length_scale=alpha_cls,
                                inducing_points_history=(history_train['z_list_reg'], history_train['z_list_cls']))

    visualization.animate_train(point_history=history_test['x_mu_list'],
                                labels=labels_test,
                                file_name='test_animation_with_inducing',
                                save_path=paths.path_result,
                                inverse_length_scale=alpha_cls,
                                inducing_points_history=(history_test['z_list_reg'], history_test['z_list_cls']))


    visualization.plot_results_gplvm(X_test, std_test, labels=labels_test, losses=losses,
                                     inverse_length_scale=alpha_cls,
                                     latent_dim=model_settings['latent_dim'],
                                     save_path=paths.path_result, file_name=f'gplvm_test_result_all',
                                     show_errorbars=True)

    """
    inducing_points = (history_test['z_list_reg'][-1], history_test['z_list_cls'][-1])

    plot_heatmap(X, labels_train, model, alpha_cls, cmap='winter', range_scale=1.2,
                 file_name='latent_heatmap_train', inducing_points=inducing_points, save_path=paths.path_result[0])
    plot_heatmap(X_test, labels_test, model, alpha_cls, cmap='winter', range_scale=1.2,
                 file_name='latent_heatmap_test', inducing_points=inducing_points, save_path=paths.path_result[0])
    """

    return metrics


def train_fast_ldgd(data_train, labels_train, data_test, labels_test, y_train, y_test,
                    settings, paths):
    model_settings = {'data_dim': data_train.shape[-1], 'latent_dim': settings.latent_dim,
                      'num_inducing_points': settings.num_inducing_points, 'cls_weight': settings.cls_weight,
                      'reg_weight': 1.0, 'use_gpytorch': settings.use_gpytorch, 'use_shared_kernel': False,
                      'shared_inducing_points': False, 'early_stop': None}
    model_settings['latent_dim'] = 10
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

    data_train1, data_val, labels_train1, labels_val, y_train1, y_val = train_test_split(data_train, labels_train,
                                                                                         y_train_onehot,
                                                                                         test_size=0.2,
                                                                                         random_state=42,
                                                                                         stratify=labels_train)

    model = FastLDGD(torch.tensor(data_train1, dtype=torch.float32),
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

        losses, loss_dict, *_ = model.train_model(yn=data_train1, ys=y_train1,
                                                  epochs=settings.num_epochs_train,
                                                  batch_size=settings.batch_size,
                                                  yn_test=data_val,
                                                  ys_test=labels_val)
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
        plt.show()
        # early_stop=early_stop)
        model.save_wights(path_save=paths.path_model)

        with open(paths.path_model + 'model_settings.json', 'w') as f:
            json.dump(model_settings, f, indent=2)
    else:
        losses = []
        model.load_weights(paths.path_model)

    predictions, metrics, *_ = model.evaluate(yn_test=data_test, ys_test=labels_test,
                                              epochs=settings.num_epochs_test,
                                              save_path=paths.path_result)

    return metrics
