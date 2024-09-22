import os
import numpy as np
import torch
import gpytorch
import json
from LDGD.model.utils.kernels import ARDRBFKernel
from LDGD.model import LDGD, FastLDGD
from gpytorch.likelihoods import GaussianLikelihood, BernoulliLikelihood
from LDGD import visualization
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold


def train_fast_ldgd(data_train, labels_train, data_test, labels_test, y_train, y_test,
                    settings, paths, use_validation=True):
    save_path = paths.path_result + '/fast_ldgd/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    model_settings = settings.fast_ldgd_configs
    model_settings['data_dim'] = data_train.shape[-1]

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
        nn.LazyLinear(50),
        nn.ReLU(),
        nn.LazyLinear(hidden_dim1),
        nn.ReLU(),
        nn.LazyLinear(hidden_dim1),
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

        losses, loss_dict, history_train = model.train_model(yn=data_train, ys=y_train_onehot,
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
        plt.savefig(save_path + 'losses_fast_ldgd.png')
        plt.savefig(save_path + 'losses_fast_ldgd.svg')
        # plt.show()
        plt.close()
        plt.cla()
        # early_stop=early_stop)
        model.save_wights(path_save=save_path)

        with open(save_path + 'model_settings_fast_ldgd.json', 'w') as f:
            json.dump(model_settings, f, indent=2)
    else:
        losses = []
        model.load_weights(save_path)

    predictions, metrics, history_test, *_, report_results = model.evaluate(yn_test=data_test, ys_test=labels_test,
                                                                            epochs=settings.num_epochs_test,
                                                                            save_path=save_path)

    with open(save_path + 'fast_ldgd_classification_result.json', "w") as file:
        json.dump(metrics, file, indent=2)

    ########################
    if model_settings['use_gpytorch'] is False:
        alpha_reg = model.kernel_reg.alpha.detach().numpy()
        alpha_cls = model.kernel_cls.alpha.detach().numpy()
        X, std = model.x.encode(data_train.to(device))
    else:
        alpha_reg = 1 / model.kernel_reg.base_kernel.lengthscale.cpu().detach().numpy()
        alpha_cls = 1 / model.kernel_cls.base_kernel.lengthscale.cpu().detach().numpy()
        X, std = model.x.encode(data_train.to(device))
    X = X.cpu().detach().numpy()
    std = std.cpu().detach().numpy()

    visualization.plot_results_gplvm(X, np.sqrt(std), labels=np.squeeze(labels_train), losses=losses,
                                     inverse_length_scale=alpha_reg,
                                     latent_dim=model_settings['latent_dim'],
                                     save_path=save_path,
                                     file_name=f'gplvm_train_reg_result_all_ldgd',
                                     show_errorbars=True)
    visualization.plot_results_gplvm(X, np.sqrt(std), labels=np.squeeze(labels_train), losses=losses,
                                     inverse_length_scale=alpha_cls,
                                     latent_dim=model_settings['latent_dim'],
                                     save_path=save_path,
                                     file_name=f'gplvm_train_cls_result_all_ldgd',
                                     show_errorbars=True)

    X_test, std_test = model.x.encode(data_test.to(device))
    X_test = X_test.cpu().detach().numpy()
    std_test = std_test.cpu().detach().numpy()

    # plot the heatmap of the latent space
    inducing_points = (history_train['z_list_reg'][-1], history_train['z_list_cls'][-1])
    history_test = model.history_test

    visualization.plot_heatmap(X, np.squeeze(labels_train), model, alpha_cls, cmap='binary', range_scale=1.2,
                               file_name='latent_heatmap_train_ldgd', inducing_points=inducing_points,
                               save_path=save_path,
                               device=device,
                               heat_map_mode='prob', show_legend=False)

    visualization.plot_heatmap(X_test, np.squeeze(labels_test), model, alpha_cls, cmap='binary', range_scale=1.2,
                               file_name='latent_heatmap_test_ldgd', inducing_points=inducing_points,
                               save_path=save_path,
                               device=device,
                               heat_map_mode='prob', show_legend=False)

    visualization.animate_train(point_history=history_train['x_mu_list'],
                                labels=np.squeeze(labels_train),
                                file_name='train_animation_with_inducing_ldgd',
                                save_path=save_path,
                                inverse_length_scale=alpha_cls,
                                inducing_points_history=(history_train['z_list_reg'], history_train['z_list_cls']))

    visualization.animate_train(point_history=history_test['x_mu_list'],
                                labels=np.squeeze(labels_val),
                                file_name='test_animation_with_inducing_ldgd',
                                save_path=save_path,
                                inverse_length_scale=alpha_cls,
                                inducing_points_history=(history_test['z_list_reg'], history_test['z_list_cls']))

    return metrics, report_results
