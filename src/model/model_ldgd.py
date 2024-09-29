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



def train_ldgd(data_train, labels_train, data_test, labels_test, y_train, y_test,
               settings, paths, monitor_mse=False):
    save_path = paths.path_result + '/ldgd/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    model_settings = settings.ldgd_configs
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
    model = LDGD(data_train,
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
                                                             path_save=save_path,
                                                             monitor_mse=monitor_mse,
                                                             early_stop=0.01)
        model.load_weights(save_path)
        # model.save_wights(path_save=paths.path_model)

        num_figures = len(loss_dict)

        fig, axs = plt.subplots(num_figures, 1, figsize=(10, 5 * num_figures))

        for i, (key, value) in enumerate(loss_dict.items()):
            axs[i].plot(value)
            axs[i].set_title(key)
            axs[i].set_xlabel('Epoch')
            axs[i].set_ylabel(key)
        plt.tight_layout()
        plt.savefig(save_path + 'losses_train_ldgd.png')
        plt.savefig(save_path + 'losses_train_ldgd.svg')
        plt.cla()
        plt.close()
        #plt.show()

        with open(save_path + 'model_settings_ldgd.json', 'w') as f:
            json.dump(model_settings, f, indent=2)
    else:
        losses = []
        model.load_weights(save_path)

    predictions, metrics, history_test, loss_terms_test, report_results = model.evaluate(yn_test=data_test,
                                                                                         ys_test=labels_test,
                                                                                         epochs=settings.num_epochs_test,
                                                                                         save_path=save_path,
                                                                                         monitor_mse=monitor_mse)

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
    plt.savefig(save_path + 'losses_test_ldgd.png')
    plt.savefig(save_path + 'losses_test_ldgd.svg')
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

    if X.shape[1]>1:
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

    if model_settings['use_gpytorch'] is False:
        X_test = model.x_test.q_mu.detach().cpu().numpy()
        std_test = model.x_test.q_sigma.detach().numpy()
    else:
        X_test = model.x_test.q_mu.detach().cpu().numpy()
        std_test = torch.nn.functional.softplus(model.x_test.q_log_sigma).detach().cpu().numpy()

    if X.shape[1] > 1:
        # plot the heatmap of the latent space
        inducing_points = (history_test['z_list_reg'][-1], history_test['z_list_cls'][-1])

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
                                    labels=np.squeeze(labels_test),
                                    file_name='test_animation_with_inducing_ldgd',
                                    save_path=save_path,
                                    inverse_length_scale=alpha_cls,
                                    inducing_points_history=(history_test['z_list_reg'], history_test['z_list_cls']))

        visualization.plot_results_gplvm(X_test, std_test, labels=np.squeeze(labels_test), losses=losses,
                                         inverse_length_scale=alpha_cls,
                                         latent_dim=model_settings['latent_dim'],
                                         save_path=save_path, file_name=f'gplvm_test_result_all_ldgd',
                                         show_errorbars=True)

    """
    inducing_points = (history_test['z_list_reg'][-1], history_test['z_list_cls'][-1])

    plot_heatmap(X, labels_train, model, alpha_cls, cmap='winter', range_scale=1.2,
                 file_name='latent_heatmap_train', inducing_points=inducing_points, save_path=save_path[0])
    plot_heatmap(X_test, labels_test, model, alpha_cls, cmap='winter', range_scale=1.2,
                 file_name='latent_heatmap_test', inducing_points=inducing_points, save_path=save_path[0])
    """

    return metrics, report_results

