import os
import tarfile
import urllib
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
import scipy
from scipy.stats import bernoulli
from sklearn.datasets import load_iris, load_wine

from sklearn.model_selection import train_test_split


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_synthetic_data(N):
    # Generate synthetic data
    x = np.sort(np.random.uniform(-5, 5, N))  # Sampled input points
    y_true = np.sin(x)
    noise = np.random.normal(0, 0.2, len(x))
    y = y_true + noise

    x_test = torch.linspace(-7, 7, 300).view(-1, 1)
    y_test = np.sin(x_test)

    # Convert data to PyTorch tensors
    x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    # Plot the synthetic data
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_true, 'r', label='True Function')
    plt.scatter(x, y, c='b', s=20, label='Noisy Observations')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Synthetic Regression Data')
    plt.grid(True)
    plt.show()

    return x_tensor, y_tensor, x_test, y_test


def generate_synthetic_classification_data(N):
    X = np.linspace(0, 5, N).reshape(-1, 1)
    X_test = np.linspace(-0.5, 5.5, 100).reshape(-1, 1)

    a = np.sin(X * np.pi * 0.5) * 2
    Y = bernoulli.rvs(sigmoid(a))
    y_test = bernoulli.rvs(sigmoid(np.sin(X_test * np.pi * 0.5) * 2))

    plot_data_1D(X, Y)
    plt.title('1D training dataset')
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.yticks([0, 1])
    plt.legend()

    return torch.Tensor(X), torch.Tensor(Y), torch.Tensor(X_test), torch.Tensor(y_test)


def plot_with_confidence(x_train, y_train, x_test, mu_star, std_test, inducing_points=None):
    # Convert to numpy for plotting after detaching from the computation graph
    # Compute 95% confidence interval
    lower_bound = mu_star - 1.96 * std_test
    upper_bound = mu_star + 1.96 * std_test

    x_test_np = x_test.numpy().squeeze()
    mu_star_np = mu_star.detach().numpy().squeeze()
    lower_bound_np = lower_bound.detach().numpy().squeeze()
    upper_bound_np = upper_bound.detach().numpy().squeeze()

    # Plot the predictions with the confidence interval
    plt.figure(figsize=(10, 6))
    plt.fill_between(x_test_np, lower_bound_np, upper_bound_np, color='C0', alpha=0.3, label='95% Confidence Interval')
    plt.plot(x_test_np, mu_star_np, 'C0', lw=2, label='Predictive Mean')
    plt.scatter(x_train, y_train, c='r', s=20, zorder=10, edgecolors=(0, 0, 0), label='Noisy Observations')

    # Plotting the inducing points if provided
    if inducing_points is not None:
        inducing_points_np = inducing_points.squeeze()
        plt.scatter(inducing_points_np, np.zeros_like(inducing_points_np),
                    c='g', s=100, zorder=5, marker='X', label='Inducing Points')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gaussian Process Regression Predictions with Inducing Points')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_loss(losses):
    # Plot the optimization process
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Negative Log-Marginal Likelihood')
    plt.title('Optimization of Negative Log-Marginal Likelihood')
    plt.grid(True)
    plt.show()


def create_animation(x_train, y_train, inducing_points_history):
    # Setup the figure for animation
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_train, torch.sin(x_train), 'r', label='True Function')
    ax.scatter(x_train, y_train, c='b', s=20, label='Noisy Observations')
    line, = ax.plot([], [], 'go', label='Inducing Points')
    ax.legend()
    ax.grid(True)
    ax.set_title('Movement of Inducing Points during Training')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Initialization function for the animation
    def init():
        line.set_data([], [])
        return line,

    # Update function for the animation
    def update(epoch):
        line.set_data(inducing_points_history[epoch].numpy(), np.zeros_like(inducing_points_history[epoch].numpy()))
        return line,

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(inducing_points_history), init_func=init, blit=True,
                                  repeat=False)

    plt.close(fig)

    # Save animation as a gif
    gif_path = "inducing_points_animation.gif"
    ani.save(gif_path, writer='pillow', fps=10)


def plot_data_1D(X, t):
    class_0 = t == 0
    class_1 = t == 1

    plt.scatter(X[class_1], t[class_1], label='Class 1', marker='x', color='red')
    plt.scatter(X[class_0], t[class_0], label='Class 0', marker='o', edgecolors='blue', facecolors='none')


def plot_data_2D(X, y):
    color = ['red', 'green', 'blue', 'yellow', 'black', 'pink']
    for t in np.unique(y):
        class_t = list(np.ravel(y == t))
        plt.scatter(X[class_t, 0], X[class_t, 1], label=f'Class {t}', marker='x', c=color[int(t)])

    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

    plt.show()


def plot_pt_2D(grid_x, grid_y, grid_z):
    plt.contourf(grid_x, grid_y, grid_z, cmap='plasma', alpha=0.3, levels=np.linspace(0, 1, 11))
    plt.colorbar(format='%.2f')


def plot_db_2D(grid_x, grid_y, grid_z, decision_boundary=0.5):
    levels = [decision_boundary]
    cs = plt.contour(grid_x, grid_y, grid_z, levels=levels, colors='black', linestyles='dashed', linewidths=2)
    plt.clabel(cs, fontsize=20)


def load_oil_data(test_size=0.1):
    # If you are running this notebook interactively
    wdir = Path(os.path.abspath('')).parent.parent
    os.chdir(wdir)

    url = "http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/resources/3PhData.tar.gz"
    urllib.request.urlretrieve(url, '3PhData.tar.gz')
    with tarfile.open('3PhData.tar.gz', 'r') as f:
        f.extract('DataTrn.txt')
        f.extract('DataTrnLbls.txt')
    y = np.loadtxt(fname='DataTrn.txt')
    s = np.loadtxt(fname='DataTrnLbls.txt')
    yn_train, yn_test, ys_train, ys_test = train_test_split(y, s, test_size=test_size, random_state=42)

    labels_train = (torch.tensor(ys_train) @ np.diag([0, 1, 2])).sum(axis=1)
    labels_test = (torch.tensor(ys_test) @ np.diag([0, 1, 2])).sum(axis=1)

    return torch.tensor(yn_train), torch.tensor(yn_test), torch.tensor(ys_train), torch.tensor(ys_test), torch.tensor(
        labels_train), torch.tensor(labels_test)


def load_iris_data(test_size=0.1):
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # One-hot encode the labels
    y_one_hot = np.zeros((y.size, y.max() + 1))
    y_one_hot[np.arange(y.size), y] = 1

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = train_test_split(X, y_one_hot, y,
                                                                                       test_size=test_size,
                                                                                       random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    y_train_labels_tensor = torch.tensor(y_train_labels, dtype=torch.long)
    y_test_labels_tensor = torch.tensor(y_test_labels, dtype=torch.long)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, y_train_labels_tensor, y_test_labels_tensor


def plot_results_gplvm(X, std, labels, losses, inverse_length_scale, latent_dim,
                       largest=True, save_path=None, file_name='gplvm_result', title='2d latent subspace',
                       show_errorbars=True):
    values, indices = torch.topk(torch.tensor(inverse_length_scale), k=2, largest=largest)

    l1 = indices.numpy().flatten()[0]
    l2 = indices.numpy().flatten()[1]

    colors = ['r', 'b', 'g']

    plt.figure(figsize=(20, 8))
    plt.subplot(131)

    plt.title(title, fontsize='small')
    plt.xlabel('Latent dim 1')
    plt.ylabel('Latent dim 2')

    # Select index of the smallest lengthscales by examining model.covar_module.base_kernel.lengthscales
    for i, label in enumerate(np.unique(labels)):
        X_i = X[labels == label]
        scale_i = std[labels == label]
        plt.scatter(X_i[:, l1], X_i[:, l2], c=[colors[i]], label=label)
        if show_errorbars is True:
            plt.errorbar(X_i[:, l1], X_i[:, l2], xerr=scale_i[:, l1], yerr=scale_i[:, l2], label=label, c=colors[i],
                         fmt='none')
    plt.legend()  # Add legend for the third subplot

    plt.subplot(132)
    plt.bar(np.arange(latent_dim), height=inverse_length_scale.flatten())
    plt.title('Inverse Lengthscale with SE-ARD kernel', fontsize='small')

    plt.subplot(133)
    plt.plot(losses[10:], label='batch_size=100')
    plt.title('Neg. ELBO Loss', fontsize='small')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path + file_name + '.png')
        plt.savefig(save_path + file_name + '.svg')
    else:
        plt.show()
