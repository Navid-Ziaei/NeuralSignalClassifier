import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

AXIS_LABEL_FONTSIZE = 16
TITLE_LABEL_FONTSIZE = 18
TICKS_FONTSIZE = 14


def plot_metric_heatmaps(df, metric='precision', grid_size=(10, 10), save_dir='heatmaps'):
    """
    Plots and saves heatmaps for a specified metric's mean and standard deviation.

    Args:
        df (pd.DataFrame): DataFrame containing the mean and std values for the metrics.
        metric (str): The metric to plot ('precision', 'recall', 'f1').
        grid_size (tuple): The size of the grid (rows, columns).
        save_dir (str): Directory to save the heatmaps.

    Returns:
        None: Displays and saves the heatmap plots.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Initialize empty grids
    mean_grid = np.zeros(grid_size)
    std_grid = np.zeros(grid_size)

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            target = f'target_{i}_{j}'
            mean_grid[i, j] = df['xgboost'].loc[target, f'{metric}_mean']
            std_grid[i, j] = df['xgboost'].loc[target, f'{metric}_std']

    # Custom colormap: red for low, green for high
    cmap = LinearSegmentedColormap.from_list('custom_green_red', ['red', 'yellow', 'green'])

    # Plotting the mean heatmap
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(mean_grid, annot=True, cmap=cmap, fmt='.2f', cbar=True)
    plt.title(f'{metric.capitalize()} Mean')

    cmap = LinearSegmentedColormap.from_list('custom_green_red', ['green', 'yellow', 'red'])
    # Plotting the std heatmap
    plt.subplot(1, 2, 2)
    sns.heatmap(std_grid, annot=True, cmap=cmap, fmt='.2f', cbar=True)
    plt.title(f'{metric.capitalize()} Std')

    plt.tight_layout()

    # Save the heatmap
    save_path = os.path.join(save_dir, f'{metric}_heatmaps.png')
    plt.savefig(save_path, dpi=300)

    # Show the heatmap
    plt.close()

def plot_histogram(values, xlabel, ylabel, title=None):
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(values, bins=30, alpha=0.7, color='#1f77b4', edgecolor='black',
                                linewidth=1.2)

    # Title and labels with appropriate font sizes
    if title is not None:
        plt.title(title, fontsize=TITLE_LABEL_FONTSIZE, fontweight='bold')
    plt.xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)

    # Adjusting tick parameters for readability
    plt.xticks(fontsize=TICKS_FONTSIZE)
    plt.yticks(fontsize=TICKS_FONTSIZE)

    # Enhancing the grid for better readability while keeping it unobtrusive
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Removing top and right spines for a cleaner look
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Tight layout for better spacing
    plt.tight_layout()
    plt.show()