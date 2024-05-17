import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import linregress
import statsmodels.api as sm


def _get_feature_groups(features):
    """
    Categorize feature names into different groups based on their prefixes.

    Parameters
    ----------
    features : dict
        A dictionary of features where keys are feature names.

    Returns
    -------
    dict
        A dictionary categorizing feature names into frequency, coherence, coherence vector,
        and time feature groups.
    """
    freq_features, coh_features, coh_features_vec, time_features = [], [], [], []
    for feature_name, feature_data in features.items():
        if 'freq_' in feature_name:
            freq_features.append(feature_name)
        if 'coh_tot_' in feature_name and '_freqs' not in feature_name:
            coh_features.append(feature_name)
        if 'coh_vec_' in feature_name and '_freqs' not in feature_name:
            coh_features_vec.append(feature_name)
        if 'time_' in feature_name:
            time_features.append(feature_name)
    return {'freq_features': freq_features,
            'coh_features': coh_features,
            'coh_features_vec': coh_features_vec,
            'time_features': time_features}


def plot_linear_fit(feature_data, continuous_label, patient_id, feature_name='Feature', label_type='Reaction Time',
                    display=False, figsize=None, rotation=0, save_path='', reverse_axis=False):
    """
    Plot a linear fit between features and a continuous label.

    Parameters
    ----------
    feature_data : pandas.DataFrame
        DataFrame containing the feature data.
    continuous_label : numpy.ndarray
        Array of continuous labels used for correlation analysis.
    patient_id : str
        Identifier for the patient.
    feature_name : str, optional
        Name of the feature group (default is 'Feature').
    label_type : str, optional
        Type of label for the plot (default is 'Reaction Time').
    display : bool, optional
        Whether to display the plot (default is False).
    figsize : tuple, optional
        Size of the figure (default is None).
    rotation : int, optional
        Rotation angle for x-axis labels (default is 0).
    save_path : str, optional
        Path to save the plot (default is '').
    reverse_axis : bool, optional
        Whether to reverse the axis in the plot (default is False).
    """
    # Prepare DataFrame for plotting
    feature_data = feature_data / feature_data.quantile(0.99, axis=0)
    feature_data[label_type] = continuous_label

    num_features = len(feature_data.columns) - 1
    num_columns = min(num_features, 4)
    num_rows = (num_features + 3) // 4  # +7 to round up the division

    if figsize is None:
        figsize = (6 * num_columns, 4 * num_rows)
    fig, axs = plt.subplots(num_rows, num_columns, dpi=300, figsize=figsize)

    # Ensure axs is a 2D array
    axs = axs.reshape(num_rows, num_columns)

    for idx, feature in enumerate(feature_data.columns[:-1]):
        row, col = divmod(idx, num_columns)

        # Plot
        if reverse_axis is True:
            # Linear regression
            slope, intercept, r_value, _, _ = linregress(continuous_label, feature_data[feature])
            ax = axs[row, col]
            ax.scatter(continuous_label, feature_data[feature], alpha=0.7, label='Data points')
            ax.plot(continuous_label, intercept + slope * continuous_label, color='red',
                    label=f'Fitted line ($R^2$ = {r_value ** 2:.2f})')
            ax.set_title(f'{feature_name} - {feature}', fontsize=10, fontweight='bold', color='dimgray')
            ax.set_xlabel(label_type)
            ax.set_ylabel(feature)
        else:
            # Linear regression
            slope, intercept, r_value, _, _ = linregress(feature_data[feature], continuous_label)
            ax = axs[row, col]
            ax.scatter(feature_data[feature], continuous_label, alpha=0.7, label='Data points')
            ax.plot(feature_data[feature], intercept + slope * feature_data[feature], color='red',
                    label=f'Fitted line ($R^2$ = {r_value ** 2:.2f})')
            ax.set_title(f'{feature_name} - {feature}', fontsize=10, fontweight='bold', color='dimgray')
            ax.set_xlabel(feature)
            ax.set_ylabel(label_type)
        ax.legend()
        ax.grid(True)

    plt.suptitle(f'Linear Relationship with {label_type} - Patient {patient_id}', fontsize=12, fontweight='bold')
    plt.tight_layout()

    if rotation > 30:
        plt.subplots_adjust(bottom=0.2)

    if save_path:
        fig.savefig(f'{save_path}{feature_name}_{label_type}_relationship_{patient_id}.png')
        fig.savefig(f'{save_path}{feature_name}_{label_type}_relationship_{patient_id}.svg')

    if display:
        plt.show()

    plt.close(fig)


def plot_feature_discriminancy(feature_data, label, patient_id, feature_name='Frequency',
                               label_type='decision', display='False', figsize=None, rotation=0, save_path=''):
    """
    Plot feature discriminancy using boxplot visualization.

    Parameters
    ----------
    feature_data : pandas.DataFrame
        DataFrame containing the feature data.
    label : numpy.ndarray
        Array of labels used for discriminancy analysis.
    patient_id : str
        Identifier for the patient.
    feature_name : str, optional
        Name of the feature group (default is 'Frequency').
    label_type : str, optional
        Type of label for the plot (default is 'decision').
    display : bool, optional
        Whether to display the plot (default is False).
    figsize : tuple, optional
        Size of the figure (default is None).
    rotation : int, optional
        Rotation angle for x-axis labels (default is 0).
    save_path : str, optional
        Path to save the plot (default is '').
    """
    # Boxplot visualization and p-value calculation
    feature_list = feature_data.columns.to_list()
    feature_data = feature_data / feature_data.max(axis=0)
    feature_data['decision'] = label

    if figsize is None:
        figsize = (len(feature_list) * 1 + 4, 5)

    unique_labels = np.unique(label)
    if label_type == 'decision':
        label_mapping = {-1: 'Wrong Decision', 1: 'Correct Decision'}
        feature_data['decision'] = feature_data['decision'].map(label_mapping)
    elif label_type == 'rec_old':
        label_mapping = {unique_labels[0]: 'Old', unique_labels[1]: 'New'}
        feature_data['decision'] = feature_data['decision'].map(label_mapping)

    fig, ax = plt.subplots(1, 1, dpi=300, figsize=figsize, gridspec_kw={'wspace': .1, 'hspace': 0.8})

    long_format = feature_data.melt(id_vars='decision',
                                    value_vars=feature_list,
                                    var_name='Feature', value_name='Value')

    sns.boxplot(x='Feature', y='Value', hue='decision',
                data=long_format, palette='Set2', ax=ax)

    # Add strip plot
    sns.stripplot(x='Feature', y='Value', hue='decision', data=long_format,
                  jitter=True, dodge=True, linewidth=1, palette='Set2', ax=ax)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', labelsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha="center")
    ax.set_xlabel('Feature')
    ax.set_ylabel('Feature Value')
    ax.tick_params(axis='x', pad=0)
    ax.set_title(f'Feature Discriminancy for {label_type} - Patient {patient_id.partition("_")[0]}',
                 fontsize=10, fontweight='bold', color='dimgray')
    if rotation > 30:
        plt.subplots_adjust(bottom=0.2)
    if display is True:
        plt.show()
    fig.savefig(os.path.join(save_path, f'{feature_name}_{label_type}_discriminancy_{patient_id}.png'))
    fig.savefig(os.path.join(save_path, f'{feature_name}_{label_type}_discriminancy_{patient_id}.svg'))

    plt.close(fig)
    plt.clf()
    plt.cla()


def calculate_correlations(df, label):
    """
    Calculate Pearson correlation coefficients for each column in the DataFrame with the label.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing features.
    label : pandas.Series
        Continuous variable for correlation.

    Returns
    -------
    dict
        Dictionary of correlation coefficients for each feature.
    """
    correlations = {}
    for feature in df.columns:
        corr, _ = pearsonr(df[feature], label)
        correlations[feature] = corr
    return correlations


def calculate_p_values(feature_data, label):
    """
    Calculate p-values for the statistical significance of differences in feature values between groups.

    Parameters
    ----------
    feature_data : pandas.DataFrame
        DataFrame containing the features.
    label : numpy.ndarray
        Array of labels used for calculating p-values.

    Returns
    -------
    dict
        Dictionary containing p-values for each feature.
    """
    p_values = {}
    feature_data['label'] = label
    decision_values = feature_data['label'].unique()

    # if len(decision_values) != 2:
    #     print(f"Warning: Label column has more than 2 label {decision_values}")

    for feature in feature_data.columns.drop('label'):
        group1 = feature_data[feature_data['label'] == decision_values[0]][feature]
        group2 = feature_data[feature_data['label'] == decision_values[1]][feature]

        # Perform t-test
        stat, p_value = stats.ttest_ind(group1, group2)
        p_values[feature] = p_value

    return p_values


def calculate_line_fit(df, label_continuous):
    p_value_dict, slope_dict, bias_dict = {}, {}, {}

    # Ensure the label_continuous is not included in the features
    features = [feature for feature in df.columns if feature != 'label_continuous']
    df['label_continuous'] = label_continuous

    for feature in features:
        # Prepare the data for OLS
        y = df[feature]
        X = df['label_continuous']

        # Adding a constant to the model (for bias)
        X = sm.add_constant(X)

        # Fit the OLS model
        model = sm.OLS(y, X).fit()

        # Extracting the bias (intercept), slope (coefficient), and p-value for the slope
        bias = model.params['const']
        slope = model.params['label_continuous']
        p_value = model.pvalues['label_continuous']

        # Storing the results
        # correlations[feature] = {'bias': bias, 'slope': slope, 'p-value': p_value}
        p_value_dict[feature] = p_value
        slope_dict[feature] = slope
        bias_dict[feature] = bias



    return slope_dict, p_value_dict, bias_dict
