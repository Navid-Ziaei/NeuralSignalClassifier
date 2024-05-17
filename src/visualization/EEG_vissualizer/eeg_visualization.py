import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind, pointbiserialr
from sklearn.linear_model import LogisticRegression

from src.data_loader import EEGDataSet
import seaborn as sns


def visualize_erp(patient_data, channel_idx, label, label_names, fig=None, axs=None, save_plot=False, time_lim=None,
                  title_ext=''):
    if axs is None:
        fig, axs = plt.subplots(1, 1, figsize=(10, 5))

    if time_lim is not None:
        idx_start = np.argmin(np.abs(time_lim[0] - patient_data.time_ms))
        idx_end = np.argmin(np.abs(time_lim[1] - patient_data.time_ms))
        time = patient_data.time_ms[idx_start:idx_end]
        data = patient_data.data[:, :, idx_start:idx_end]
    else:
        time = patient_data.time_ms
        data = patient_data.data

    label_list = np.unique(label)

    # Extract trials based on type
    trials_type1 = data[np.array([l == label_list[0] for l in label])]
    trials_type2 = data[np.array([l == label_list[1] for l in label])]

    # Calculate mean and standard deviation for each type
    mean_data1 = np.mean(trials_type1, axis=0)[channel_idx]
    std_data1 = np.std(trials_type1, axis=0)[channel_idx] / np.sqrt(len(trials_type1))

    mean_data2 = np.mean(trials_type2, axis=0)[channel_idx]
    std_data2 = np.std(trials_type2, axis=0)[channel_idx] / np.sqrt(len(trials_type2))

    # Calculate confidence intervals
    ci1 = 1.96 * std_data1
    ci2 = 1.96 * std_data2

    # Plotting
    axs.plot(time, mean_data1, label=label_names[0])
    axs.fill_between(time, mean_data1 - ci1, mean_data1 + ci1, color='b', alpha=0.2)
    axs.plot(time, mean_data2, label=label_names[1])
    axs.fill_between(time, mean_data2 - ci2, mean_data2 + ci2, color='r', alpha=0.2)

    # Additional plotting details
    axs.set_xlabel("Time (ms)", fontsize=22)
    axs.set_ylabel("ERP", fontsize=22)
    axs.set_title(
        " Patient " + patient_data.file_name.split('_')[0] + "Channel " + patient_data.channel_names[
            channel_idx] + ' ' + title_ext,
        fontsize=26)
    axs.tick_params(axis='x', labelsize=18)
    axs.tick_params(axis='y', labelsize=18)

    axs.legend()

    y_min, y_max = axs.get_ylim()
    idx_onset = np.argmin(np.abs(time))
    axs.vlines(0, y_min, y_max, color='black', linewidth=2, linestyles='--')
    axs.set_ylim([y_min, y_max])

    if save_plot is True:
        plt.tight_layout()
        plt.show()

    return fig, axs


def visualize_block_ERP(single_patient_data, channel_idx, stim='w', axs=None, fig=None, show_plot=False,
                        label_list=["no exp", "exp"]):
    block_idx = np.unique(single_patient_data.labels['block_number'])
    label_df = pd.DataFrame(single_patient_data.labels)
    blocks = {idx: list(np.unique(label_df[label_df['block_number'] == idx]['block_type'])) for idx in block_idx}
    blocks_with_stim = [idx for idx, vals in blocks.items() if any(stim in val for val in vals)]

    data_desc = ''

    if axs is None:
        fig, axs = plt.subplots(1, len(blocks_with_stim) + 1, figsize=(50, 8))
    for i in range(len(blocks_with_stim)):
        block_index = (label_df['block_number'] == blocks_with_stim[i]) & (label_df['is_correct'] == True) & (
                label_df['stim'] != 'ctl')
        single_block_data = EEGDataSet()
        single_block_data.labels = {key: label_df[key][block_index].to_list() for key in label_df.columns.to_list()}
        single_block_data.data = single_patient_data.data[block_index.values]
        single_block_data.fs = single_patient_data.fs
        single_block_data.time_ms = single_patient_data.time_ms
        single_block_data.channel_names = single_patient_data.channel_names
        single_block_data.file_name = single_patient_data.file_name
        visualize_erp(patient_data=single_block_data,
                      channel_idx=channel_idx,
                      label=single_block_data.labels['is_experienced'],
                      label_names=['Not Experienced', 'Experienced'],
                      time_lim=[-1000, 1000],
                      axs=axs[i],
                      title_ext='Block ' + blocks[blocks_with_stim[i]][0])
        data_desc += f" Block{blocks_with_stim[i]}: {blocks[blocks_with_stim[i]][0]}" \
                     f" ({label_list[0]}: {np.sum(np.array([l == 0 for l in single_block_data.labels['is_experienced']]))}" \
                     f" {label_list[1]}: {np.sum(np.array([l == 1 for l in single_block_data.labels['is_experienced']]))})"

    block_index = label_df['block_number'].isin(blocks_with_stim) & \
                  (label_df['is_correct'] == True) & \
                  (label_df['stim'] != 'ctl')
    single_block_data = EEGDataSet()
    single_block_data.labels = {key: label_df[key][block_index].to_list() for key in label_df.columns.to_list()}
    single_block_data.data = single_patient_data.data[block_index.values]
    single_block_data.fs = single_patient_data.fs
    single_block_data.time_ms = single_patient_data.time_ms
    single_block_data.channel_names = single_patient_data.channel_names
    single_block_data.file_name = single_patient_data.file_name
    visualize_erp(patient_data=single_block_data,
                  channel_idx=channel_idx,
                  label=single_block_data.labels['is_experienced'],
                  label_names=['Not Experienced', 'Experienced'],
                  time_lim=[-1000, 1000],
                  axs=axs[i + 1],
                  title_ext='Block ' + str([blocks[b_idx][0] for b_idx in blocks_with_stim]))
    print("Channel " + single_patient_data.channel_names[channel_idx] + data_desc)
    if show_plot is True:
        plt.tight_layout()
        plt.show()

    return fig, axs


def visualize_feature_box_plot(patient_data, channel_idx, feature_df, labels,
                               palette={True: "green", False: "red"}, fig=None, ax=None, title_ext=''):
    feature_list = feature_df.columns.to_list()
    feature_list = [feature_name for feature_name in feature_list if feature_name not in ['is_experienced', 'is_resp']]
    single_channel_feature_df = feature_df.copy()

    single_channel_feature_df.loc[:, 'is_experienced'] = np.array(labels['is_experienced'])
    single_channel_feature_df.loc[:, 'is_resp'] = np.array(labels['is_resp'])
    melted_df = pd.melt(single_channel_feature_df, id_vars=['is_experienced', 'is_resp'],
                        value_vars=feature_list,
                        var_name='Feature', value_name='Value')

    # Plot
    sns.boxplot(x='Feature', y='Value', hue='is_experienced', data=melted_df, dodge=True, ax=ax)

    # Add strip plot with colors based on 'another_label'
    # Example colors for True and False

    sns.stripplot(x='Feature', y='Value', hue='is_experienced',
                  data=melted_df, edgecolor='black',
                  alpha=1, dodge=True, palette=palette, ax=ax)

    # ax.tick_params(axis='x', labelsize=18)
    # ax.tick_params(axis='y', labelsize=18)

    ax.legend(title='Is Experienced', loc='upper right', fontsize=18)
    ax.set_title(
        " Patient " + patient_data.file_name.split('_')[0] + " Channel " + patient_data.channel_names[
            channel_idx] + ' ' + title_ext,
        fontsize=20)

    # Handle legends: one for 'is_experienced' and another for 'another_label'
    # handles, labels = plt.gca().get_legend_handles_labels()
    # Split the handles and labels into two groups
    # is_exp_handles, is_exp_labels = handles[:2], labels[:2]  # Assuming 'is_experienced' has 2 unique values
    # is_response_handles, is_response_labels = handles[2:], labels[2:]  # Adjust numbers based on your data

    # Create two separate legends
    # plt.legend(is_exp_handles, is_exp_labels, title='Is Experienced', loc='upper right')
    # plt.gca().add_artist(plt.legend(is_response_handles, is_response_labels, title='go_nogo', loc='lower right'))
    # plt.gca().add_artist(plt.legend(is_exp_handles, is_exp_labels, title='Is Experienced', loc='lower left'))
    return fig, ax


def visualize_p_value_barplots(single_patient_data, time_features, channel_idx, stim='w', fig=None, ax=None,
                               show_plot=False):
    channel_name = single_patient_data.channel_names[channel_idx]
    block_idx = np.unique(single_patient_data.labels['block_number'])
    label_df = pd.DataFrame(single_patient_data.labels)

    # specify blocks with a stim 'w' or 'i'
    blocks = {idx: list(np.unique(label_df[label_df['block_number'] == idx]['block_type'])) for idx in block_idx}
    blocks_with_stim = [idx for idx, vals in blocks.items() if any(stim in val for val in vals)]

    # select specific channel
    time_feature_df = pd.DataFrame({key: time_features[key][:, channel_idx] for key in time_features.keys()})

    visualize_block_features(single_patient_data, time_features, channel_idx=channel_idx, stim=stim, show_plot=True)
    # Bar plot data preparation
    results = []

    for block in blocks_with_stim:
        block_data = label_df[label_df['block_number'] == block]
        block_data = block_data[block_data['is_correct'] & (block_data['stim'] != 'ctl')]
        single_channel_feature_df = time_feature_df.loc[block_data.index]
        single_channel_feature_df['is_experienced'] = block_data['is_experienced']

        for feature_name, data in single_channel_feature_df.items():
            if feature_name == 'is_experienced':
                continue
            exp = data[block_data['is_experienced']]
            inexp = data[~block_data['is_experienced']]

            t_stat, p_val = ttest_ind(exp.dropna(), inexp.dropna(), equal_var=False)
            correlation, p_value = pointbiserialr(block_data['is_experienced'], data)
            model = LogisticRegression()
            model.fit(data.values[:, None], block_data['is_experienced'].values)

            # Get the coefficient for the feature
            coef = model.coef_[0][0]

            results.append({
                'Block': f'Block {block} {blocks[block]}',
                'Feature': feature_name,
                'P-Value': p_val,
                'T-Statistic': t_stat,
                'coef': coef
            })

    block_data = block_data[block_data['is_correct'] & (block_data['stim'] != 'ctl')]
    single_channel_feature_df = time_feature_df.loc[block_data.index]
    single_channel_feature_df['is_experienced'] = block_data['is_experienced']

    for feature_name, data in single_channel_feature_df.items():
        if feature_name == 'is_experienced':
            continue
        exp = data[block_data['is_experienced']]
        inexp = data[~block_data['is_experienced']]

        t_stat, p_val = ttest_ind(exp.dropna(), inexp.dropna(), equal_var=False)
        correlation, p_value = pointbiserialr(block_data['is_experienced'], data)
        model = LogisticRegression()
        model.fit(data.values[:, None], block_data['is_experienced'].values)

        # Get the coefficient for the feature
        coef = model.coef_[0][0]

        results.append({
            'Block': f'Combined',
            'Feature': feature_name,
            'P-Value': p_val,
            'T-Statistic': t_stat,
            'coef': coef
        })
    results_df = pd.DataFrame(results)

    # Plotting
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    sns.barplot(x='Feature', y='P-Value', hue='Block', data=results_df,
                palette=np.where(results_df['T-Statistic'] > 0, 'green', 'red'), ax=ax)
    # plt.yscale('log')  # Log scale for better visibility if small p-values are present
    ax.set_title(f'P-Values by Block and Feature for {channel_name}', fontsize=32)
    ax.set_ylabel('P-Value (log scale)', fontsize=24)
    ax.set_xlabel('Feature', fontsize=24)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)

    if show_plot is True:
        plt.tight_layout()
        plt.show()


def visualize_block_features(single_patient_data, features, channel_idx=1, stim='w', show_plot=False, fig=None,
                             axs=None):
    block_idx = np.unique(single_patient_data.labels['block_number'])
    label_df = pd.DataFrame(single_patient_data.labels)
    blocks = {idx: list(np.unique(label_df[label_df['block_number'] == idx]['block_type'])) for idx in block_idx}
    blocks_with_stim = [idx for idx, vals in blocks.items() if any(stim in val for val in vals)]
    time_feature_df = pd.DataFrame({key: features[key][:, channel_idx] for key in features.keys()})

    if axs is None:
        fig, axs = plt.subplots(1, len(blocks_with_stim) + 1, figsize=(50, 8))

    for i in range(len(blocks_with_stim)):
        block_index = (label_df['block_number'] == blocks_with_stim[i]) & (label_df['is_correct'] == True) & (
                label_df['stim'] != 'ctl')
        labels = {key: label_df[key][block_index].to_list() for key in label_df.columns.to_list()}
        single_channel_feature_df = time_feature_df[block_index.values].copy()

        single_channel_feature_df.loc[:, 'is_experienced'] = labels['is_experienced']
        single_channel_feature_df.loc[:, 'is_resp'] = labels['is_resp']

        visualize_feature_box_plot(single_patient_data, channel_idx, single_channel_feature_df, labels,
                                   palette={True: "green", False: "red"}, ax=axs[i],
                                   title_ext='Block ' + blocks[blocks_with_stim[i]][0])
    block_index = label_df['block_number'].isin(blocks_with_stim) & \
                  (label_df['is_correct'] == True) & \
                  (label_df['stim'] != 'ctl')
    labels = {key: label_df[key][block_index].to_list() for key in label_df.columns.to_list()}
    single_channel_feature_df = time_feature_df[block_index.values].copy()

    single_channel_feature_df.loc[:, 'is_experienced'] = labels['is_experienced']
    single_channel_feature_df.loc[:, 'is_resp'] = labels['is_resp']

    visualize_feature_box_plot(single_patient_data, channel_idx, single_channel_feature_df, labels,
                               palette={True: "green", False: "red"}, ax=axs[i + 1],
                               title_ext='Block ' + str([blocks[b_idx][0] for b_idx in blocks_with_stim]))

    if show_plot is True:
        plt.tight_layout()
        plt.show()

    return fig, axs
