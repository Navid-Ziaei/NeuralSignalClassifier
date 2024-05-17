import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from mne import create_info
from mne.viz import plot_topomap, plot_sensors, plot_brain_colorbar
import mne


def plot_feature_coherency(df_p_value, channel_counts, save_path, p_value_thresh, target):
    """

    Args:
        df_p_value:
        channel_counts:
        save_path:
        p_value_thresh:

    Returns:

    """
    # Plotting the histograms
    for feature in df_p_value.columns[1:]:
        plt.figure(figsize=(20, 8), dpi=150)
        channels = list(channel_counts.keys())
        n_channels = len(channels)
        bar_width = 0.25
        indices = np.arange(n_channels) - bar_width

        # Data for plotting
        counts_p_gt_005 = np.array([channel_counts[ch][feature][0] for ch in channels], dtype=float)
        counts_p_gt_005_slope_lt_0 = np.array([channel_counts[ch][feature][1] for ch in channels], dtype=float)
        counts_p_gt_005_slope_gt_0 = np.array([channel_counts[ch][feature][2] for ch in channels], dtype=float)

        # Calculate total counts and percentages
        total_counts = counts_p_gt_005 + counts_p_gt_005_slope_lt_0 + counts_p_gt_005_slope_gt_0
        percents_p_gt_005 = np.divide(counts_p_gt_005, total_counts, out=np.zeros_like(counts_p_gt_005, dtype=float),
                                      where=total_counts != 0) * 100
        percents_p_gt_005_slope_lt_0 = np.divide(counts_p_gt_005_slope_lt_0, total_counts,
                                                 out=np.zeros_like(counts_p_gt_005_slope_lt_0, dtype=float),
                                                 where=total_counts != 0) * 100
        percents_p_gt_005_slope_gt_0 = np.divide(counts_p_gt_005_slope_gt_0, total_counts,
                                                 out=np.zeros_like(counts_p_gt_005_slope_gt_0, dtype=float),
                                                 where=total_counts != 0) * 100

        # Plotting each group of bars
        plt.bar(indices, percents_p_gt_005, bar_width, label=f'p > {p_value_thresh}',
                color='lightblue')
        plt.bar(indices + bar_width, percents_p_gt_005_slope_lt_0, bar_width,
                label=f'p < {p_value_thresh} and slope < 0',
                color='green')
        plt.bar(indices + 2 * bar_width, percents_p_gt_005_slope_gt_0,
                bar_width, label=f'p < {p_value_thresh} and slope > 0',
                color='red')

        plt.xticks(indices + bar_width, [ch.split('_')[-1]+'Hz' for ch in channels],
                                rotation=0, ha='center', fontsize=28)
        plt.xlabel('Frequency', fontsize=28)
        plt.ylabel('Percentage (%)', fontsize=28)
        plt.title(f'Percentage Distribution for total Coherency - {target}', fontsize=32)
        plt.yticks(fontsize=28)
        plt.legend(fontsize=28)

        # Adjusting x-axis limits
        plt.xlim(min(indices) - 1.5 * bar_width, max(indices) + 3.5 * bar_width)

        # Removing top and right spines and disabling grid
        sns.despine()
        plt.grid(False)

        plt.tight_layout()
        plt.savefig(save_path + f"Coherency_slope_p_value_{target}.png")

        plt.close()
        plt.clf()
        plt.cla()


def plot_feature_coherency_vector(df_p_value, channel_counts, save_path, p_value_thresh, target):
    channels = list(channel_counts.keys())
    for feature in df_p_value.columns[1:]:
        # Define the frequencies and groups
        frequencies = [5.0, 8.0, 13.0, 30.0]
        groups = ['Group0', 'Group1', 'Group2', 'Group3', 'Group4', 'Group5', 'Group6', 'Group7']
        bar_width = 0.25
        # Create subplots for each frequency
        fig, axes = plt.subplots(len(frequencies), 1, figsize=(20, 15), dpi=150, sharex=True)

        for i, freq in enumerate(frequencies):
            # Filter channels for the current frequency
            channels_for_freq = [ch for ch in channels if f'coh_vec_coh_{freq}' in ch]
            n_channelsfor_freq = len(channels_for_freq)
            indices = np.arange(n_channelsfor_freq) - bar_width

            # Data for plotting
            counts_p_gt_005 = np.array([channel_counts[ch][feature][0] for ch in channels_for_freq], dtype=float)
            counts_p_gt_005_slope_lt_0 = np.array([channel_counts[ch][feature][1] for ch in channels_for_freq], dtype=float)
            counts_p_gt_005_slope_gt_0 = np.array([channel_counts[ch][feature][2] for ch in channels_for_freq], dtype=float)

            # Calculate total counts and percentages
            total_counts = counts_p_gt_005 + counts_p_gt_005_slope_lt_0 + counts_p_gt_005_slope_gt_0
            percents_p_gt_005 = np.divide(counts_p_gt_005, total_counts, out=np.zeros_like(counts_p_gt_005, dtype=float),
                                          where=total_counts != 0) * 100
            percents_p_gt_005_slope_lt_0 = np.divide(counts_p_gt_005_slope_lt_0, total_counts,
                                                     out=np.zeros_like(counts_p_gt_005_slope_lt_0, dtype=float),
                                                     where=total_counts != 0) * 100
            percents_p_gt_005_slope_gt_0 = np.divide(counts_p_gt_005_slope_gt_0, total_counts,
                                                     out=np.zeros_like(counts_p_gt_005_slope_gt_0, dtype=float),
                                                     where=total_counts != 0) * 100

            # Plotting the channels for the current frequency

            axes[i].bar(indices, percents_p_gt_005, bar_width, label=f'p > {p_value_thresh}',
                        color='lightblue')
            axes[i].bar(indices + bar_width, percents_p_gt_005_slope_lt_0, bar_width,
                        label=f'p < {p_value_thresh} and slope < 0',
                        color='green')
            axes[i].bar(indices + 2 * bar_width, percents_p_gt_005_slope_gt_0,
                        bar_width, label=f'p < {p_value_thresh} and slope > 0',
                        color='red')

            axes[i].set_xticks(indices + bar_width)
            axes[i].set_xticklabels([ch.split('_')[4] for ch in channels_for_freq],
                                    rotation=0, ha='center', fontsize=28)
            axes[i].tick_params(axis='y', labelsize=28)
            axes[i].legend()

            axes[i].set_title(f'Coherency Vector at {freq} - {target}', fontsize=36)
            axes[i].set_xlabel('Groups')
            axes[i].set_ylabel('Percentage (%)', fontsize=28)  # Add a single ylabel for all subplots

        # Add common xlabel and adjust layout
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4)  # Adjust vertical spacing between subplots

        # Show the plot
        fig.savefig(save_path + f"Cohernecy_Vector_slope_p_value_{target}.png")
        plt.close()
        plt.clf()
        plt.cla()


def plot_feature_histograms(df_columns, channel_counts, save_path, feature, p_value_thresh, file_name=''):
    sns.set(style='ticks', context='talk', font_scale=1.1, rc={
        'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black',
        'axes.titlesize': 32, 'axes.labelsize': 48})

    for col in df_columns:
        plt.figure(figsize=(60, 10), dpi=150)
        channels = list(channel_counts[col].keys())
        n_channels = len(channels)
        bar_width = 0.25
        indices = np.arange(n_channels) - bar_width / 2

        counts = [channel_counts[col][ch] for ch in channels]

        # Plotting the bar
        plt.bar(indices, counts, bar_width, color='lightblue')

        # Labeling
        plt.xlabel('Channel')
        plt.ylabel(f'Count of Values < {p_value_thresh}')
        plt.title(f'Count of Values < {p_value_thresh} for {feature} - {col} - {file_name}')
        plt.xticks(indices, channels, rotation=90)
        plt.yticks()

        # Removing top and right spines and disabling grid
        sns.despine()
        plt.grid(False)

        plt.tight_layout()

        plt.savefig(save_path + f"{file_name}_{feature}_{col}.png")

        plt.close()
        plt.clf()
        plt.cla()


def analyze_p_value(feature_list, path_results, p_value_thresh, save_path, target='rec_old'):
    folder_list = os.listdir(path_results)

    for feature in feature_list:
        column_name = 'Unnamed: 0' if feature in ['freq_features', 'time_features'] else 'feature'

        for folder in folder_list:
            path_data = os.path.join(path_results, folder, f"p_value_{target}_{feature}.csv")
            try:
                df = pd.read_csv(path_data)
                if column_name != 'Unnamed: 0':
                    df = df.drop('Unnamed: 0', axis=1)

                # Initialize a dictionary to count values for each column
                channel_counts = {col: {} for col in df.columns[1:]}  # Skip the 'Unnamed: 0' column
                patient_list = {col: {} for col in df.columns[1:]}

                # Iterate through each row in the DataFrame
                for _, row in df.iterrows():
                    channel = row[column_name]

                    # Check and count values for each column
                    for col in df.columns[1:]:  # Skip the 'Unnamed: 0' column
                        if channel not in channel_counts[col]:
                            channel_counts[col][channel] = 0
                            patient_list[col][channel] = []
                        if np.abs(row[col]) < p_value_thresh:
                            channel_counts[col][channel] += 1
                            patient_list[col][channel].append(folder)
            except FileNotFoundError:
                print(f"File not found: {path_data}")
                continue
            except pd.errors.EmptyDataError:
                print(f"Empty CSV file: {path_data}")
                continue

        # Plotting the histogram for each column of the feature
        plot_feature_histograms(df.columns[1:], channel_counts, save_path, feature, p_value_thresh,
                                file_name=target)
        with open(save_path + feature + f"_best_patient_{feature}_{target}_p_value.json", 'w') as f:
            json.dump(patient_list, f, indent=3)


def analyze_line_fit_p_value(feature_list, path_results, p_value_thresh, save_path, target='decision'):
    folder_list = os.listdir(path_results)
    # Initialize a dictionary to count values based on conditions for each channel
    for feature in feature_list:
        channel_counts = {}
        patient_list = {}
        column_name = 'Unnamed: 0' if feature in ['freq_features', 'time_features'] else 'feature'

        for folder in folder_list:
            try:
                path_data_p_value = os.path.join(path_results, folder, f"line_fit_pvalue_{target}_{feature}.csv")
                path_data_slope = os.path.join(path_results, folder, f"line_fit_slope_{target}_{feature}.csv")

                df_p_value = pd.read_csv(path_data_p_value)
                df_slope = pd.read_csv(path_data_slope)

                if column_name != 'Unnamed: 0':
                    df_p_value = df_p_value.drop('Unnamed: 0', axis=1)
                    df_slope = df_slope.drop('Unnamed: 0', axis=1)

                # Ensure channel names match in both dataframes
                if not all(df_p_value[column_name] == df_slope[column_name]):
                    print(f"Channel mismatch in folder {folder}")
                    continue

                # Iterate through each row in the DataFrame
                for idx, row_p in df_p_value.iterrows():
                    channel = row_p[column_name]
                    # Initialize the count for this channel if not already present
                    if channel not in channel_counts:
                        channel_counts[channel] = {col: [0, 0, 0] for col in df_p_value.columns[1:]}
                        patient_list[channel] = {col: [[], []] for col in df_p_value.columns[1:]}

                    row_s = df_slope.iloc[idx]

                    # Check conditions and count values for each feature
                    for col_p, col_s in zip(df_p_value.columns[1:],
                                            df_slope.columns[1:]):  # Skip the 'Unnamed: 0' column
                        p_val = row_p[col_p]
                        slope_val = row_s[col_s]

                        if p_val < p_value_thresh:
                            if slope_val < 0:
                                channel_counts[channel][col_p][1] += 1  # Count for condition 2
                                patient_list[channel][col_p][0].append(folder)
                            elif slope_val > 0:
                                channel_counts[channel][col_p][2] += 1  # Count for condition 3
                                patient_list[channel][col_p][1].append(folder)
                        else:
                            channel_counts[channel][col_p][0] += 1  # Count for condition 1
            except:
                print(folder)

        if feature == 'freq_features':
            df = pd.DataFrame.from_dict(channel_counts).transpose()
            df['channel_name'] = df.index
            plot_the_brain_heatmap_spectral(df_p_value, column_name, save_path, target, file_name='p_value_freqs',
                                            time_label='250 to 750', p_value_thresh=p_value_thresh)
            plot_the_brain_heatmap_spectral(df_p_value, column_name, save_path, target, file_name='p_value_freqs',
                                            time_label='0 to 500', p_value_thresh=p_value_thresh)

            plot_the_brain_heatmap_spectral(df, column_name='channel_name',
                                            save_path=save_path, target=target, file_name='sig_hist_freqs_positive',
                                            time_label='0 to 500', plot_positive_beta=True, p_value_thresh=p_value_thresh)

            plot_the_brain_heatmap_spectral(df, column_name='channel_name',
                                            save_path=save_path, target=target, file_name='sig_hist_freqs_positive',
                                            time_label='250 to 750', plot_positive_beta=True, p_value_thresh=p_value_thresh)

            plot_the_brain_heatmap_spectral(df, column_name='channel_name',
                                            save_path=save_path, target=target, file_name='sig_hist_freqs_negative',
                                            time_label='0 to 500', plot_positive_beta=False, p_value_thresh=p_value_thresh)

            plot_the_brain_heatmap_spectral(df, column_name='channel_name',
                                            save_path=save_path, target=target, file_name='sig_hist_freqs_negative',
                                            time_label='250 to 750', plot_positive_beta=False, p_value_thresh=p_value_thresh)
        elif feature == 'time_features':
            df = pd.DataFrame.from_dict(channel_counts).transpose()
            df['channel_name'] = df.index

            plot_the_brain_heatmap_time(df, channel_column_name='channel_name',
                                        save_path=save_path, target=target, file_name='sig_hist_times',
                                        plot_positive_beta=True, p_value_thresh=p_value_thresh)

            plot_the_brain_heatmap_time(df, channel_column_name='channel_name',
                                        save_path=save_path, target=target, file_name='sig_hist_time',
                                        plot_positive_beta=False, p_value_thresh=p_value_thresh)
        elif feature == 'coh_features_vec':
            plot_feature_coherency_vector(df_p_value, channel_counts, save_path, p_value_thresh, target=target)
        else:
            plot_feature_coherency(df_p_value, channel_counts, save_path, p_value_thresh, target=target)

        # Load your EEG data from the ELP file


def plot_the_brain_heatmap_spectral(df, column_name, save_path, target, file_name='p_value_freqs',
                                    time_label='250 to 750', plot_positive_beta=True, p_value_thresh=0.05):
    # create array with 4 points for our 4 channels
    # in the same order as provided in ch_names
    info = create_info(ch_names=df[column_name].to_list(), sfreq=1000, ch_types='eeg')
    # channel names I provided are part of a standard montage
    info.set_montage('biosemi128')
    feature_title = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

    cmap = 'viridis'  # Change to your desired colormap

    data_list = []
    for i, feature_name in enumerate(feature_title):
        data = df[f'freq_{feature_name.lower()}_time {time_label}'].values
        if isinstance(data[0], list):
            data_p = [d[1] / np.sum(d) for d in list(data)]
            data_n = [d[2] / np.sum(d) for d in list(data)]
            data = data_p if plot_positive_beta is True else data_n
        data_list.append(np.max(data))

    if np.max(data_list) > 0.7:
        max_val = 1
    else:
        max_val = np.max(data_list)

    fig, ax = plt.subplots(1, 5, figsize=(50, 10), dpi=100)
    # Plot each topomap and set titles
    for i, feature_name in enumerate(feature_title):
        data = df[f'freq_{feature_name.lower()}_time {time_label}'].values
        if isinstance(data[0], list):
            data_p = [d[1] / np.sum(d) for d in list(data)]
            data_n = [d[2] / np.sum(d) for d in list(data)]
            data = data_p if plot_positive_beta is True else data_n
            cmap = 'Greens' if plot_positive_beta is True else 'Reds'

        plot_topomap(data,
                     info, res=100, size=40,
                     cmap=cmap, axes=ax[i], show=False, vlim=[0, max_val])
        ax[i].set_title(feature_name, fontsize=48, pad=0)

    # Add a super title for the whole figure
    plt.suptitle(f"EEG Spectral Features (Time window from {time_label} ms)", fontsize=48, y=1.00)

    # Add a colorbar

    # cbar = plt.colorbar(ax[-1].images[0], ax=ax, format='%0.2f')

    plt.subplots_adjust(right=0.85)  # Adjust the value as needed

    # Add a colorbar with larger font size
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])  # Adjust the position and size as needed
    cbar = plt.colorbar(ax[4].images[0], cax=cbar_ax, format='%0.2f')
    cbar.ax.tick_params(labelsize=52)

    fig.savefig(save_path + file_name + f'_{target}_{time_label}.png')


def plot_the_brain_heatmap_time(df, channel_column_name, save_path, target, file_name='p_value_time',
                                plot_positive_beta=True, p_value_thresh=0.05):
    # create array with 4 points for our 4 channels
    # in the same order as provided in ch_names
    info = create_info(ch_names=df[channel_column_name].to_list(), sfreq=1000, ch_types='eeg')
    # channel names I provided are part of a standard montage
    info.set_montage('biosemi128')
    feature_title = ['N200', 'P300', 'Late Positive Components']
    feature_columns = ['time_n200', 'time_p300', 'time_post_p300']

    cmap = 'viridis'  # Change to your desired colormap

    data_list = []
    for i, (feature_name, column_name) in enumerate(zip(feature_title, feature_columns)):
        data = df[column_name].values
        if isinstance(data[0], list):
            data_p = [d[1] / np.sum(d) for d in list(data)]
            data_n = [d[2] / np.sum(d) for d in list(data)]
            data = data_p if plot_positive_beta is True else data_n
        data_list.append(np.max(data))

    if np.max(data_list) > 0.7:
        max_val = 1
    else:
        max_val = np.max(data_list)

    fig, ax = plt.subplots(1, 3, figsize=(50, 10), dpi=100)

    # Plot each topomap and set titles
    for i, (feature_name, column_name) in enumerate(zip(feature_title, feature_columns)):
        data = df[column_name].values
        if isinstance(data[0], list):
            data_p = [d[1] / np.sum(d) for d in list(data)]
            data_n = [d[2] / np.sum(d) for d in list(data)]
            data = data_p if plot_positive_beta is True else data_n
            cmap = 'Greens' if plot_positive_beta is True else 'Reds'

        plot_topomap(data,
                     info, res=100, size=40,
                     cmap=cmap, axes=ax[i], show=False, vlim=[0, max_val])
        ax[i].set_title(feature_name, fontsize=48, pad=0)

    # Add a super title for the whole figure
    plt.suptitle(f"EEG Time Features", fontsize=48, y=1.00)

    # Add a colorbar

    # cbar = plt.colorbar(ax[-1].images[0], ax=ax, format='%0.2f')

    plt.subplots_adjust(right=1.85, hspace=5)  # Adjust the value as needed

    # Add a colorbar with larger font size
    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])  # Adjust the position and size as needed
    cbar = plt.colorbar(ax[-1].images[0], cax=cbar_ax, format='%0.2f')
    cbar.ax.tick_params(labelsize=52)

    # Increase the vertical space between subplots
    plt.tight_layout()

    fig.savefig(save_path + file_name + f'_{target}.png')
