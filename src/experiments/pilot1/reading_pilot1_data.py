import mne
import pandas as pd
import pyxdf
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import detrend

# Path to your XDF file
directory_path = "C:\\Users\\nziaei\OneDrive - Worcester Polytechnic Institute (wpi.edu)\\eeg_project\\pilot1\\"
file_list = os.listdir(directory_path)
xdf_file_list = list(sorted(file for file in file_list if file.endswith(".xdf")))
csv_marker_file_list = list(sorted(file for file in file_list if file.endswith("markers.csv")))
csv_bad_channels_file_list = [file for file in file_list if file.endswith("bad_channels.csv")]

for subject_idx in range(1, len(xdf_file_list)):
    path_to_xdf = xdf_file_list[subject_idx]
    path_to_marker = csv_marker_file_list[subject_idx]
    path_to_bad_channels = csv_bad_channels_file_list[subject_idx]
    # Load the XDF file
    streams, fileheader = pyxdf.load_xdf(directory_path + path_to_xdf)

    for stream_idx in range(len(streams)):
        channel_info = streams[stream_idx]['info']['desc'][0]['channels']
        if len(channel_info) > 0 and len(channel_info[0]['channel']) > 100:
            # Assuming you have one EEG stream, and it's the first stream (index 0)
            eeg_data = streams[stream_idx]['time_series'].T  # Transpose to have channels as rows
            eeg_times = streams[stream_idx]['time_stamps']
            channel_info = streams[stream_idx]['info']['desc'][0]['channels']
            channel_names = [channel['label'][0] for channel in channel_info[0]['channel']]
            subject = streams[stream_idx]['info']['desc'][0]['subject']
            stream_id = streams[stream_idx]['info']['stream_id']
            s_rate = float(streams[stream_idx]['info']['nominal_srate'][0])
            subject_id = streams[stream_idx]['info']['desc'][0]['subject'][0]['id'][0]
            subject_group = streams[stream_idx]['info']['desc'][0]['subject'][0]['group'][0]

            eeg_data = np.apply_along_axis(detrend, axis=1, arr=eeg_data)
            eeg_data = eeg_data - np.mean(eeg_data, axis=-1, keepdims=True)

            print(f" File {path_to_xdf} uses stream {stream_idx}")
            break

    fig, axs = plt.subplots(4, 4, figsize=(20, 10))
    for i in range(4):
        for j in range(4):
            axs[i, j].plot(eeg_times, eeg_data[i + j * 4, :])
            axs[i, j].set_title(channel_names[i + j * 4])
            axs[i, j].set_xlabel("Time")
            axs[i, j].set_ylabel("Amp")
    plt.tight_layout()
    plt.show()

    marker_df = pd.read_csv(directory_path + path_to_xdf.split('.')[0] + '.xdf.curated.markers.csv')
    column_name = marker_df.columns.to_list()
    if 'event' not in column_name:
        marker_df.rename(columns={column_name[0]: 'event', column_name[1]: 'time'}, inplace=True)

    # Initialize a trial index column
    marker_df['trial_index'], marker_df['block_index'], marker_df['block_type'] = None, None, None

    # Variable to keep track of the current trial index
    current_trial_index, current_block_index, current_block_type = None, None, None

    # Iterate through the DataFrame to assign trial index based on "block-begin-x" events
    for index, row in marker_df.iterrows():
        if 'trial-begin' in row['event']:
            # Extract the trial index from the event string
            current_trial_index = int(row['event'].split('-')[-1])
        marker_df.at[index, 'trial_index'] = current_trial_index

        if 'block-begin' in row['event']:
            # Extract the trial index from the event string
            current_block_index = int(row['event'].split('-')[2])

            if 'type' in row['event']:
                current_block_type = row['event'].split('_')[-1]
                marker_df.at[index, 'block_type'] = current_block_type
            else:
                raise ValueError("Can not locate the block type")
        marker_df.at[index, 'block_index'] = current_block_index
        marker_df.at[index, 'block_type'] = current_block_type

    marker_df['subject_group'] = subject_group
    marker_df['stim'] = None
    marker_df['stim_indicator'] = None
    marker_df['go_nogo'] = None
    marker_df['exp_label'] = None
    marker_df['is_resp'] = None
    marker_df['is_correct'] = None
    marker_df['response_time'] = None
    marker_df['block_type'] = None
    marker_df['stim_desc'] = None
    for index, row in marker_df.iterrows():
        if 'stim_' in row['event']:
            stim_type = row['event'].split('_')[1]
            block_type = row['event'].split('_')[2]
            task_type = row['event'].split('_')[3]
            stim_desc = row['event'].split('_')[4]

            if stim_type in ['stp', 'msp', 'ctl']:
                marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'stim'] = stim_type
            else:
                raise ValueError(f"Stim type mismatch: {stim_type} in {row['event']}")

            if block_type in ['w+e', 'w-e']:
                marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'stim_indicator'] = 'word'
                marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'block_type'] = block_type
            elif block_type in ['i+e', 'i-e']:
                marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'stim_indicator'] = 'image'
                marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'block_type'] = block_type
            else:
                raise ValueError(f"Block type type mismatch: {block_type} in {row['event']}")

            if task_type in ['nogo', 'go']:
                marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'go_nogo'] = task_type
            else:
                raise ValueError(f"task type mismatch: {task_type} in {row['event']}")

            marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'stim_desc'] = stim_desc

        if 'resp_' in row['event']:
            is_resp = row['event'].split('_')[-1]
            is_correct = row['event'].split('_')[-2]

            if is_resp in ['noresp']:
                marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'is_resp'] = False
            elif is_resp.isdigit():
                marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'is_resp'] = True
                marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'response_time'] = float(is_resp)
            else:
                raise ValueError(f"is_resp mismatch: {is_resp} in {row['event']}")

            if is_correct in ['correct', 'incorrect']:
                marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'is_correct'] = is_correct == 'correct'
            else:
                raise ValueError(f"is_correct mismatch: {is_correct} in {row['event']}")

    """
    marker_df['stim_indicator'] = None
    marker_df['go_nogo'] = None
    marker_df['exp_label'] = None
    marker_df['is_resp'] = None
    marker_df['is_correct'] = None
    for index, row in marker_df.iterrows():
        if 'word' in row['event']:
            marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'stim_indicator'] = 'word'
        elif 'img' in row['event']:
            marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'stim_indicator'] = 'img'

        if '-go-' in row['event'] or '_go_' in row['event']:
            marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'go_nogo'] = 'go'
        elif '-nogo-' in row['event'] or '_nogo_' in row['event']:
            marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'go_nogo'] = 'nogo'

        if '-exp-' in row['event'] or '_exp_' in row['event']:
            marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'exp_label'] = 'exp'
        elif '-noexp-' in row['event'] or '_noexp_' in row['event']:
            marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'exp_label'] = 'noexp'
        elif '-ctl-' in row['event'] or '_ctl_' in row['event']:
            marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'exp_label'] = 'ctrl'

        if '-resp' in row['event'] or '_resp' in row['event']:
            marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'is_resp'] = True
        elif '-noresp' in row['event'] or '_noresp' in row['event']:
            marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'is_resp'] = False

        if '_correct_' in row['event']:
            marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'is_correct'] = True
        elif '_incorrect_' in row['event']:
            marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'is_correct'] = False
    """

    marker_df = marker_df.dropna(subset=['trial_index'])
    grouped = marker_df.groupby('trial_index')

    # Create a new DataFrame to hold the reformatted data
    formatted_data = pd.DataFrame()

    # Define the columns for the new DataFrame
    formatted_columns = ['trial_number', 'block_number', 'block_type', 'stim_indicator', 'subject_group',
     'stim', 'go_nogo', 'is_experienced', 'is_resp', 'is_correct', 'response_time', 'trial_begin_time', 'stim_time',
     'trial_end_time', 'stim_desc', 'stim_type_info']


    # Initialize these columns in the formatted DataFrame
    for col in formatted_columns:
        formatted_data[col] = None

    # Iterate over each group, extract the times, and populate the new DataFrame
    for name, group in grouped:
        trial_index = int(name)
        trial_begin_time = group[group['event'].str.contains('trial-begin')]['time'].values[0]
        stim_time = group[group['event'].str.contains('stim')]['time'].values[0]
        trial_end_time = group[group['event'].str.contains('trial-end')]['time'].values[0]
        response_time = group['response_time'].values[0]

        """
        try:
            response_time = group[group['event'].str.contains('response-received')]['time'].values[0]
        except:
            response_time = None
        """

        stim_indicator = group['stim_indicator'].values[0]  # Assuming this value is constant within each trial
        go_nogo = group['go_nogo'].values[0]  # Assuming this value is constant within each trial
        is_resp = group['is_resp'].values[0]  # Assuming this value is constant within each trial
        is_correct = group['is_correct'].values[0]  # Assuming this value is constant within each trial
        block_type = group['block_type'].values[0]
        block_number = int(group['block_index'].values[0])
        stim = group['stim'].values[0]
        stim_desc = group['stim_desc'].values[0]
        is_experienced = stim == subject_group

        stim_row = group[group['event'].str.contains('stim_')].iloc[0] if not group[
            group['event'].str.contains('stim_')].empty else None
        img_or_word_row = group[(group['event'].str.contains('img-')) | (group['event'].str.contains('word-'))]
        img_or_word_row = img_or_word_row['event'].values
        if len(img_or_word_row) == 0:
            img_or_word_row = None

        # Add the row to the new DataFrame
        new_trial_row = {
            'trial_number': trial_index,
            'block_number': block_number,
            'block_type': block_type,
            'stim_indicator': stim_indicator,
            'subject_group': subject_group,
            'stim': stim,
            'go_nogo': go_nogo,
            'is_experienced': is_experienced,
            'is_resp': is_resp,
            'is_correct': is_correct,
            'response_time': response_time,
            'trial_begin_time': trial_begin_time,
            'stim_time': stim_time,
            'trial_end_time': trial_end_time,
            'stim_desc': stim_row['event'],
            'stim_type_info': stim_desc
        }
        for col in formatted_data.columns:
            if formatted_data[col].dtype == 'object' and all(
                    isinstance(val, bool) for val in formatted_data[col].dropna()):
                formatted_data[col] = formatted_data[col].astype(bool)
        new_trial_df = pd.DataFrame([new_trial_row.values()],
                                    columns=list(new_trial_row.keys()))
        formatted_data = pd.concat([formatted_data, new_trial_df], ignore_index=True)

    formatted_data.reset_index(drop=True, inplace=True)
    formatted_data['trial_number'] = pd.to_numeric(formatted_data['trial_number'],
                                                  errors='coerce')  # This will convert to numeric and set errors to NaN
    test_data_sorted = formatted_data.dropna(subset=['trial_number']).sort_values(
        by='trial_number')  # Drop rows where trial_index could not be converted
    test_data_sorted.set_index('trial_number', inplace=True)
    formatted_data.to_csv(directory_path + path_to_xdf.split('.')[0] + '_reformatted.csv', index=False)

    trial_begin_time = formatted_data['trial_begin_time']
    trial_onset_time = formatted_data['stim_time']
    trial_response_time = formatted_data['response_time']
    trial_end_time = formatted_data['trial_end_time']

    trial_length = trial_end_time - trial_onset_time
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(trial_length.values, bins=30, alpha=0.7, color='#1f77b4', edgecolor='black',
                                linewidth=1.2)

    # Title and labels with appropriate font sizes
    plt.title('Histogram of Trial lengths', fontsize=18, fontweight='bold')
    plt.xlabel('Length trial (second)', fontsize=16)
    plt.ylabel('Number of Trials', fontsize=16)

    # Adjusting tick parameters for readability
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Enhancing the grid for better readability while keeping it unobtrusive
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # Removing top and right spines for a cleaner look
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Tight layout for better spacing
    plt.tight_layout()
    plt.show()

    """
    trial_begin_df = marker_df[marker_df['event'].str.contains('trial-begin')]
    trial_end_df = marker_df[marker_df['event'].str.contains('trial-end')]
    stim_time_df = marker_df[marker_df['event'].str.contains('stim')]
    resp_time_df = marker_df[marker_df['event'].str.contains('response')]
    
    img_labels_df = marker_df[marker_df['event'].str.contains('img')]['event']
    word_labels_df = marker_df[marker_df['event'].str.contains('word')]['event']
    """

    eeg_data_list = []
    real_time, relative_time = [], []
    trial_length = []
    for start_time, end_time, stim_time in zip(trial_begin_time, trial_end_time, trial_onset_time):
        idx_start = np.argmin(np.abs(eeg_times - start_time))
        idx_end = np.argmin(np.abs(eeg_times - end_time))
        idx_stim = np.argmin(np.abs(eeg_times - stim_time))

        trial_length.append((idx_end - idx_stim) / s_rate)
        idx_start = int(idx_stim - 2 * s_rate)
        idx_end = int(idx_stim + s_rate)

        eeg_data_list.append(eeg_data[:, idx_start:idx_end])
        real_time.append(eeg_times[idx_start:idx_end])

        time = (np.arange(0, idx_end - idx_start) - (idx_stim - idx_start)) / s_rate

        relative_time.append(time)

    eeg_data_array = np.stack(eeg_data_list)

    label_str = formatted_data['go_nogo'].values
    label = np.array([val == 'go' for val in label_str])

    channel = 1
    plt.plot(relative_time[0], np.mean(eeg_data_array[label == 'go', channel, :], axis=0))
    plt.plot(relative_time[0], np.mean(eeg_data_array[label == 'nogo', channel, :], axis=0))
    plt.xlabel("Time")
    plt.ylabel("ERP")
    plt.show()
    plt.show()
