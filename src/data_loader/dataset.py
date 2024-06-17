import pickle

import mne
import torch
from typing import List, Union
import os
import numpy as np
import scipy.io
import logging
from abc import ABC, abstractmethod
import pyxdf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from torch.utils.data import Dataset

from src.visualization.visualization_utils import plot_histogram
from ieeg_data_loader.data import iEEGDataLoader
import h5py

class AbstractEEGDataLoader(ABC):
    def __init__(self, paths: object, settings):
        """
        Initialize the data loader with paths and settings.

        Parameters
        ----------
        paths : object
            An object containing various file paths necessary for data loading.
        settings : object
            An object containing various settings for data processing.
        """
        self.fs = None
        self.paths = paths
        self.data_directory = paths.raw_dataset_path
        self.channel_group_file = paths.channel_group_file
        self.history_length = 1
        # Initializations can vary based on subclass implementation
        self.channel_groups = None
        self.all_patient_data = {}
        self.mne_data = {}
        self.settings = settings

    @abstractmethod
    def load_data(self, patient_ids='all'):
        """
        Load EEG data for specified patients.

        Parameters
        ----------
        patient_ids : str or list of str, optional
            The IDs of the patients whose data is to be loaded.
            If 'all', data for all patients in the directory will be loaded.
        """
        pass

    @abstractmethod
    def load_single_patient_data(self, data, dgd_outputs=None):
        """
        Process and load data for a single patient.

        Parameters
        ----------
        data : dict
            The raw data loaded from the file.
        dgd_outputs : ndarray
            The DGD output data corresponding to the patient.

        Returns
        -------
        Any
            An instance containing the processed data for a single patient.
        """
        pass


class PilotEEGDataLoader(AbstractEEGDataLoader):
    def __init__(self, paths, settings):
        super().__init__(paths, settings)
        self.channel_groups = scipy.io.loadmat(self.channel_group_file)['ChGrp'][0]
        self.settings = settings

    def load_data(self, patient_ids='all'):
        """
        Load EEG data for specified patients.

        This method loads the EEG data and associated DGD outputs for each patient
        and processes it into a format suitable for further analysis.

        Parameters
        ----------
        patient_ids : str or list of str, optional
            The IDs of the patients whose data is to be loaded.
            If 'all', data for all patients in the directory will be loaded.

        Raises
        ------
        ValueError
            If no files are found for the specified patient IDs.
        """
        file_list = os.listdir(self.data_directory)
        if isinstance(patient_ids, str) and patient_ids == 'all':
            xdf_file_list = [file for file in file_list if file.endswith(".xdf") or file.endswith(".h5")]
        else:
            if isinstance(patient_ids, list) is False:
                patient_ids = [patient_ids]
            xdf_file_list = []
            for patient in patient_ids:
                xdf_file_list.extend([f for f in os.listdir(self.data_directory) if f.endswith(".xdf") or f.endswith(".h5")
                                      and f.startswith(str(patient))])

            if len(file_list) == 0:
                raise ValueError(f"No patient found with name {patient_ids}")

        for index, file_name in enumerate(xdf_file_list):
            print(f"Subject {index} from {len(xdf_file_list)}: {file_name.split('.')[0]} Load Data")
            logging.info(f"Subject {index} from {len(file_list)}: {file_name.split('.')[0]} Load Data ...")
            # If specific patient IDs are provided, skip files not matching those IDs

            # Attempt to load HD5 file if available
            dataset = self.load_hd5_file(file_name)
            if dataset is None:
                prepaired_data_path = self.paths.raw_dataset_path + file_name.split('.')[0] + "_trials.pkl"
                if os.path.exists(prepaired_data_path) and self.settings.load_epoched_data is True:
                    with open(prepaired_data_path, 'rb') as file:
                        dataset = pickle.load(file)
                else:
                    dataset = self.load_single_patient_data(file_name)
                    if self.settings.save_epoched_data is True:
                        dataset.save_to_pickle(
                            file_path=self.paths.raw_dataset_path + file_name.split('.')[0] + "_trials.pkl")
            self.all_patient_data[file_name.split('.')[0]] = dataset

    def load_single_patient_data(self, data, dgd_outputs=None, preprocess_continuous_data=False):
        file_name = data

        file_path = os.path.join(self.data_directory, file_name)

        streams, fileheader = pyxdf.load_xdf(file_path)
        for stream_idx in range(len(streams)):
            channel_info = streams[stream_idx]['info']['desc'][0]['channels']
            if len(channel_info) > 0 and len(channel_info[0]['channel']) > 100:
                # Assuming you have one EEG stream, and it's the first stream (index 0)
                eeg_data = streams[stream_idx]['time_series'].T  # Transpose to have channels as rows
                eeg_times = streams[stream_idx]['time_stamps'] - streams[stream_idx]['time_stamps'][0]
                channel_info = streams[stream_idx]['info']['desc'][0]['channels']
                channel_names = [channel['label'][0] for channel in channel_info[0]['channel']]
                subject = streams[stream_idx]['info']['desc'][0]['subject']
                stream_id = streams[stream_idx]['info']['stream_id']
                s_rate = float(streams[stream_idx]['info']['nominal_srate'][0])
                reference_electrode = streams[stream_idx]['info']['desc'][0]['reference'][0]['label']
                subject_id = streams[stream_idx]['info']['desc'][0]['subject'][0]['id'][0]
                if len(streams[stream_idx]['info']['desc'][0]['subject'][0]['group']) > 0:
                    subject_group = streams[stream_idx]['info']['desc'][0]['subject'][0]['group'][0]
                else:
                    subject_group = None

                channel_types = ['eeg' for _ in range(eeg_data.shape[0])]

                print(f" File {file_name} uses stream {stream_idx}")
                break

        fig, axs = plt.subplots(4, 4, figsize=(40, 10))
        for i in range(4):
            for j in range(4):
                axs[i, j].plot(eeg_times[:1000], eeg_data[i + j * 4, :1000])
                axs[i, j].set_title(channel_names[i + j * 4])
                axs[i, j].set_xlabel("Time")
                axs[i, j].set_ylabel("Amp")
        plt.tight_layout()
        plt.show()

        keep_indices = [i for i, name in enumerate(channel_names) if "AUX" not in name and "Trig" not in name]
        # Filter the EEG data to remove channels with "AUX" in their names
        eeg_data = eeg_data[keep_indices, :]

        # Also, filter the channel names list
        channel_names = [name for i, name in enumerate(channel_names) if i in keep_indices]

        if eeg_data.shape[0] != len(channel_names):
            eeg_data = eeg_data.T  # Transpose if necessary

            # Create an Info object

        # Create the RawArray
        if preprocess_continuous_data is True:
            info = mne.create_info(ch_names=channel_names, sfreq=s_rate, ch_types='eeg')
            raw = mne.io.RawArray(eeg_data, info)
            raw.filter(l_freq=1.0, h_freq=60, phase='zero')
            raw.notch_filter(freqs=60, notch_widths=1)

            # asr = asrpy.ASR(sfreq=raw.info["sfreq"], cutoff=10)
            # asr.fit(raw)
            # raw = asr.transform(raw)

            eeg_data = raw._data

            eeg_data = np.apply_along_axis(detrend, axis=1, arr=eeg_data)

        eeg_data = eeg_data / np.quantile(np.abs(eeg_data), 0.99, axis=-1, keepdims=True)
        eeg_data = eeg_data - np.mean(eeg_data, axis=-1, keepdims=True)

        """
        fig, axs = plt.subplots(4, 4, figsize=(20, 10))
        for i in range(4):
            for j in range(4):
                axs[i, j].plot(eeg_times[:200], eeg_data[i + j * 4, :200])
                axs[i, j].set_title(channel_names[i + j * 4])
                axs[i, j].set_xlabel("Time")
                axs[i, j].set_ylabel("Amp")
        plt.tight_layout()
        plt.show()
        """
        formatted_marker_df = self._reformat_marker_file(
            marker_path=self.data_directory + file_name.split('.')[
                0] + '.recoded.merged.preproc.curated.result.xdf.markers.csv',
            file_name=file_name, subject_group=subject_group, load_reformatted_data=False)

        eeg_trial_data, eeg_trial_time, eeg_labels, trial_length, trial_index = self._convert_continuous_to_trial(
            eeg_times=eeg_times,
            eeg_data=eeg_data,
            s_rate=s_rate,
            file_name=file_name,
            formatted_marker_df=formatted_marker_df,
            save_data=True,
            load_trialed_data=False)

        plot_histogram(trial_length, xlabel='Length trial (second)', ylabel='Number of Trials',
                       title=f"Histogram of Trial lengths for subject {file_name.split('_')[0]}")

        try:
            bad_channels_df = pd.read_csv(self.data_directory + file_name + '.bad_channels.csv')
            bad_channels = bad_channels_df.columns.to_list()
        except:
            print("N vbadchannel is detected")
            bad_channels = []

        dataset = EEGDataSet()
        dataset.data = eeg_trial_data
        dataset.response_time = np.squeeze(trial_length)
        dataset.labels = eeg_labels
        dataset.trial_index = np.squeeze(trial_index)
        dataset.fs = s_rate
        dataset.time_ms = np.squeeze(np.round(eeg_trial_time * 1000))
        dataset.channel_names = channel_names
        dataset.file_name = file_name
        dataset.stream_id = stream_id
        dataset.reference_electrodes = reference_electrode
        dataset.channel_group = self.channel_groups
        dataset.bad_channels = bad_channels

        return dataset

    def decode_marker(self, marker):
        parts = marker.decode('utf-8').split('-')
        if len(parts) != 4:
            raise ValueError(f"Unexpected marker format: {marker}")
        return parts[0], parts[1], parts[2], parts[3]
    def load_hd5_file(self, patient_id):
        """
        Load HD5 file if available.

        Parameters
        ----------
        patient_id : str
            The ID of the patient whose HD5 data is to be loaded.

        Returns
        -------
        EEGDataSet or None
            The loaded EEG dataset or None if the HD5 file is not found.
        """
        hd5_file_path = os.path.join(self.data_directory, f"{patient_id}")
        if os.path.exists(hd5_file_path):
            print(f"Loading H5 file for patient {patient_id}")
            file = h5py.File(hd5_file_path, 'r')

            # Explore the structure of the file
            print("Keys: %s" % file.keys())
            eeg_data = np.squeeze(file['chunks']['eeg']['block']['data'])
            eeg_data = np.transpose(eeg_data, (0, 2, 1))

            eeg_times = np.squeeze(file['chunks']['eeg']['block']['axes']['axis1']['times'])

            # Access and decode channel names
            channel_names_raw = file['chunks']['eeg']['block']['axes']['axis2']['names']
            channel_names = [name.decode('utf-8') for name in channel_names_raw]

            # Print shapes and types to verify
            print(f"EEG data shape: {eeg_data.shape}")
            print(f"EEG times shape: {eeg_times.shape}")
            print(f"Number of channels: {len(channel_names)}")

            labels_data = np.array(file['chunks']['eeg']['block']['axes']['axis0']['data'])
            # Function to decode the first element


            processed_data = []
            for entry in labels_data:
                wrd_img, exp_noexp, go_nogo, correct = self.decode_marker(entry['Marker'])
                target_value = entry['TargetValue']
                is_correct = entry['IsGood']
                trial_index = entry['TrialIndex']
                target_trial_index_asc = entry['TargetTrialIndexAsc']
                processed_data.append(
                    (go_nogo, exp_noexp, wrd_img, target_value, is_correct, trial_index, target_trial_index_asc))

            # Creating DataFrame
            df_target = pd.DataFrame(processed_data,
                              columns=['go_nogo', 'is_experienced', 'Wrd_Img', 'TargetValue', 'is_correct', 'TrialIndex',
                                       'target_trial_index_asc'])

            dataset = EEGDataSet()
            dataset.data = eeg_data
            dataset.response_time = np.squeeze(eeg_times)
            dataset.labels = df_target  # Assuming labels need to be processed
            dataset.trial_index = []  # Assuming trial index needs to be processed
            dataset.fs = int(1/(eeg_times[1]-eeg_times[0]))
            dataset.time_ms = np.squeeze(np.round(eeg_times * 1000))
            dataset.channel_names = channel_names
            dataset.file_name = f"{patient_id}.hd5"
            dataset.stream_id = patient_id.split('.')[0]
            dataset.channel_group = self.channel_groups
            dataset.bad_channels = []  # Assuming bad channels need to be processed

            return dataset
        else:
            print(f"No HD5 file found for patient {patient_id}")
            return None

    def _reformat_marker_file(self, marker_path, file_name, subject_group, load_reformatted_data=True):
        if load_reformatted_data is True:
            formatted_data = pd.read_csv(self.data_directory + file_name.split('.')[0] + '_reformatted.csv')
        else:
            marker_df = pd.read_csv(marker_path)
            column_name = marker_df.columns.to_list()
            if 'event' not in column_name:
                marker_df.rename(columns={column_name[0]: 'event', column_name[1]: 'time'}, inplace=True)

            # Initialize a trial index column
            marker_df['trial_index'] = None

            # Variable to keep track of the current trial index
            current_trial_index = None

            # Initialize a trial index, block index and block type column
            marker_df['trial_index'], marker_df['block_index'], marker_df['block_type'] = None, None, None

            # Variable to keep track of the current trial index
            current_trial_index, current_block_index, current_block_type = None, None, None

            # Iterate through the DataFrame to assign trial index based on "block-begin-x" events
            for index, row in marker_df.iterrows():
                if 'subject_id' in row['event'] and subject_group is None:
                    subject_group = row['event'].split(':')[-1].replace(' ', '')
                if 'trial-begin' in row['event']:
                    # Extract the trial index from the event string
                    current_trial_index = int(row['event'].split('-')[-1])

                if 'block-begin' in row['event']:
                    # Extract the trial index from the event string
                    current_block_index = int(row['event'].split('-')[2])

                    if 'type' in row['event']:
                        current_block_type = row['event'].split('_')[-1]
                        marker_df.at[index, 'block_type'] = current_block_type
                    else:
                        raise ValueError("Can not locate the block type")
                marker_df.at[index, 'trial_index'] = current_trial_index
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

                    if block_type in ['w+e', 'w-e', 'w+e+x', 'w-e+x']:
                        marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'stim_indicator'] = 'word'
                        marker_df.loc[marker_df['trial_index'] == row['trial_index'], 'block_type'] = block_type
                    elif block_type in ['i+e', 'i-e', 'i+e+x', 'i-e+x']:
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
                        marker_df.loc[
                            marker_df['trial_index'] == row['trial_index'], 'is_correct'] = is_correct == 'correct'
                    else:
                        raise ValueError(f"is_correct mismatch: {is_correct} in {row['event']}")

            marker_df = marker_df.dropna(subset=['trial_index'])
            grouped = marker_df.groupby('trial_index')

            # Create a new DataFrame to hold the reformatted data
            formatted_data = pd.DataFrame()

            # Define the columns for the new DataFrame
            formatted_columns = ['trial_number', 'block_number', 'block_type', 'stim_indicator', 'subject_group',
                                 'stim', 'go_nogo', 'is_experienced', 'is_resp', 'is_correct', 'response_time',
                                 'response_time_real', 'trial_begin_time', 'stim_time',
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
                    'response_time': trial_end_time - stim_time,
                    'response_time_real': response_time,
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
            formatted_data.to_csv(self.data_directory + file_name.split('.')[0] + '_reformatted.csv', index=False)

            unique_blocks = formatted_data['block_number'].nunique()
            unique_block_types = [
                str(formatted_data[formatted_data['block_number'] == block_idx]['block_type'].unique()) for
                block_idx in formatted_data['block_number'].unique()]
            total_trials = formatted_data.shape[0]

            print(
                f"The marker contains {unique_blocks} blocks ({', '.join(unique_block_types)}) and {total_trials} trials")

        return formatted_data

    def _convert_continuous_to_trial(self, eeg_times, eeg_data, s_rate, formatted_marker_df, file_name,
                                     load_trialed_data=False,
                                     save_data=False):
        if load_trialed_data is True:
            with open(self.data_directory + file_name.split('.')[0] + '_trial_data.pkl', 'rb') as file:
                data_loaded = pickle.load(file)

            # Extract variables from the loaded dictionary
            eeg_data_array = data_loaded['eeg_data_array']
            eeg_time = data_loaded['eeg_time']
            eeg_labels = data_loaded['eeg_labels']
            trial_length = data_loaded['trial_length']
            trial_index = data_loaded['trial_index']
        else:
            trial_begin_times = formatted_marker_df['trial_begin_time']
            trial_onset_times = formatted_marker_df['stim_time']
            trial_response_times = formatted_marker_df['response_time']
            trial_end_times = formatted_marker_df['trial_end_time']

            trial_length = trial_end_times - trial_onset_times

            eeg_data_list, trial_index = [], []
            real_time, relative_time = [], []
            trial_length = []
            eeg_labels = {
                'block_number': [],
                'block_type': [],
                'stim_indicator': [],
                'go_nogo': [],
                'is_experienced': [],
                'is_resp': [],
                'is_correct': [],
                'stim': []
            }

            eeg_time = np.arange(int(- 2 * s_rate), int(s_rate)) / s_rate

            error_list = []
            for index, row in formatted_marker_df.iterrows():
                trial_begin_time = row['trial_begin_time']
                trial_onset_time = row['stim_time']
                trial_response_time = row['response_time']
                trial_end_time = row['trial_end_time']

                idx_start = np.argmin(np.abs(eeg_times - trial_begin_time))
                idx_end = np.argmin(np.abs(eeg_times - trial_end_time))
                idx_stim = np.argmin(np.abs(eeg_times - trial_onset_time))

                tl = (idx_end - idx_stim) / s_rate
                if tl < 0.8 * row['response_time']:
                    print(
                        f"Segmentation Error: The matched segment with Trial {row['trial_number']} has task duration "
                        f"{tl} which is less that expected duration {row['response_time']} ")
                    error_list.append(index)
                else:
                    eeg_labels['block_number'].append(row['block_number'])
                    eeg_labels['block_type'].append(row['block_type'])
                    eeg_labels['stim_indicator'].append(row['stim_indicator'])
                    eeg_labels['go_nogo'].append(row['go_nogo'])
                    eeg_labels['is_experienced'].append(row['is_experienced'])
                    eeg_labels['is_resp'].append(row['is_resp'])
                    eeg_labels['is_correct'].append(row['is_correct'])
                    eeg_labels['stim'].append(row['stim'])

                    trial_length.append((idx_end - idx_stim) / s_rate)

                    idx_start = int(idx_stim - 2 * s_rate)
                    idx_end = int(idx_stim + s_rate)

                    eeg_data_list.append(eeg_data[:, idx_start:idx_end])
                    real_time.append(eeg_times[idx_start:idx_end])

                    trial_index.append(row['trial_number'])

            eeg_data_array = np.stack(eeg_data_list)

            unique_blocks = np.unique(eeg_labels['block_number'])
            df = pd.DataFrame(eeg_labels)
            unique_block_types = [
                str(df[df['block_number'] == block_idx]['block_type'].unique()) for
                block_idx in df['block_number'].unique()]
            total_trials = eeg_data_array.shape[0]
            print(
                f"The trialed data contains {len(unique_blocks)} blocks ({', '.join(unique_block_types)}) and {total_trials} trials")

            if save_data is True:
                data_to_save = {
                    'eeg_data_array': eeg_data_array,
                    'eeg_time': eeg_time,
                    'eeg_labels': eeg_labels,
                    'trial_length': trial_length,
                    'trial_index': trial_index
                }

                # Save the dictionary into a file
                with open(self.data_directory + file_name.split('.')[0] + '_trial_data.pkl', 'wb') as file:
                    pickle.dump(data_to_save, file)

        return eeg_data_array, eeg_time, eeg_labels, trial_length, trial_index


class VerbMemEEGDataLoader(AbstractEEGDataLoader):
    def __init__(self, paths, settings):
        super().__init__(paths, settings)
        self.channel_groups = scipy.io.loadmat(self.channel_group_file)['ChGrp'][0]
        logging.basicConfig(filename=paths.path_result + 'data_loading_log.txt', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.exclude_list = ['1065', '1034']

    def load_data(self, patient_ids='all'):
        """
        Load EEG data for specified patients.

        This method loads the EEG data and associated DGD outputs for each patient
        and processes it into a format suitable for further analysis.

        Parameters
        ----------
        patient_ids : str or list of str, optional
            The IDs of the patients whose data is to be loaded.
            If 'all', data for all patients in the directory will be loaded.

        Raises
        ------
        ValueError
            If no files are found for the specified patient IDs.
        """
        "=================== Load and prepaire Data =================="
        file_extension = '_VerbMem_ecnt_af.mat'
        if isinstance(patient_ids, str) and patient_ids == 'all':
            file_list = [f for f in os.listdir(self.data_directory) if f.endswith(file_extension)]
        else:
            if isinstance(patient_ids, str):
                patient_ids = [patient_ids]
            file_list = []
            for patient in patient_ids:
                file_list.extend([f for f in os.listdir(self.data_directory) if f.endswith(file_extension)
                                  and f.startswith(str(patient))])
            if len(file_list) == 0:
                raise ValueError(f"No patient found with name {patient_ids}")

        for index, file_name in enumerate(file_list):
            print(f"Subject {index} from {len(file_list)}: {file_name.split('.')[0]} Load Data")
            logging.info(f"Subject {index} from {len(file_list)}: {file_name.split('.')[0]} Load Data ...")
            # If specific patient IDs are provided, skip files not matching those IDs
            file_path = os.path.join(self.data_directory, file_name)
            try:
                dgd_outputs = self.load_dgd_data(file_name)
            except:
                print(f"DGD output not available for {file_name}")
                logging.info(f"DGD output not available for {file_name}")
                dgd_outputs = None
            data = scipy.io.loadmat(file_path)
            if self._check_data_errors(data, dgd_outputs,
                                       file_name.partition('.')[0]) is True and dgd_outputs is not None:
                self.all_patient_data[file_name.split('.')[0]] = self.load_single_patient_data(data, dgd_outputs)

    def load_dgd_data(self, file_name):
        """
        Load DGD data for a given file name.

        Parameters
        ----------
        file_name : str
            The name of the file for which to load the DGD data.

        Returns
        -------
        ndarray
            The DGD feature data for the specified file.
        """
        file_path = self.paths.dgd_output_path + file_name.split('_')[0]
        b1_data = np.squeeze(np.stack(np.squeeze(scipy.io.loadmat(file_path + '_block1.mat')['XPos'])))
        b2_data = np.squeeze(np.stack(np.squeeze(scipy.io.loadmat(file_path + '_block2.mat')['XPos'])))
        b3_data = np.squeeze(np.stack(np.squeeze(scipy.io.loadmat(file_path + '_block3.mat')['XPos'])))
        x_feature = np.concatenate([b1_data, b2_data, b3_data], axis=0)
        return x_feature

    def load_single_patient_data(self, data, dgd_outputs):
        """
        Process and load data for a single patient.

        This method extracts relevant information from the loaded data file,
        restructures it, and prepares it for analysis.

        Parameters
        ----------
        data : dict
            The raw data loaded from the file.
        dgd_outputs : ndarray
            The DGD output data corresponding to the patient.

        Returns
        -------
        EEGDataSet
            An instance of EEGDataSet containing the processed data for a single patient.
        """
        # Extracting data from loaded file
        ecnt_af = data['ecnt_af'][0, 0]
        rt = ecnt_af['sweep']['rt'][0, 0] / 1000
        dec = ecnt_af['sweep']['correct'][0, 0]
        task = ecnt_af['sweep']['task'][0, 0]
        block = ecnt_af['sweep']['block'][0, 0]
        rec_old = ecnt_af['sweep']['rec_old'][0, 0]
        et_bin = ecnt_af['etbin_e']
        good_epochs = ecnt_af['badEpoch']['goodEpochs'][0, 0]
        eeg_data = ecnt_af['data']
        channel_names = [elec_name.replace(' ', '') for elec_name in list(ecnt_af['elecnames'])]
        channel_idx = ecnt_af['elec']
        self.fs = ecnt_af['samplerate'][0][0]

        # Restructure data
        tEEG = np.zeros((len(task), eeg_data.shape[0], et_bin[0, 1] - et_bin[0, 0] + 1))
        for i in range(len(rt)):
            if good_epochs[i] == 1:
                cnt = np.sum(good_epochs[:i + 1])
                tEEG[i] = eeg_data[:, et_bin[cnt - 1, 0] - 1:et_bin[cnt - 1, 1]]
            else:
                tEEG[i] = tEEG[i] * np.nan

        # Task = 3, information
        trial_ind = np.where(task == 3)[0]
        trial_type = rec_old[trial_ind]
        trial_dec = dec[trial_ind]
        trial_block = block[trial_ind]
        trial_rt = rt[trial_ind]

        FEAT = [None] * len(trial_ind)
        EEG_experiment = [None] * len(trial_ind)
        for i, base_ind in enumerate(trial_ind):
            feat = []
            for j in range(self.history_length + 1):
                temp = tEEG[base_ind - j] if base_ind - j >= 0 else None
                tfeat = []
                if temp is not None:
                    for grp in self.channel_groups:
                        grp_feat = temp[grp - 1, :]
                        tfeat.append(np.mean(grp_feat, axis=0))
                else:
                    feat = []
                    break
                feat.extend(tfeat)
            FEAT[i] = feat
            EEG_experiment[i] = tEEG[base_ind]

        EEG_experiment = np.stack(EEG_experiment)
        non_nan_idx = np.where(np.isnan(EEG_experiment[:, 0, 0]) == False)
        data = EEG_experiment[non_nan_idx]
        trial_rt = trial_rt[non_nan_idx]
        trial_type = trial_type[non_nan_idx]
        trial_dec = trial_dec[non_nan_idx]
        trial_block = trial_block[non_nan_idx]
        trial_ind = trial_ind[non_nan_idx]
        dgd_outputs = dgd_outputs[non_nan_idx]
        time = np.arange(0, data.shape[-1]) / self.fs - 0.3
        idx_onset = np.argmin(np.abs(time))

        dataset = EEGDataSet()
        dataset.data = data
        dataset.response_time = np.squeeze(trial_rt)
        dataset.decision = np.squeeze(trial_dec)
        dataset.trial_block = np.squeeze(trial_block)
        dataset.trial_type = np.squeeze(trial_type)
        dataset.trial_index = np.squeeze(trial_ind)
        dataset.fs = self.fs
        dataset.time_ms = np.squeeze(np.round(time * 1000))
        dataset.channel_names = channel_names
        dataset.channel_index = np.squeeze(channel_idx)
        dataset.channel_group = self.channel_groups
        dataset.dgd_outputs = dgd_outputs

        return dataset

    def to_mne_format(self) -> dict:
        """
        Convert all loaded patient data into MNE format.

        Returns
        -------
        dict
            A dictionary with patient IDs as keys and MNE data objects as values.
        """
        # Conversion implementation
        for patient_id, data in self.all_patient_data.items():
            self.mne_data[patient_id] = self.array_to_mne_epoch(data)

    def prepare_for_deep_learning(self, data: dict, batch_size: int = 32) -> torch.utils.data.DataLoader:
        """
        Prepare data for deep learning models in PyTorch.

        Parameters
        ----------
        data : dict
            A dictionary with patient data.
        batch_size : int, optional
            Batch size for the data loader.

        Returns
        -------
        torch.utils.data.DataLoader
            PyTorch DataLoader with the prepared data.
        """
        # Prepare data for PyTorch and return DataLoader
        pass

    def _check_data_errors(self, data, dgd_outputs, patient_name):
        """
        Check for errors in the loaded data.

        Parameters
        ----------
        data : dict
            The loaded data for a patient.
        patient_name : str
            The name of the patient.

        Returns
        -------
        bool
            True if no errors are found, False otherwise.
        """
        ecnt_af = data['ecnt_af'][0, 0]
        et_bin = ecnt_af['etbin_e']
        good_epochs = ecnt_af['badEpoch']['goodEpochs'][0, 0]
        if np.sum(good_epochs) != et_bin.shape[0]:
            logging.info(f"Error for {patient_name}: The number of good epochs is {np.sum(good_epochs)} ,But etbin_e"
                         f"size is {et_bin.shape} - Size mismatch")
            print(f"Error for {patient_name}: The number of good epochs is {np.sum(good_epochs)} ,But etbin_e"
                  f"size is {et_bin.shape} - Size mismatch")

            return False
        if patient_name.split('_')[0] in self.exclude_list:
            logging.info(f"Subject {patient_name} is in Exclude list")
            print(f"Subject {patient_name} is in Exclude list")
            return False

        if dgd_outputs is None:
            logging.info(f"dgd_outputs does not exist for Subject {patient_name}")
            print(f"dgd_outputs does not exist for Subject {patient_name}")
            return False
        elif np.sum(np.isnan(dgd_outputs[:, 1])) > 0:
            logging.info(f"dgd_outputs contains NaN for Subject {patient_name}")
            print(f"dgd_outputs contains NaN for Subject {patient_name}")
            return False

        return True


class CLEARDataLoader(AbstractEEGDataLoader):
    def __init__(self, paths, settings):
        super().__init__(paths, settings)
        self.channel_groups = scipy.io.loadmat(self.channel_group_file)['ChGrp'][0]
        logging.basicConfig(filename=paths.path_result + 'data_loading_log.txt', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def load_data(self, patient_ids='all', task='flicker', target_class='color'):
        data_loader = iEEGDataLoader(patient=patient_ids,
                                     target_class=target_class,
                                     prepared_dataset_path=self.data_directory,
                                     task=task,
                                     file_format='npy')
        dataset_list = data_loader.load_data()

        for patients_data in dataset_list:
            dataset = self.load_single_patient_data(patients_data)
            self.all_patient_data[patients_data.meta[-1]] = dataset

    def load_single_patient_data(self, data, dgd_outputs=None, preprocess_continuous_data=False):
        data.get_time_annotation(patient=data.meta[1], task=data.meta[0].split('_')[0])

        dataset = EEGDataSet()
        dataset.data = data.data
        dataset.response_time = None
        dataset.labels = data.label
        dataset.trial_index = np.arange(0, data.data.shape[0])
        dataset.fs = data.fs
        dataset.time_ms = data.time * 1000
        dataset.channel_names = data.channel_name
        dataset.file_name = data.meta[-1]
        dataset.task = data.meta[0]
        dataset.channel_group = self.channel_groups

        return dataset


class EEGDataSet:
    """
        A class to represent EEG dataset.

        Attributes
        ----------
        data : ndarray or None
            The EEG data.
        response_time : ndarray or None
            Array containing the response times of the trials.
        decision : ndarray or None
            Array containing the decision results of the trials.
        fs : float or None
            The sampling frequency of the EEG data.
        time_ms : ndarray or None
            Time in milliseconds for each data point.
        trial_block : ndarray or None
            The block number for each trial.
        trial_index : ndarray or None
            The index of each trial.
        trial_type : ndarray or None
            The type of each trial.
        channel_names : list of str or None
            Names of EEG channels.
        channel_index : ndarray or None
            Indices of the channels.
        channel_group : ndarray or None
            Group information of the channels.
        dgd_outputs : ndarray or None
            Outputs from the DGD process.
        __mne_data : object or None
            Private attribute to hold MNE data object.

        Methods
        -------
        mne_data()
            Property to get or calculate MNE data.
        array_to_mne_epoch()
            Converts array data to MNE Epochs object.
    """

    def __init__(self):
        """
        Initializes the EEGDataSet with default values.
        """
        self.data = None
        self.response_time = None
        self.decision = None
        self.labels = None
        self.fs = None
        self.time_ms = None
        self.trial_block = None
        self.trial_index = None
        self.trial_type = None
        self.channel_names = None
        self.channel_index = None
        self.channel_group = None
        self.bad_channels = None

        self.dgd_outputs = None
        self.__mne_data = None

        self.file_name = None
        self.stream_id = None
        self.reference_electrodes = None

    @property
    def mne_data(self):
        """
        Gets or calculates MNE data from the EEG dataset.

        This property checks if the MNE data is already calculated. If not,
        it calls the 'array_to_mne_epoch' method to calculate it.

        Returns
        -------
        object
            The MNE data object.
        """
        if self.__mne_data is not None:
            return self.__mne_data
        else:
            self.__mne_data = self.array_to_mne_epoch()

    def array_to_mne_epoch(self):
        """
        Converts the EEG dataset into an MNE Epochs object.

        This method classifies the channels based on their names and
        creates an MNE EpochsArray object using the data, channel information,
        and events derived from trial indices and decisions.

        Returns
        -------
        mne.EpochsArray
            An MNE EpochsArray object representing the EEG dataset.
        """
        channel_types = {}
        # Classify the channels
        for channel in self.channel_names:
            if channel.startswith('A') or channel.startswith('B') or channel.startswith('C') or channel.startswith('D'):
                channel_types[channel] = 'eeg'  # Assuming these are EEG channels
            elif channel.startswith('Ref'):
                channel_types[channel] = 'ref_meg'  # Assuming these are MEG reference channels
            elif channel.startswith('VEOG') or channel.startswith('HEOG'):
                # VEOG (Vertical Electrooculography) and HEOG (Horizontal Electrooculography) are used for tracking
                # eye movements.
                channel_types[channel] = 'eog'
            elif channel.startswith('rEKG') or channel.startswith('lEKG'):
                # These appear to be EKG (electrocardiogram) channels, possibly right and left side EKGs. These
                # should be classified as ‘ecg’
                channel_types[channel] = 'ecg'
            else:
                channel_types[channel] = 'misc'  # Default to miscellaneous

        channel_types = [channel_types[ch_name] for ch_name in self.channel_names]
        # Create an MNE Info object with the channel information
        info = mne.create_info(ch_names=self.channel_names,
                               sfreq=self.fs,
                               ch_types=channel_types)

        # Create events array using 'trial_ind' for indices and 'trial_dec' for event values
        n_trials = self.data.shape[0]
        events = np.column_stack((self.trial_index, np.zeros(n_trials, dtype=int), self.decision))

        # Create EpochsArray
        epochs = mne.EpochsArray(self.data, info, events=events)

        return epochs

    def save_to_pickle(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)


class TorchEEGDataset(Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
