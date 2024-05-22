from mne.time_frequency import psd_array_multitaper
from src.utils.spectral_matrix.spectral_features import SpectralMatrixFeatures
import pandas as pd
import logging
import numpy as np
import pickle as pkl
import os
from pathlib import Path


class FeatureExtractor:
    """
    A class for extracting features from EEG data.

    This class provides methods for extracting various features such as time-domain features,
    coherence features, and frequency-domain features from EEG data.

    Parameters
    ----------
    paths : object
        An object containing various file paths necessary for feature extraction.
    settings : object
        An object containing various settings for feature extraction.

    Attributes
    ----------
    fs : float or None
        The sampling frequency of the EEG data.
    time : numpy array or None
        Time vector for EEG data.
    all_patient_features : dict
        Dictionary to store features for all patients.
    paths : object
        Object containing paths for feature extraction.
    settings : object
        Object containing settings for feature extraction.
    """

    def __init__(self, paths, settings):
        """
        Initialize the FeatureExtractor with EEGDataSet and sampling frequency.

        Args:
        paths (Path): An instance of the EEGDataSet class containing EEG data.
        settings (Settings): analysis settings
        """
        self.fs = None
        self.time = None
        self.all_patient_features = {}
        self.paths = paths
        self.settings = settings
        logging.basicConfig(filename=paths.path_result + 'feature_extraction_log.txt', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def extract_features(self, eeg_dataset, config):
        """
        Extract features from EEG data for all patients in the EEGDataSet.

        This method iterates over all patient data in the provided EEGDataSet instance
        and extracts features from the EEG data.

        Parameters
        ----------
        eeg_dataset : EEGDataSet
            The EEGDataSet instance containing patient data.

        Returns
        -------
        EEGDataSet
            The EEGDataSet instance with features extracted for each patient.
        """
        for index, (patient_id, patient_dataset) in enumerate(eeg_dataset.all_patient_data.items()):
            # Print progress information
            print(f"Subject {index} from {len(eeg_dataset.all_patient_data.keys())}: {patient_id} Feature Extraction")

            # Log progress information
            logging.info(f"Subject {index} from {len(eeg_dataset.all_patient_data.keys())}: {patient_id} "
                         f"Feature Extraction ...")

            # Set the time and sampling frequency attributes for feature extraction
            self.time = patient_dataset.time_ms
            self.fs = patient_dataset.fs

            # Define the path to save or load feature data
            feature_file = os.path.join(self.paths.feature_path, f"{patient_id}_features.json")

            # Check if feature data already exists and should be loaded
            if os.path.exists(feature_file) and self.settings.load_features is True:
                # Load and store existing feature data
                self.all_patient_features[patient_id] = self.load_features(feature_file)
                logging.info("Successfully Loaded!")
            else:
                # Extract features for the current patient dataset
                features = self.apply_feature_extraction(patient_dataset, **config)

                # Store the extracted features
                self.all_patient_features[patient_id] = features

                # Save the features to a file for future use
                self.save_features(features, feature_file)
                logging.info("Successfully Extracted!")

        return eeg_dataset

    def apply_feature_extraction(self, dataset, **kwargs) -> dict:
        """
        Apply feature extraction process on the given EEG data.

        Parameters
        ----------
        dataset : EEGDataSet
            An instance of EEGDataSet containing data for feature extraction.

        Returns
        -------
        dict
            A dictionary containing the extracted features.
        """
        # Initialize an empty dictionary to store the extracted features
        features = {}

        for feature_type, params in kwargs.items():
            if feature_type == 'time_n200':
                features['time_n200'] = self.extract_time_features(dataset.data, **params)
            elif feature_type == 'time_p300':
                features['time_p300'] = self.extract_time_features(dataset.data, **params)
            elif feature_type == 'time_post_p300':
                features['time_post_p300'] = self.extract_time_features(dataset.data, **params)
            elif feature_type.startswith('coherence'):
                features.update(self.extract_coherence_features(dataset, **params))
            elif feature_type.startswith('frequency'):
                features.update(self.extract_frequency_features(dataset, **params))
            # Add more feature extraction steps as needed

        return features

    def get_feature_array(self, eeg_dataset):
        """
        Extract features from EEG data for all patients in the EEGDataSet.

        Args:
            eeg_dataset:  EEGDataSet instance containing patient data.

        Returns:
            tuple: A tuple containing the extracted features, labels, patient IDs, and feature names.

        """
        # Initialize lists to store extracted features, labels, and patient IDs
        train_data, train_patient, train_patient_name = [], [], []

        label = eeg_dataset.all_patient_data[list(self.all_patient_features.keys())[0]].labels
        train_labels = {key: [] for key in label.keys()}

        for patient_index, (patient_id, patient_features) in enumerate(self.all_patient_features.items()):
            print(f"sub {patient_index} from {len(self.all_patient_features.keys())}")
            features_list, features_list_name = [], []
            label = eeg_dataset.all_patient_data[patient_id].labels

            if self.settings.dataset == 'pilot01':
                response_time_patient = eeg_dataset.all_patient_data[patient_id].response_time[:, None]
                # features_list.append(response_time_patient)
                # features_list_name.extend(['reaction_time'])
            if self.settings.dataset == 'verbmem':
                old_new_label = eeg_dataset.all_patient_data[patient_id].trial_type - 1
                decision_label = eeg_dataset.all_patient_data[patient_id].decision * 0.5 + 0.5
                trial_block = eeg_dataset.all_patient_data[patient_id].block_number
            channel_names = eeg_dataset.all_patient_data[patient_id].channel_names

            for trial_index, (feature_name, subject_features) in enumerate(patient_features.items()):
                # Get the label and response time for the current trial
                if feature_name.startswith('coh_') and feature_name.endswith('_freqs'):
                    pass
                else:
                    if subject_features.shape[-1] == len(channel_names):
                        feature_labels = [ch + '-' + feature_name for ch in channel_names]
                        features_list_name.extend(feature_labels)
                    elif feature_name == 'coh_tot_coh':
                        feature_labels = ['total coherency ' + freq for freq in ['5', '8', '13', '30']]
                        features_list_name.extend(feature_labels)
                    else:
                        subject_features = subject_features.reshape(subject_features.shape[0], -1)
                        frequencies = [5, 8, 13, 30]
                        groups = ['group', 'group1', 'group2', 'group3', 'group4', 'group5', 'group6', 'group7']
                        feature_labels = [f'coh_vec_coh_{freq}_{group}' for freq in frequencies for group in groups]
                        features_list_name.extend(feature_labels)
                    if subject_features.shape[-1] != len(feature_labels):
                        raise ValueError("feature_labels should be the same size with subject features")
                    features_list.append(subject_features)
            train_data.append(np.concatenate(features_list, axis=1))
            for key in train_labels.keys():
                train_labels[key].append(label[key])

            train_patient.append(np.ones(features_list[-1].shape[0]) * patient_index)
            train_patient_name.extend([patient_id] * features_list[-1].shape[0])

        # Convert lists to numpy arrays
        patients_ids = np.concatenate(train_patient, axis=0)
        labels_array = train_labels.copy()
        for key in train_labels.keys():
            labels_array[key] = np.concatenate(train_labels[key], axis=0)

        # old_new_label_array = np.concatenate(train_old_new_label, axis=0)
        # decision_label_array = np.concatenate(train_decision_label, axis=0)
        # block_idx_array = np.concatenate(train_block_idx, axis=0)

        features_matrix = np.concatenate(train_data, axis=0)

        features_df = pd.DataFrame(features_matrix, columns=features_list_name)
        features_df['id'] = patients_ids
        features_df['subject_file'] = train_patient_name
        for key in train_labels.keys():
            features_df[key] = labels_array[key]

        return features_df, features_matrix, labels_array, patients_ids, features_list_name


    def extract_time_features(self, data, start_time=150, end_time=250):
        """
        Extract time-domain features from EEG data within a specified time window.

        Parameters
        ----------
        data : numpy.ndarray
            The EEG data from which to extract time features.
        start_time : int
            The starting time (in ms) for the feature extraction window.
        end_time : int
            The ending time (in ms) for the feature extraction window.

        Returns
        -------
        numpy.ndarray
            Extracted time-domain features.
        """
        # Define time indices based on sampling frequency
        # conversion from time to index based on milli-second
        data = data / np.sqrt(np.sum(np.square(data), axis=-1, keepdims=True))
        ind = self._time_to_indices(start_time=start_time, end_time=end_time)
        return np.mean(data[:, :, ind[0]:ind[1]], axis=-1)

    def extract_coherence_features(self, dataset, time_start=50, end_time=1050):
        """
        Extract coherence features from EEG data.

        Parameters
        ----------
        dataset : EEGDataSet
            The EEGDataSet instance containing the EEG data.
        time_start : int
            The starting time (in ms) for the coherence analysis.
        end_time : int
            The ending time (in ms) for the coherence analysis.

        Returns
        -------
        dict
            A dictionary containing extracted coherence features.
        """
        eeg_data = dataset.data
        channel_groups = dataset.channel_group

        # Convert time to indices
        start_idx, end_idx = self._time_to_indices(time_start, end_time)

        connectivity_features = {}  # Initialize a dictionary to store coherence features

        for ind in range(eeg_data.shape[0]):  # Loop over EEG data samples
            # Assume temp is an MNE Epochs object
            temp = eeg_data[ind]
            max_ch = min(len(ch_list[0]) for ch_list in channel_groups)
            Data = np.zeros((max_ch, len(channel_groups), end_idx - start_idx))

            for grp_idx, ch_list in enumerate(channel_groups):  # Loop over channel groups
                ch_list = np.squeeze(ch_list)
                for ch_idx in range(max_ch):
                    if ch_idx < len(ch_list):
                        ch = ch_list[ch_idx] - 1
                        eeg_x = temp[ch, start_idx:end_idx]  # Extract the EEG segment
                        Data[ch_idx, grp_idx, :] = eeg_x - np.mean(eeg_x)  # Subtract mean and assign
                    else:
                        # Handle cases where the current channel group has less than max_ch channels
                        Data[ch_idx, grp_idx, :] = np.nan  # or some other placeholder value

            # Coherence Analysis
            fmin, fmax = 4, 60  # Define the frequency band of interest

            for method in ['coh']:  # Choose the coherence method (e.g., 'coh', 'imcoh', 'plv', etc.)
                group_names = [f'group{idx}' for idx in range(Data.shape[1])]

                # Initialize an object for spectral matrix features
                spectral_features = SpectralMatrixFeatures(dataset)

                # Calculate the spectral matrix and coherence features
                spectral_features.calculate_matrix(Data, fmin=fmin, fmax=fmax, ch_names=group_names, bandwidth=5,
                                                   adaptive=True, desired_freqs=[4, 8, 13, 30], verbose=False)
                coh_matrix, coh_tot, coh_ent, coh_vec = spectral_features.coherency_matrix()

                # Store the coherence features in the connectivity_features dictionary
                if 'coh_tot_' + method in connectivity_features.keys():
                    connectivity_features['coh_tot_' + method].append(coh_tot)
                else:
                    connectivity_features['coh_tot_' + method] = [coh_tot]

                if 'coh_vec_' + method in connectivity_features.keys():
                    connectivity_features['coh_vec_' + method].append(coh_vec)
                else:
                    connectivity_features['coh_vec_' + method] = [coh_vec]

                connectivity_features['coh_' + method + '_freqs'] = spectral_features.freqs

        # Stack the coherence features into numpy arrays
        for method in connectivity_features.keys():
            connectivity_features[method] = np.stack(connectivity_features[method], axis=0)

        return connectivity_features

    def extract_frequency_features(self, dataset, time_start=0, end_time=750):
        """
        Extract frequency-domain features from EEG data.

        Parameters
        ----------
        dataset : EEGDataSet
            The EEGDataSet instance containing the EEG data.
        time_start : int
            The starting time (in ms) for the frequency analysis.
        end_time : int
            The ending time (in ms) for the frequency analysis.

        Returns
        -------
        dict
            A dictionary containing extracted frequency-domain features.
        """
        eeg_data = dataset.data  # Assuming this is a 3D array (trials x channels x timepoints)
        features = {}

        # Frequency bands
        freq_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (13, 30),
            'gamma': (30, 45)
        }

        # Convert time to indices
        start_idx, end_idx = self._time_to_indices(time_start, end_time)

        for band_name, freq_range in freq_bands.items():
            # Compute power in the given frequency band
            features['freq_' + band_name + f'_time {time_start} to {end_time}'] = self._compute_band_power(eeg_data,
                                                                                                           freq_range,
                                                                                                           start_idx,
                                                                                                           end_idx)

        return features

    def _compute_band_power(self, eeg_data, freq_range, start_idx, end_idx):
        """
        Compute the average power in a specific frequency band for EEG data.

        Parameters
        ----------
        eeg_data : numpy.ndarray
            The EEG data for power computation.
        freq_range : tuple
            A tuple containing the lower and upper frequency limits of the band.
        start_idx : int
            The starting index for the analysis.
        end_idx : int
            The ending index for the analysis.

        Returns
        -------
        numpy.ndarray
            The average power in the specified frequency band.
        """
        band_power = []
        for trial in eeg_data:
            # Compute PSD using multitaper method
            psd, freqs = psd_array_multitaper(trial[:, start_idx:end_idx], sfreq=self.fs,
                                              fmin=freq_range[0], fmax=freq_range[1],
                                              adaptive=True, normalization='full', verbose=False)
            # Average power in the frequency band
            avg_power = np.mean(psd, axis=-1)  # Average across frequencies

            band_power.append(avg_power)
        return np.array(band_power)

    def _time_to_indices(self, start_time, end_time):
        """
        Convert time in milliseconds to indices in the EEG data array.

        Parameters
        ----------
        start_time : int
            The start time in milliseconds.
        end_time : int
            The end time in milliseconds.

        Returns
        -------
        tuple
            A tuple containing the start and end indices corresponding to the given times.
        """
        start_index = np.argmin(np.abs(self.time - start_time))
        end_index = np.argmin(np.abs(self.time - end_time))
        return start_index, end_index

    def save_features(self, features, file_path):
        """
        Save extracted features to a file.

        Parameters
        ----------
        features : dict
            The extracted features to be saved.
        file_path : str
            The file path where the features will be saved.
        """

        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            pkl.dump(features, f)

    def load_features(self, file_path):
        """
        Load features from a file.

        Parameters
        ----------
        file_path : str
            The file path from which to load the features.

        Returns
        -------
        dict
            The loaded features.
        """
        with open(file_path, 'rb') as f:
            return pkl.load(f)
