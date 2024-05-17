import matplotlib.pyplot as plt

from .utils import *
import numpy as np
import logging

import seaborn as sns

def plot_feature_heatmap(features_matrix, selected_features, old_new_labels, decision_labels):
    plt.figure(figsize=(20, 10))
    sns.heatmap(features_matrix.T, cmap='viridis', yticklabels=selected_features)
    plt.xlabel('Trial Index')
    plt.ylabel('Features')
    plt.title('Features Heatmap by Trial Index with Decision and Old/New Labels')

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(20, 2))
    sns.heatmap(np.concatenate([old_new_labels[:, None],
                                decision_labels[:, None]], axis=-1).T, cmap='viridis',
                yticklabels=['old new label', 'decision label'])
    plt.yticks(fontsize=22)
    plt.xticks(fontsize=10)

    plt.tight_layout()
    plt.show()

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


class FeatureVisualizer:
    """
    A class for visualizing and analyzing features extracted from EEG data.

    This class provides methods to visualize feature discriminancy, analyze correlations,
    and perform statistical tests on the features.

    Parameters
    ----------
    dataset : EEGDataSet
        The dataset containing EEG data and related information.
    feature_extractor : FeatureExtractor
        The feature extractor object used to extract features from EEG data.
    paths : object
        An object containing various file paths necessary for visualization and analysis.

    Attributes
    ----------
    save_path : str
        Path where the visualizations and analysis results will be saved.
    channel_names : list of str or None
        Names of the EEG channels.
    num_channel : int or None
        Number of EEG channels.
    dataset : EEGDataSet
        The EEG dataset.
    feature_extractor : FeatureExtractor
        The feature extractor object.
    paths : object
        Object containing paths for visualization and analysis.
    """

    def __init__(self, dataset, feature_extractor, paths):
        """
        Initializes the FeatureVisualizer with a dataset, feature extractor, and file paths.
        Args:
            dataset:
            feature_extractor:
            paths:
        """
        self.save_path = ''
        self.channel_names = None
        self.num_channel = None
        self.dataset = dataset
        self.feature_extractor = feature_extractor
        self.paths = paths
        logging.basicConfig(filename=paths.path_result + 'visualization_log.txt', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def visualize_and_analyze(self):
        """
        Visualize and analyze extracted features for each patient in the dataset.

        This method iterates over all patient features, performs analysis such as calculating
        p-values and correlations, and generates visualizations for the features.
        """
        for index, (patient_id, features) in enumerate(self.feature_extractor.all_patient_features.items()):
            print(f"Subject {index} from {len(self.feature_extractor.all_patient_features.keys())}: "
                  f"{patient_id} Analysis")
            logging.info(f"Subject {index} from {len(self.feature_extractor.all_patient_features.keys())}: "
                         f"{patient_id} Analysis")
            self.paths.create_paths_subject(patient_id)
            self.save_path = self.paths.path_subject_result[patient_id]
            decision = self.dataset.all_patient_data[patient_id].decision
            response_time = self.dataset.all_patient_data[patient_id].response_time
            rec_old = self.dataset.all_patient_data[patient_id].trial_type
            dgd_outputs = self.dataset.all_patient_data[patient_id].dgd_outputs
            trial_block = self.dataset.all_patient_data[patient_id].trial_block
            self.channel_names = self.dataset.all_patient_data[patient_id].channel_names
            self.num_channel = self.dataset.all_patient_data[patient_id].data.shape[1]

            # Analyze and visualize features
            feature_groups = _get_feature_groups(features)
            self._analyze_line_fit(features, feature_groups,
                                   continuous_label=dgd_outputs[:, 1],
                                   patient_id=patient_id,
                                   labels_desc='x_new')

            """
            self._analyze_line_fit(features, feature_groups,
                                   continuous_label=rec_old,
                                   patient_id=patient_id,
                                   labels_desc='rec_old')
            """

            """
            results = {'discriminancy_decision': self._analyze_p_value(features, feature_groups, labels=decision,
                                                                       patient_id=patient_id,
                                                                       labels_desc='decision'),
                       'discriminancy_rec_old': self._analyze_p_value(features, feature_groups, labels=rec_old,
                                                                      patient_id=patient_id,
                                                                      labels_desc='rec_old'),
                       'correlation_reaction_time': self._analyze_correlation(features, feature_groups,
                                                                              continuous_label=response_time,
                                                                              patient_id=patient_id,
                                                                              labels_desc='reaction_time'),
                       'correlation_x': self._analyze_correlation(features, feature_groups,
                                                                  continuous_label=dgd_outputs[:, 1],
                                                                  patient_id=patient_id,
                                                                  labels_desc='x_new',
                                                                  reverse_axis=True)}

            # Plot feature discriminancy for decision and rec_old labels
            plot_feature_discriminancy(pd.DataFrame({'X_new': dgd_outputs[:, 1]}),
                                       decision, patient_id, feature_name='X_new',
                                       label_type='decision', display=False, rotation=0,
                                       save_path=self.save_path)
            plot_feature_discriminancy(pd.DataFrame({'X_new': dgd_outputs[:, 1]}),
                                       rec_old, patient_id, feature_name='X_new',
                                       label_type='rec_old', display=False, rotation=0,
                                       save_path=self.save_path)
            """

            # Save results
            # self._save_results(patient_id, results)

    def _analyze_p_value(self, features, feature_groups, patient_id, labels, labels_desc='decision'):
        """
        Analyze the discriminancy of features using p-values.

        Parameters
        ----------
        features : dict
            Dictionary containing features for a patient.
        feature_groups : dict
            Dictionary categorizing feature names into different groups.
        patient_id : str
            Identifier for the patient.
        labels : numpy.ndarray
            Array of labels used for discriminancy analysis.
        labels_desc : str, optional
            Description of the labels (default is 'decision').

        Returns
        -------
        dict
            Dictionary containing p-values for each feature group.
        """

        features_p_value, list_best_channels = {}, {}

        # Loop through each feature group
        for feature_name in feature_groups.keys():
            # Calculate p-values for the current feature group
            features_p_value[feature_name] = self._get_p_value(features_data=features,
                                                               feature_groups=feature_groups,
                                                               feature_group_name=feature_name,
                                                               label=labels)

            # Save the p-values to a CSV file
            features_p_value[feature_name].to_csv(self.save_path + f'p_value_{labels_desc}_{feature_name}.csv')

            if feature_name in ['freq_features', 'time_features']:
                list_best_channels[feature_name] = list(np.argmin(features_p_value['freq_features'].values, axis=0))
            else:
                list_best_channels[feature_name] = None

            feature_df = self._get_feature_df(features, feature_groups=feature_groups, feature_group_name=feature_name,
                                              channels=list_best_channels[feature_name])

            # Plot feature discriminancy
            plot_feature_discriminancy(feature_df, labels, patient_id, feature_name=feature_name,
                                       label_type=labels_desc, save_path=self.save_path, rotation=45)

        return features_p_value

    def _analyze_correlation(self, features, feature_groups, patient_id, continuous_label, labels_desc='response time',
                             reverse_axis=False):
        """
        Analyze the correlation between features and a continuous label.

        Parameters
        ----------
        features : dict
            Dictionary containing features for a patient.
        feature_groups : dict
            Dictionary categorizing feature names into different groups.
        patient_id : str
            Identifier for the patient.
        continuous_label : numpy.ndarray
            Array of continuous labels used for correlation analysis.
        labels_desc : str, optional
            Description of the labels (default is 'response time').
        reverse_axis : bool, optional
            Whether to reverse the axis in the plot (default is False).

        Returns
        -------
        dict
            Dictionary containing correlation coefficients for each feature group.
        """
        features_corr, list_best_channels = {}, {}

        # Loop through each feature group
        for feature_name in feature_groups.keys():
            # Calculate correlation coefficients for the current feature group
            features_corr[feature_name] = self._get_correlation(features_data=features,
                                                                feature_groups=feature_groups,
                                                                feature_group_name=feature_name,
                                                                continuous_label=continuous_label)

            # Save the correlation coefficients to a CSV file
            features_corr[feature_name].to_csv(self.save_path + f'correlation_{labels_desc}_{feature_name}.csv')

            if feature_name in ['freq_features', 'time_features']:
                list_best_channels[feature_name] = list(np.argmin(features_corr['freq_features'].values, axis=0))
            else:
                list_best_channels[feature_name] = None

            feature_df = self._get_feature_df(features, feature_groups=feature_groups, feature_group_name=feature_name,
                                              channels=list_best_channels[feature_name])

            # Plot linear fit between features and continuous label
            plot_linear_fit(feature_df, continuous_label, patient_id, feature_name=feature_name, display=False,
                            save_path=self.save_path, label_type=labels_desc, reverse_axis=reverse_axis)

        return features_corr

    def _analyze_line_fit(self, features, feature_groups, patient_id, continuous_label, labels_desc='response time'):
        """
        Analyze the line fit on features and a continuous label.

        Parameters
        ----------
        features : dict
            Dictionary containing features for a patient.
        feature_groups : dict
            Dictionary categorizing feature names into different groups.
        patient_id : str
            Identifier for the patient.
        continuous_label : numpy.ndarray
            Array of continuous labels used for correlation analysis.
        labels_desc : str, optional
            Description of the labels (default is 'response time').
        reverse_axis : bool, optional
            Whether to reverse the axis in the plot (default is False).

        Returns
        -------
        dict
            Dictionary containing correlation coefficients for each feature group.
        """
        line_fit_slope, line_fit_bias, line_fit_p_value, list_best_channels = {}, {}, {}, {}
        # Loop through each feature group
        for feature_name in feature_groups.keys():
            # Calculate correlation coefficients for the current feature group
            line_fit_slope[feature_name], line_fit_p_value[feature_name], line_fit_bias[
                feature_name] = self._get_continuous_fit(
                features_data=features,
                feature_groups=feature_groups,
                feature_group_name=feature_name,
                continuous_label=continuous_label)

            # Save the correlation coefficients to a CSV file
            line_fit_p_value[feature_name].to_csv(self.save_path + f'line_fit_pvalue_{labels_desc}_{feature_name}.csv')
            line_fit_slope[feature_name].to_csv(self.save_path + f'line_fit_slope_{labels_desc}_{feature_name}.csv')

            if feature_name == 'coh_features':
                df = self._get_feature_df(features, feature_groups=feature_groups,
                                          feature_group_name=feature_name)
                features_list = [feature for feature in df.columns if feature != 'label_continuous']
                df['label_continuous'] = continuous_label
                fig, axs = plt.subplots(1, len(features_list), figsize=(len(features_list)*7, 5), dpi=100)
                for idx, feature in enumerate(features_list):
                    y = df[feature]

                    df_slope = line_fit_slope[feature_name]
                    df_pvalue = line_fit_p_value[feature_name]
                    df_bias = line_fit_bias[feature_name]

                    slope = df_slope[df_slope['feature'] == feature]['line_fit_slope'].values[0]
                    bias = df_bias[df_bias['feature'] == feature]['line_fit_bias'].values[0]
                    p_value = df_pvalue[df_pvalue['feature'] == feature]['line_fit_p_value'].values[0]

                    frq = feature.partition('_')[-1].partition('_')[-1].partition('_')[-1]
                    y_label = f'Total Coherency ({frq} Hz)'

                    axs[idx].scatter(continuous_label, y)
                    axs[idx].plot(continuous_label, continuous_label * slope + bias, color='red')
                    axs[idx].set_title(fr"$Feature = \beta_1 X + \beta_0$ pvalue= {np.round(p_value, 2)}", fontsize=20)
                    axs[idx].set_xlabel("Rec/New", fontsize=20)
                    axs[idx].set_ylabel(y_label, fontsize=20)
                    # Set the font size of axis ticks
                    plt.xticks(fontsize=14)
                    plt.yticks(fontsize=14)
                    plt.tight_layout()
                plt.savefig(self.save_path + f'Regress_{feature_name}_{labels_desc}_pvalue.png')

    def _get_feature_df(self, features_data, feature_groups, feature_group_name, channels=None):
        """
        Create a DataFrame from specified features and channels.

        Parameters
        ----------
        features_data : dict
            Dictionary containing features for a patient.
        feature_groups : dict
            Dictionary categorizing feature names into different groups.
        feature_group_name : str
            Name of the feature group to extract data from.
        channels : list of int or int, optional
            Specific channels to include in the DataFrame (default is None).

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the specified features and channels.
        """
        # Check if the feature group name is 'coh_features'
        if feature_group_name in ['coh_features']:
            # Create a DataFrame with feature names based on frequency and channel
            df = pd.DataFrame({feature + f'_{freq}': features_data[feature][:, idx]
                               for idx, freq in enumerate(features_data['coh_coh_freqs'])
                               for feature in feature_groups[feature_group_name]})
        # Check if the feature group name is 'coh_features_vec'
        elif feature_group_name in ['coh_features_vec']:
            # Create a DataFrame with feature names based on frequency, group index, and channel
            df = pd.DataFrame({feature + f'_{freq}_group{idx_group}': features_data[feature][:, idx, idx_group]
                               for idx, freq in enumerate(features_data['coh_coh_freqs'])
                               for idx_group in range(features_data['coh_vec_coh'].shape[-1])
                               for feature in feature_groups[feature_group_name]})
        # Check if the feature group name is 'freq_features' or 'time_features'
        elif feature_group_name in ['freq_features', 'time_features']:
            if isinstance(channels, list):
                # Create a DataFrame with feature names based on channel names
                df = pd.DataFrame(
                    {feature + ' ' + self.channel_names[ch]: features_data[feature][:, ch] for feature, ch in
                     zip(feature_groups[feature_group_name], channels)})
            elif isinstance(channels, int):
                # Create a DataFrame with feature names based on a single channel name
                df = pd.DataFrame({feature + ' ' + self.channel_names[channels]: features_data[feature][:, channels]
                                   for feature, ch in feature_groups[feature_group_name]})
            else:
                logging.info(f"Error: when the feature_group_name is {feature_group_name}, channels should "
                             f"be an int or a list")
                raise ValueError(
                    f" when the feature_group_name is {feature_group_name}, channels should be an int or a list")
        return df

    def _get_correlation(self, features_data, feature_groups, feature_group_name, continuous_label):
        """
        Calculate correlations for each channel with the continuous variable.

        Parameters
        ----------
        features_data : dict
            Dictionary of features.
        feature_groups : dict
            Dictionary categorizing feature names into different groups.
        feature_group_name : str
            Name of the feature group to analyze.
        continuous_label : numpy.ndarray
            Continuous variable for correlation.

        Returns
        -------
        pandas.DataFrame
            DataFrame of correlation coefficients.
        """
        # Check if the feature group name is 'freq_features' or 'time_features'
        if feature_group_name in ['freq_features', 'time_features']:
            feature_corr_list = []

            # Loop through each channel
            for ch in range(self.num_channel):
                # Create a DataFrame for the selected features in the current channel
                df = pd.DataFrame(
                    {feature: features_data[feature][:, ch] for feature in feature_groups[feature_group_name]})
                # Calculate correlations between the features and the continuous variable in the current channel
                feature_corr_list.append(calculate_correlations(df, label=continuous_label))

            # Convert the list of dictionaries to a dictionary of dictionaries
            feature_corr_dict = {self.channel_names[i]: feature_corr_list[i] for i in range(self.num_channel)}
            # Create a DataFrame from the dictionary
            corr_df = pd.DataFrame.from_dict(feature_corr_dict, orient='index')
        else:
            # If the feature group is not 'freq_features' or 'time_features'
            # Get a DataFrame containing the specified feature group
            data_df = self._get_feature_df(features_data, feature_groups=feature_groups,
                                           feature_group_name=feature_group_name)
            # Calculate correlations between the features and the continuous variable in the specified feature group
            corr_df = pd.DataFrame(list(calculate_correlations(data_df, label=continuous_label).items()),
                                   columns=['feature', 'correlation'])
        return corr_df

    def _get_continuous_fit(self, features_data, feature_groups, feature_group_name, continuous_label):
        """
        Calculate p-value for line fit for each channel with the continuous variable.

        Parameters
        ----------
        features_data : dict
            Dictionary of features.
        feature_groups : dict
            Dictionary categorizing feature names into different groups.
        feature_group_name : str
            Name of the feature group to analyze.
        continuous_label : numpy.ndarray
            Continuous variable for correlation.

        Returns
        -------
        pandas.DataFrame
            DataFrame of correlation coefficients.
        """
        # Check if the feature group name is 'freq_features' or 'time_features'
        if feature_group_name in ['freq_features', 'time_features']:
            feature_slope_list, feature_bias_list = [], []
            feature_slope_p_value_list = []

            # Loop through each channel
            for ch in range(self.num_channel):
                # Create a DataFrame for the selected features in the current channel
                df = pd.DataFrame(
                    {feature: features_data[feature][:, ch] for feature in feature_groups[feature_group_name]})
                # Calculate correlations between the features and the continuous variable in the current channel
                slope, p_value, bias = calculate_line_fit(df, label_continuous=continuous_label)
                feature_slope_list.append(slope)
                feature_slope_p_value_list.append(p_value)
                feature_bias_list.append(bias)

            # Convert the list of dictionaries to a dictionary of dictionaries
            feature_slope_dict = {self.channel_names[i]: feature_slope_list[i] for i in range(self.num_channel)}
            feature_bias_dict = {self.channel_names[i]: feature_bias_list[i] for i in range(self.num_channel)}
            feature_slope_p_value_dict = {self.channel_names[i]: feature_slope_p_value_list[i] for i in
                                          range(self.num_channel)}

            # Create a DataFrame from the dictionary
            line_fit_slope_df = pd.DataFrame.from_dict(feature_slope_dict, orient='index')
            line_fit_bias_df = pd.DataFrame.from_dict(feature_bias_dict, orient='index')
            line_fit_slope_p_value_df = pd.DataFrame.from_dict(feature_slope_p_value_dict, orient='index')
        else:
            # If the feature group is not 'freq_features' or 'time_features'
            # Get a DataFrame containing the specified feature group
            data_df = self._get_feature_df(features_data, feature_groups=feature_groups,
                                           feature_group_name=feature_group_name)
            # Calculate correlations between the features and the continuous variable in the specified feature group
            slope, p_value, bias = calculate_line_fit(data_df, label_continuous=continuous_label)

            line_fit_slope_df = pd.DataFrame(list(slope.items()), columns=['feature', 'line_fit_slope'])
            line_fit_bias_df = pd.DataFrame(list(bias.items()), columns=['feature', 'line_fit_bias'])
            line_fit_slope_p_value_df = pd.DataFrame(list(p_value.items()), columns=['feature', 'line_fit_p_value'])

        return line_fit_slope_df, line_fit_slope_p_value_df, line_fit_bias_df

    def _get_p_value(self, features_data, feature_groups, feature_group_name, label):
        """
        Calculate p-values for each channel with the given label.

        Parameters
        ----------
        features_data : dict
            Dictionary of features.
        feature_groups : dict
            Dictionary categorizing feature names into different groups.
        feature_group_name : str
            Name of the feature group to analyze.
        label : numpy.ndarray
            Labels used for calculating p-values.

        Returns
        -------
        pandas.DataFrame
            DataFrame of p-values.
        """
        # Check if the feature group name is 'freq_features' or 'time_features'
        if feature_group_name in ['freq_features', 'time_features']:
            feature_p_value_list = []

            # Loop through each channel
            for ch in range(self.num_channel):
                # Create a DataFrame for the selected features in the current channel
                df = pd.DataFrame(
                    {feature: features_data[feature][:, ch] for feature in feature_groups[feature_group_name]})
                # Calculate p-values for the features in the current channel
                feature_p_value_list.append(calculate_p_values(df, label=label))

            # Convert the list of dictionaries to a dictionary of dictionaries
            feature_p_value_dict = {self.channel_names[i]: feature_p_value_list[i] for i in range(self.num_channel)}
            # Create a DataFrame from the dictionary
            p_value_df = pd.DataFrame.from_dict(feature_p_value_dict, orient='index')
        else:
            # If the feature group is not 'freq_features' or 'time_features'
            # Get a DataFrame containing the specified feature group
            data_df = self._get_feature_df(features_data, feature_groups=feature_groups,
                                           feature_group_name=feature_group_name)
            # Calculate p-values for the features in the specified feature group
            p_value_df = pd.DataFrame(list(calculate_p_values(data_df, label=label).items()),
                                      columns=['feature', 'p_value'])

        return p_value_df
