import matplotlib.pyplot as plt
import pandas as pd
from src.settings import Paths, Settings
from src.data_loader import VerbMemEEGDataLoader, PilotEEGDataLoader, CLEARDataLoader
from src.data_preprocess import DataPreprocessor
from src.visualization.EEG_vissualizer import visualize_erp, visualize_block_ERP, visualize_feature_box_plot, \
    visualize_block_features
from src.data_loader import EEGDataSet
from src.feature_extraction import FeatureExtractor
import numpy as np
import seaborn as sns
import copy
from src.experiments.utils.train_gplvm_utils import *
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.svm import SVC
from scipy import signal
from scipy.fft import fft
from matplotlib.backends.backend_pdf import PdfPages
import os
from pylatex import Document, Section, Subsection, Figure, NoEscape, Math, Matrix, Tabular, Itemize

# Load settings from settings.json
settings = Settings()  # Initialize settings object
settings.load_settings()  # Load settings from a JSON file

# Set up paths for data
paths = Paths(settings)  # Initialize paths object with loaded settings
paths.load_device_paths()  # Load device-specific paths
paths.create_paths()  # Create any necessary file paths

dataset = PilotEEGDataLoader(paths=paths, settings=settings)
dataset.load_data(patient_ids=settings.patient)

data_preprocessor = DataPreprocessor(paths=paths, settings=settings)  # Initialize data preprocessor
dataset = data_preprocessor.preprocess(dataset)  # Apply preprocessing steps to the dataset

single_patient_data = copy.copy(dataset.all_patient_data[list(dataset.all_patient_data.keys())[0]])

doc = Document('EEG Analysis Report')
doc.preamble.append(NoEscape(r'\title{Subject' + single_patient_data.file_name.split('_')[0] + '}'))
doc.append(NoEscape(r'\maketitle'))

with doc.create(Section('Introduction')):
    doc.append('This report presents the results of the Fast Fourier Transform (FFT) analysis conducted on EEG data. '
               'The analysis aims to explore the spectral characteristics of EEG signals across various channel groups '
               'to understand the frequency distribution and its implications on the observed neurological phenomena.')

print(f"Selected patient is {single_patient_data.file_name.split('_')[0]}")
channels = single_patient_data.channel_names

all_channels = single_patient_data.channel_names
channel_groups = {
    "CF": ['C15', 'C16', 'C17', 'C18', 'C19', 'C28', 'C29'],
    "LAL": ['B30', 'B31', 'D5', 'D6', 'D7', 'D8', 'D9'],
    "LAM": ['D2', 'D3', 'D4', 'D12', 'D13', 'C24', 'C25'],
    "RAM": ['C2', 'C3', 'C4', 'C11', 'C12', 'B31', 'B32'],
    "RAL": ['C5', 'C6', 'C7', 'C8', 'C9', 'B27', 'B28'],
    "LOT": ['A10', 'A11', 'D26', 'D27', 'D30', 'D31', 'D32'],
    "LPM": ['A5', 'A6', 'A7', 'A18', 'D16', 'D17', 'D28'],
    "RPM": ['A31', 'A32', 'B2', 'B3', 'B4', 'B18', 'B19'],
    "ROT": ['B7', 'B8', 'B10', 'B11', 'B12', 'B16', 'B17'],
    "CO": ['A14', 'A15', 'A22', 'A23', 'A24', 'A27', 'A28']}

channel_groups_indices = {group: [all_channels.index(channel) for channel in channels_name] for group, channels_name in
                          channel_groups.items()}
for group_name, indices in channel_groups_indices.items():
    print(f"{group_name}: {indices}")

with doc.create(Section('Data Overview')):
    doc.append('The EEG data analyzed in this report is structured into several channel groups, each representing a '
               'distinct region of the electrode array used during the recording. The channel groups are categorized '
               'as follows:')
    with doc.create(Itemize()) as itemize:
        itemize.add_item(NoEscape(r'\textbf{CF:} Central Frontal regions'))
        itemize.add_item(NoEscape(r'\textbf{LAL:} Left Anterior Lateral regions'))
        itemize.add_item(NoEscape(r'\textbf{LAM:} Left Anterior Medial regions'))
        itemize.add_item(NoEscape(r'\textbf{RAM:} Right Anterior Medial regions'))
        itemize.add_item(NoEscape(r'\textbf{RAL:} Right Anterior Lateral regions'))
        itemize.add_item(NoEscape(r'\textbf{LOT:} Left Occipito-Temporal regions'))
        itemize.add_item(NoEscape(r'\textbf{LPM:} Left Posterior Medial regions'))
        itemize.add_item(NoEscape(r'\textbf{RPM:} Right Posterior Medial regions'))
        itemize.add_item(NoEscape(r'\textbf{ROT:} Right Occipito-Temporal regions'))
        itemize.add_item(NoEscape(r'\textbf{CO:} Central Occipital regions'))

with doc.create(Section('Methodology')):
    doc.append('The Fast Fourier Transform (FFT) is employed to transform the time-domain EEG signals into their '
               'frequency-domain representations. This transformation allows for the analysis of the signal’s '
               'frequency components, facilitating a deeper understanding of the underlying brain activities. '
               'The FFT was computed using a Hanning window to mitigate the spectral leakage.')

    with doc.create(Subsection('FFT Computation')):
        doc.append('The FFT computation process is as follows:')
        with doc.create(Itemize()) as itemize:
            itemize.add_item(NoEscape(r'\textbf{Data Preparation:} The EEG signal data for each channel is extracted, '
                                      'and a Hanning window is applied to each segment of the data. This windowing '
                                      'process helps in reducing the spectral leakage by tapering the signal edges.'))
            itemize.add_item(NoEscape(r'\textbf{Hanning Window:} The Hanning window used is defined mathematically as: '
                                      r'$w[n] = 0.5 - 0.5 \cos\left(\frac{2\pi n}{N-1}\right)$, where $N$ is the number of points in the window, '
                                      r'and $n$ is the sample number. This window is applied to the signal to emphasize the middle portions of the time segment.'))
            itemize.add_item(NoEscape(r'\textbf{FFT Execution:} The FFT is computed over the windowed signal. '
                                      'The number of points for the FFT, denoted as $n_{FFT}$, is set to 512. '
                                      'This setting determines the frequency resolution of the FFT output by dividing the sampling rate by $n_{FFT}$.'))
            itemize.add_item(
                NoEscape(r'\textbf{Normalization and Conversion:} The raw FFT output is complex, representing '
                         'both amplitude and phase. The magnitude is obtained by normalizing the absolute values '
                         'of the FFT results by $n_{FFT}$. This normalization ensures that the magnitude does '
                         'not depend on the length of the input signal.'))
            itemize.add_item(NoEscape(r'\textbf{Spectrum Adjustment:} To convert the two-sided spectrum into a '
                                      'single-sided spectrum, all frequency components above Nyquist frequency are discarded, '
                                      'and the magnitudes of the non-DC components are doubled (except for the Nyquist component, if present).'))
            itemize.add_item(
                NoEscape(r'\textbf{Frequency Vector Creation:} A frequency vector is created that corresponds to the '
                         'magnitudes in the single-sided spectrum. This vector spans from 0 Hz up to the Nyquist frequency, '
                         'allowing the magnitudes to be plotted against their respective frequencies.'))

        doc.append('This methodology is applied consistently across all channels and trials to ensure comprehensive '
                   'and uniform spectral analysis.')


def compute_fft(data, fs):
    """Compute the FFT with a Hanning window."""
    data_len = data.shape[-1]
    n_fft = 512
    hann_window = np.hanning(data_len)
    Y = fft(hann_window[np.newaxis] * data, n=n_fft)

    # Calculate the two-sided spectrum P2 then the single-sided spectrum P1
    P2 = np.abs(Y / n_fft)  # Normalize by the number of FFT points not the data length
    P1 = P2[:, :n_fft // 2 + 1]
    P1[:, 1:-1] = 2 * P1[:, 1:-1]  # Only double non-unique parts

    # Create frequency array for the output spectrum
    f = fs * np.arange((n_fft // 2) + 1) / n_fft

    valid_indices = (f <= 40)

    return f[valid_indices], P1[:, valid_indices]


label_df = pd.DataFrame(single_patient_data.labels)
trial_index_exp = (label_df['is_correct'] == True) & (label_df['stim'] != 'ctl') & (label_df['is_experienced'] == True)
trial_index_noexp = (label_df['is_correct'] == True) & (label_df['stim'] != 'ctl') & (
        label_df['is_experienced'] == False)

# Define the frequency range and sampling rate
freq_range = (0, 50)  # in Hz
fs = 250  # Sampling rate in Hz

fft_signals = []
time_start = 250
end_time = 750
start_index = np.argmin(np.abs(single_patient_data.time_ms - time_start))
end_index = np.argmin(np.abs(single_patient_data.time_ms - end_time))

with doc.create(Section('FFT Analysis in Groups')):
    doc.append(NoEscape('This section details the Fourier Transform results organized by specific channel groups, '
               'which represent distinct regions of the EEG cap setup. Each group\'s data is processed to '
               'extract frequency components via the Fast Fourier Transform (FFT), crucial for identifying '
               'dominant brainwave frequencies associated with various mental states and activities.'))

    for group_name, channel_indices in channel_groups_indices.items():
        doc.append(NoEscape(r'\subsection{%s}' % group_name))
        doc.append('The FFT analysis for the %s group involves channels %s. The figures below include '
                   'individual channel FFTs and their aggregated mean with a confidence interval to '
                   'highlight the variability across channels within this group.' % (
                       group_name, ", ".join([all_channels[ch] for ch in channel_indices])))

        # Process signal group for FFT
        signal_group = single_patient_data.data[:, channel_indices, start_index:end_index]
        fft_group = []
        for ch_in_gp in range(signal_group.shape[1]):  # Transpose to iterate over channels
            f, P1 = compute_fft(signal_group[:, ch_in_gp, :], fs)
            fft_group.append(P1)
        fft_group = np.array(fft_group)
        mean_fft = np.mean(fft_group, axis=0)
        std_fft = np.std(fft_group, axis=0)

        # Plotting each channel FFT
        trial_idx = 0
        plt.figure(figsize=(14, 7))
        for i, fft_data in enumerate(fft_group):
            plt.plot(f, 20 * np.log10(fft_data[trial_idx]), label=f'Channel {channel_indices[i] + 1}')
        plt.title(f'FFT of Each Channel in {group_name} for trial {trial_idx + 1}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.legend()
        plt.grid(True)
        group_fft_filename = f'results/Spectral_Report/FFT_each_channel_{group_name}.png'
        plt.savefig(group_fft_filename)
        plt.close()

        with doc.create(Figure(position='h!')) as group_plot_fig:
            group_plot_fig.add_image(group_fft_filename, width='400px')
            group_plot_fig.add_caption(f'FFT results for each channel in the {group_name} group for {trial_idx + 1}'
                                       f'-th trial, showcasing the '
                                       f'spectral characteristics unique to this segmentation of electrodes.')

        # Plotting the mean FFT with confidence interval
        plt.figure(figsize=(10, 5))
        plt.plot(f, 20 * np.log10(mean_fft[trial_idx]), label='Mean FFT', color='b')
        plt.fill_between(f, 20 * np.log10(mean_fft[trial_idx] - std_fft[trial_idx]),
                         20 * np.log10(mean_fft[trial_idx] + std_fft[trial_idx]), color='b',
                         alpha=0.3, label='1 std dev')
        plt.title(f'Mean FFT with Confidence Interval for {group_name} (trial {trial_idx + 1})')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.legend()
        plt.grid(True)
        mean_fft_filename = f'results/Spectral_Report/Mean_FFT_{group_name}.png'
        plt.savefig(mean_fft_filename)
        plt.close()

        with doc.create(Figure(position='h!')) as mean_plot_fig:
            mean_plot_fig.add_image(mean_fft_filename, width='400px')
            mean_plot_fig.add_caption('Mean FFT with confidence intervals for the %s group. This plot provides '
                                      'a clear visual representation of the central tendencies and variabilities '
                                      'within the frequency domain.' % group_name)

        # Create heatmaps for experienced and not experienced trials
        exp_mean_fft = mean_fft[trial_index_exp.values]
        noexp_mean_fft = mean_fft[trial_index_noexp.values]
        fig, axs = plt.subplots(2, 1, figsize=(15, 10))
        exp_heatmap = axs[0].imshow(20 * np.log10(exp_mean_fft), aspect='auto',
                                    extent=[f[0], f[-1], 1, exp_mean_fft.shape[0]])
        noexp_heatmap = axs[1].imshow(20 * np.log10(noexp_mean_fft), aspect='auto',
                                      extent=[f[0], f[-1], 1, noexp_mean_fft.shape[0]])
        axs[0].set_title('Experienced Trials')
        axs[1].set_title('Non-Experienced Trials')
        axs[0].set_xlabel('Frequency (Hz)')
        axs[0].set_ylabel('Trial Index')
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylabel('Trial Index')
        plt.colorbar(exp_heatmap, ax=axs[0], orientation='vertical')
        plt.colorbar(noexp_heatmap, ax=axs[1], orientation='vertical')
        plt.tight_layout()
        heatmap_filename = f'results/Spectral_Report/Heatmaps_{group_name}.png'
        plt.savefig(heatmap_filename)
        plt.close()

        with doc.create(Figure(position='h!')) as heatmap_fig:
            heatmap_fig.add_image(heatmap_filename, width='400px')
            heatmap_fig.add_caption('Heatmaps of FFT magnitudes across trials differentiated by experience '
                                    'levels for the %s group. These visualizations help in comparing spectral '
                                    'dynamics between different cognitive engagement levels.' % group_name)
        fft_signals.append(mean_fft)

with doc.create(Section('Visualization of FFT Signals Across Trials')):
    doc.append('This analysis section delves into the FFT signals processed across different trials, '
               'segmented into experienced and non-experienced groups. The primary objective is to compare '
               'these groups based on their mean spectral magnitudes and understand the differential '
               'brain activity characterized by varying frequency bands (Delta, Theta, Alpha, Beta, Gamma). '
               'This section provides visualizations in the form of heatmaps and detailed band-specific plots to '
               'aid in the comparative analysis.')

    doc.append('Initially, FFT signals are restructured for analysis:')
    doc.append(NoEscape(r'\begin{itemize}'))
    doc.append(NoEscape(
        r'\item FFT signals are expanded along a new axis to facilitate separate analyses by trial and channel group.'))
    doc.append(NoEscape(
        r'\item Signals are then concatenated across this new axis to form a coherent dataset for group comparisons.'))
    doc.append(NoEscape(r'\end{itemize}'))
    fft_signals = [fft_signal[:, np.newaxis, :] for fft_signal in fft_signals]
    fft_signals = np.concatenate(fft_signals, axis=1)  # Assuming this shape is (trials, channels, frequency_bins)

    mean_over_trials_exp = np.mean(fft_signals[trial_index_exp.values], axis=0)
    std_over_trials_exp = np.std(fft_signals[trial_index_exp.values], axis=0)

    mean_over_trials_noexp = np.mean(fft_signals[trial_index_noexp.values], axis=0)
    std_over_trials_noexp = np.std(fft_signals[trial_index_noexp.values], axis=0)

    fig, axs = plt.subplots(3, 1, figsize=(20, 8))
    im = axs[0].imshow(20 * np.log10(mean_over_trials_exp), aspect='auto', extent=[f[0], f[-1], 1, len(channel_groups)])
    fig.colorbar(im, ax=axs[0], orientation='vertical', label='Magnitude (Linear scale)')

    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('Channel Group')
    axs[0].set_title('Heatmap of Difference of Mean FFT for Experienced')
    axs[0].set_yticks(ticks=np.arange(1, len(channel_groups) + 1), labels=list(channel_groups.keys()))

    im = axs[1].imshow(20 * np.log10(mean_over_trials_noexp), aspect='auto',
                       extent=[f[0], f[-1], 1, len(channel_groups)])
    fig.colorbar(im, ax=axs[1], orientation='vertical', label='Magnitude (Linear scale)')

    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Channel Group')
    axs[1].set_title('Heatmap of Difference of Mean FFT for Not Experienced')
    axs[1].set_yticks(ticks=np.arange(1, len(channel_groups) + 1), labels=list(channel_groups.keys()))

    im = axs[2].imshow(20 * np.log10(np.abs(mean_over_trials_exp - mean_over_trials_noexp)), aspect='auto',
                       extent=[f[0], f[-1], 1, len(channel_groups)])
    fig.colorbar(im, ax=axs[2], orientation='vertical', label='Magnitude (Linear scale)')

    axs[2].set_xlabel('Frequency (Hz)')
    axs[2].set_ylabel('Channel Group')
    axs[2].set_title('Heatmap of Difference of Mean FFT between Experienced/ Not Experienced')
    axs[2].set_yticks(ticks=np.arange(1, len(channel_groups) + 1), labels=list(channel_groups.keys()))

    plt.tight_layout()
    heatmap_expnoexp_filename = 'results/Spectral_Report/Heatmap_expnoexp.png'
    plt.savefig(heatmap_filename)
    plt.close()

    with doc.create(Figure(position='h!')) as heatmap_exp_fig:
        heatmap_exp_fig.add_image(heatmap_expnoexp_filename, width='400px')
        heatmap_exp_fig.add_caption(
            'Heatmap of mean FFT magnitudes for the Experienced ns non experienced group across all channel groups and trials.')

freq_bands = {
    'delta': (1, 3),
    'theta': (5, 7),
    'alpha': (9, 12),
    'beta': (14, 30),
    'gamma': (30, 45)
}

frequency_features = {}
for band_f in freq_bands.keys():
    f_start, f_end = freq_bands[band_f]
    band_start_idx = np.argmin(np.abs(f - f_start))
    band_end_idx = np.argmin(np.abs(f - f_end))
    frequency_features[band_f] = np.mean(fft_signals[:, :, band_start_idx:band_end_idx], axis=-1)
    frequency_features[band_f] = frequency_features[band_f] / np.quantile(np.abs(frequency_features[band_f]), 0.995,
                                                                          axis=0, keepdims=True)

channel_group_name = list(channel_groups.keys())
fig, axs = plt.subplots(len(channel_group_name), 5, figsize=(50, 5 * len(channel_group_name)))

for idx, channel_idx in enumerate(channel_group_name):
    visualize_block_features(single_patient_data, frequency_features, channel_idx=idx, stim='i', fig=fig,
                             axs=axs[idx, :])
plt.tight_layout()
fig.savefig("Spectral_Report/spectral_features_boxplot.png")
plt.close()
with doc.create(Section('Frequency Feature Analysis')):
    doc.append('This section focuses on specific frequency bands that are crucial for interpreting EEG signals. '
               'Each band corresponds to different brain activities or states:')
    with doc.create(Itemize()) as itemize:
        itemize.add_item('Delta (1-3 Hz): Associated with sleep and restorative states.')
        itemize.add_item('Theta (4-7 Hz): Related to creativity and emotional expression.')
        itemize.add_item('Alpha (8-12 Hz): Linked to physical and mental relaxation.')
        itemize.add_item('Beta (13-30 Hz): Involved in active, engaged thinking and problem solving.')
        itemize.add_item('Gamma (30-45 Hz): Important for learning, memory formation, and data processing.')

    # Box plot for spectral features across bands
    spectral_features_plot = 'Spectral_Features_Bands.png'
    with doc.create(Figure(position='h!')) as spectral_fig:
        spectral_fig.add_image(spectral_features_plot, width='120mm')
        spectral_fig.add_caption('Box plots of spectral features across Delta, Theta, Alpha, Beta, and Gamma bands.')

import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier  # Example classifier
import numpy as np

# Assuming frequency_features and channel_group_name are defined as per previous sections of the code
training_features = {}
for key in frequency_features.keys():
    for ch_idx, ch in enumerate(channel_group_name):
        training_features[key + ' ' + ch] = frequency_features[key][:, ch_idx]
training_features_df = pd.DataFrame(training_features)


def train_kfold(X, y, model):
    results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X):
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        precision = precision_score(y_test, predictions, average='macro')
        recall = recall_score(y_test, predictions, average='macro')
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='macro')

        results.append({
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1_score': f1
        })

    return pd.DataFrame(results)


# Create an empty DataFrame to store summaries
# Summarizing results in the document
with doc.create(Section('Classification Results')):
    doc.append('This section summarizes the classification results obtained from the analysis. '
               'We employed a machine learning model to predict categories based on the engineered features. '
               'The results from K-fold cross-validation are tabulated below.')
    summary_df = pd.DataFrame()
    stim = 'w'
    block_idx = np.unique(single_patient_data.labels['block_number'])
    label_df = pd.DataFrame(single_patient_data.labels)
    blocks = {idx: list(np.unique(label_df[label_df['block_number'] == idx]['block_type'])) for idx in block_idx}
    blocks_with_stim = [idx for idx, vals in blocks.items() if any(stim in val for val in vals)]

    for i in range(len(blocks_with_stim)):
        print(f"============ Block {blocks_with_stim[i]} {blocks[blocks_with_stim[i]]} ===============")
        block_index = (label_df['block_number'] == blocks_with_stim[i]) & (label_df['is_correct'] == True) & (
                label_df['stim'] != 'ctl')
        labels = {key: label_df[key][block_index].to_list() for key in label_df.columns.to_list()}
        single_channel_feature_df = training_features_df[block_index.values].copy()
        fs = single_patient_data.fs
        time_ms = single_patient_data.time_ms
        channel_names = single_patient_data.channel_names
        file_name = single_patient_data.file_name
        single_channel_feature_df.head()
        model = xgb.XGBClassifier(objective="multi:softmax", num_class=2)
        # model = linear_model.Lasso(alpha=1)
        # model = SVC()

        # Train and get results
        result_df = train_kfold(X=single_channel_feature_df, y=np.array(labels['is_experienced']), model=model)
        result_summary = result_df.mean().to_frame().T.describe().loc[['mean', 'std']]

        with doc.create(Subsection(f'Classification Metrics for {blocks[blocks_with_stim[i]]}')):
            with doc.create(Tabular('lcccc')) as table:
                table.add_hline()
                table.add_row(('Metric', 'Precision', 'Recall', 'Accuracy', 'F1 Score'))
                table.add_hline()
                table.add_row(('Mean', f"{result_summary['precision']['mean']:.2f}",
                               f"{result_summary['recall']['mean']:.2f}",
                               f"{result_summary['accuracy']['mean']:.2f}",
                               f"{result_summary['f1_score']['mean']:.2f}"))
                table.add_row(('Standard Deviation', f"{result_summary['precision']['std']:.2f}",
                               f"{result_summary['recall']['std']:.2f}",
                               f"{result_summary['accuracy']['std']:.2f}",
                               f"{result_summary['f1_score']['std']:.2f}"))
                table.add_hline()
        result_df_mean = result_df.mean().to_frame().T
        result_df_mean.index = [f'Block {blocks_with_stim[i]}']
        summary_df = pd.concat([summary_df, result_df_mean])

    # Overall results
    overall_results = train_kfold(training_features_df, label_df['is_experienced'], model)
    overall_results_mean = overall_results.mean().to_frame().T
    result_summary = overall_results.mean().to_frame().T.describe().loc[['mean', 'std']]

    with doc.create(Subsection('Summary of Classification Metrics')):
        with doc.create(Tabular('lcccc')) as table:
            table.add_hline()
            table.add_row(('Metric', 'Precision', 'Recall', 'Accuracy', 'F1 Score'))
            table.add_hline()
            table.add_row(('Mean', f"{result_summary['precision']['mean']:.2f}",
                           f"{result_summary['recall']['mean']:.2f}",
                           f"{result_summary['accuracy']['mean']:.2f}",
                           f"{result_summary['f1_score']['mean']:.2f}"))
            table.add_row(('Standard Deviation', f"{result_summary['precision']['std']:.2f}",
                           f"{result_summary['recall']['std']:.2f}",
                           f"{result_summary['accuracy']['std']:.2f}",
                           f"{result_summary['f1_score']['std']:.2f}"))
            table.add_hline()
    overall_results_mean.index = ['Overall']
    summary_df = pd.concat([summary_df, overall_results_mean])

    # Print summary DataFrame
    print(summary_df)

# doc.generate_pdf('detailed_classification_report', clean_tex=True)
doc.generate_tex(f"detailed_report_tex_{single_patient_data.file_name.split('_')[0]}")
