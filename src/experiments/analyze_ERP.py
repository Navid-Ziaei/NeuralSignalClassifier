import matplotlib.pyplot as plt
import pandas as pd
import torch
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from src.settings import Paths, Settings
from src.data_loader import VerbMemEEGDataLoader, PilotEEGDataLoader, CLEARDataLoader, TorchEEGDataset
from src.data_preprocess import DataPreprocessor
from src.visualization.EEG_vissualizer import visualize_erp, visualize_block_ERP, visualize_feature_box_plot, \
    visualize_block_features, visualize_p_value_barplots
from src.data_loader import EEGDataSet
from src.feature_extraction import FeatureExtractor
import numpy as np
import seaborn as sns
from scipy.stats import pointbiserialr
from torcheeg.models import EEGNet
from torcheeg.trainers import ClassifierTrainer
import pytorch_lightning as pl
from torcheeg.model_selection import KFoldGroupbyTrial

# Load settings from settings.json
settings = Settings()  # Initialize settings object
settings.load_settings()  # Load settings from a JSON file
use_groups = False

# Set up paths for data
paths = Paths(settings)  # Initialize paths object with loaded settings
paths.load_device_paths()  # Load device-specific paths
paths.create_paths()  # Create any necessary file paths

# Load EEG dataset using configured settings and paths
if settings.dataset == 'pilot01':
    dataset = PilotEEGDataLoader(paths=paths, settings=settings)  # Initialize EEG dataset loader
elif settings.dataset == 'verbmem':
    dataset = VerbMemEEGDataLoader(paths=paths, settings=settings)  # Initialize EEG dataset loader
elif settings.dataset == 'clear':
    dataset = CLEARDataLoader(paths=paths, settings=settings)

dataset.load_data(patient_ids=settings.patient)  # Load EEG data for specified patients

# Preprocess the loaded dataset
data_preprocessor = DataPreprocessor(paths=paths, settings=settings)  # Initialize data preprocessor
dataset = data_preprocessor.preprocess(dataset)  # Apply preprocessing steps to the dataset


"""
for idx in range(len(list(dataset.all_patient_data.keys()))):
    single_patient_data = dataset.all_patient_data[list(dataset.all_patient_data.keys())[idx]]
    visualize_erp(patient_data=single_patient_data, channel_idx=45,
                  label=single_patient_data.labels['is_experienced'],
                  label_names=['Not Experienced', 'Experienced'],
                  time_lim=[-1000, 1000])
    visualize_erp(patient_data=single_patient_data, channel_idx=45,
                  label=single_patient_data.labels['go_nogo'], label_names=['NoGo', 'Go'])
"""

single_patient_data = dataset.all_patient_data[list(dataset.all_patient_data.keys())[0]]

dataset_train = TorchEEGDataset(data=single_patient_data.data[:-80], labels=single_patient_data.labels['is_experienced'][:-80])
dataset_test = TorchEEGDataset(data=single_patient_data.data[-80:], labels=single_patient_data.labels['is_experienced'][-80:])

train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset_test, batch_size=64, shuffle=False)

model = EEGNet(num_classes=2, num_electrodes=128)
y = model(torch.Tensor(single_patient_data.data[0:4].copy()))
trainer = ClassifierTrainer(model=model,
                            num_classes=2,
                            lr=1e-4,
                            weight_decay=1e-4,
                            accelerator="gpu")
trainer.fit(train_loader,
            val_loader,
            max_epochs=50,
            enable_progress_bar=True,
            enable_model_summary=True,
            limit_val_batches=0.0)
score = trainer.test(val_loader,
                     enable_progress_bar=True,
                     enable_model_summary=True)[0]
print(f'test accuracy: {score["test_accuracy"]:.4f}')


# a_channels = [idx for idx,channel_name in enumerate(single_patient_data.channel_names) if 'A' in channel_name]
all_channels = single_patient_data.channel_names

if settings.dataset == 'pilot01' and use_groups == True:
    channel_groups = {
        "CF": ['C15', 'C16', 'C17', 'C18', 'C19', 'C28', 'C29'],
        "LAL": ['C30', 'C31', 'D5', 'D6', 'D7', 'D8', 'D9'],
        "LAM": ['D2', 'D3', 'D4', 'D12', 'D13', 'C24', 'C25'],
        "RAM": ['C2', 'C3', 'C4', 'C11', 'C12', 'B31', 'B32'],
        "RAL": ['C5', 'C6', 'C7', 'C8', 'C9', 'B27', 'B28'],
        "LOT": ['A10', 'A11', 'D26', 'D27', 'D30', 'D31', 'D32'],
        "LPM": ['A5', 'A6', 'A7', 'A18', 'D16', 'D17', 'D28'],
        "RPM": ['A31', 'A32', 'B2', 'B3', 'B4', 'B18', 'B19'],
        "ROT": ['B7', 'B8', 'B10', 'B11', 'B12', 'B16', 'B17'],
        "CO": ['A14', 'A15', 'A22', 'A23', 'A24', 'A27', 'A28']}

    channel_groups_indices = {group: [all_channels.index(channel) for channel in channels] for group, channels in
                              channel_groups.items()}

    new_channel_names, new_data = [], []
    for group_name, channel_indices in channel_groups_indices.items():
        print(f"group {group_name}: {[all_channels[ch - 1] for ch in channel_indices]}")
        new_data.append(np.mean(single_patient_data.data[:, np.array(channel_indices), :], axis=1, keepdims=True))
        new_channel_names.append(group_name)
    new_data = np.concatenate(new_data, axis=1)
    single_patient_data.data = new_data
    single_patient_data.channel_names = new_channel_names
    all_channels = new_channel_names

"""fig, axs = plt.subplots(len(all_channels), 5, figsize=(50, 5 * len(all_channels)))

for idx, channel_idx in enumerate(all_channels):
    fig, _ = visualize_block_ERP(single_patient_data, stim='w', channel_idx=idx, axs=axs[idx, :], fig=fig)
plt.tight_layout()
fig.savefig("channel_iEEG.svg")
plt.show()"""


def get_time_idx(time, start_time, end_time):
    start_index = np.argmin(np.abs(time - start_time))
    end_index = np.argmin(np.abs(time - end_time))
    return start_index, end_index


time_features = {}
start_index, end_index = get_time_idx(single_patient_data.time_ms, 150, 250)
time_features['n200'] = np.mean(single_patient_data.data[:, :, start_index:end_index], axis=-1)

start_index, end_index = get_time_idx(single_patient_data.time_ms, 250, 500)
time_features['p300'] = np.mean(single_patient_data.data[:, :, start_index:end_index], axis=-1)

start_index, end_index = get_time_idx(single_patient_data.time_ms, 500, 750)
time_features['post_300'] = np.mean(single_patient_data.data[:, :, start_index:end_index], axis=-1)

# #########################################  Analyze p value #########################################
visualize_p_value_barplots(single_patient_data, time_features, channel_idx=1, show_plot=True)


visualize_block_features(single_patient_data, time_features, channel_idx=1, stim='w', show_plot=True, fig=None,
                         axs=None)
# #########################################  Visualize feature box plot #########################################


fig, axs = plt.subplots(len(all_channels), 5, figsize=(50, 5 * len(all_channels)))

for idx, channel_idx in enumerate(all_channels):
    visualize_block_features(single_patient_data, time_features, channel_idx=idx, stim='w', fig=fig,
                             axs=axs[idx, :])
plt.tight_layout()
fig.savefig("features_ieeg.svg")
plt.show()

"""
channel_name = dataset.all_patient_data['101_toi1gng_2023-11-20_14-09-06_1'].channel_names
time = dataset.all_patient_data['101_toi1gng_2023-11-20_14-09-06_1'].time_ms
idx = np.array(dataset.all_patient_data['101_toi1gng_2023-11-20_14-09-06_1'].labels['block_number']) == 0
data = dataset.all_patient_data['101_toi1gng_2023-11-20_14-09-06_1'].data[idx, :, :]
label = np.array(dataset.all_patient_data['101_toi1gng_2023-11-20_14-09-06_1'].labels['is_experienced'])[idx]
for i in range(len(label)):
    pd.DataFrame(data[i], index=channel_name, columns=time).to_csv(f'trial{i}_exp_{label[i]}.csv')
"""
