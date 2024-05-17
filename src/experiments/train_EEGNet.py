from src.settings import Paths, Settings
from src.data_loader import VerbMemEEGDataLoader, PilotEEGDataLoader
from src.data_preprocess import DataPreprocessor
from src.model.EEGNet import EEGNet

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
from sklearn.model_selection import train_test_split
from src.experiments.utils import SimpleEEGDataset
import matplotlib.pyplot as plt
import pandas as pd
from torcheeg.models import EEGNet
from torcheeg.model_selection import KFold

# Load settings from settings.json
settings = Settings()  # Initialize settings object
settings.load_settings()  # Load settings from a JSON file

# Set up paths for data
paths = Paths(settings)  # Initialize paths object with loaded settings
paths.load_device_paths()  # Load device-specific paths
paths.create_paths()  # Create any necessary file paths

# Load EEG dataset using configured settings and paths
if settings.dataset == 'pilot01':
    dataset = PilotEEGDataLoader(paths=paths, settings=settings)  # Initialize EEG dataset loader
elif settings.dataset == 'verbmem':
    dataset = VerbMemEEGDataLoader(paths=paths, settings=settings)  # Initialize EEG dataset loader

dataset.load_data(patient_ids=settings.patient)  # Load EEG data for specified patients

# Preprocess the loaded dataset
data_preprocessor = DataPreprocessor(paths=paths, settings=settings)  # Initialize data preprocessor
dataset = data_preprocessor.preprocess(dataset)  # Apply preprocessing steps to the dataset

stim = 'w'
data_train, labels_train = [], []
for idx, (subject_name, subject_dataset) in enumerate(dataset.all_patient_data.items()):
    time = subject_dataset.time_ms
    start_time_idx = np.argmin(np.abs(time + 500))
    end_time_idx = np.argmin(np.abs(time - 1500))
    time = time[start_time_idx:end_time_idx]
    data = subject_dataset.data[:, :, start_time_idx:end_time_idx]
    labels_df = pd.DataFrame(subject_dataset.labels)

    block_idx = np.unique(subject_dataset.labels['block_number'])
    label_df = pd.DataFrame(subject_dataset.labels)

    # specify blocks with a stim 'w' or 'i'
    blocks = {idx: list(np.unique(label_df[label_df['block_number'] == idx]['block_type'])) for idx in block_idx}
    blocks_with_stim = [idx for idx, vals in blocks.items() if any(stim in val for val in vals)]

    data = data[np.isin(label_df['block_number'], blocks_with_stim)]
    label_df = label_df[label_df['block_number'].isin(blocks_with_stim)]

    data = data[label_df['is_correct'] & (label_df['stim'] != 'ctl')]
    label_df = label_df[label_df['is_correct'] & (label_df['stim'] != 'ctl')]
    label_df['sub_id'] = idx

    data_train.append(data)
    labels_train.append(label_df)

data_train = np.concatenate(data_train, axis=0)
labels_train = pd.concat(labels_train)

# data_train = np.transpose(data_train, (0, 2, 1))

data_train = data_train[:, np.newaxis, :, :]
data_train, data_test, labels_train, labels_test = train_test_split(data_train, labels_train, test_size=0.2,
                                                                    random_state=42)
data_train, data_val, labels_train, labels_val = train_test_split(data_train, labels_train, test_size=0.2,
                                                                  random_state=42)

def one_hot_encode(labels):
    # Assumes labels are already torch tensors of type long and contain 0 for not experienced, 1 for experienced, etc.
    num_classes = labels.max() + 1  # Calculate the number of classes based on the maximum label
    return torch.nn.functional.one_hot(labels, num_classes=num_classes)

# Convert your DataFrames' 'is_experienced' columns to tensors first
labels_train_tensor = torch.tensor(labels_train['is_experienced'].values).to(torch.long)
labels_val_tensor = torch.tensor(labels_val['is_experienced'].values).to(torch.long)
labels_test_tensor = torch.tensor(labels_test['is_experienced'].values).to(torch.long)

labels_train_one_hot = torch.nn.functional.one_hot(labels_train_tensor, num_classes=2).to(torch.float32)
labels_val_one_hot = torch.nn.functional.one_hot(labels_val_tensor, num_classes=2).to(torch.float32)
labels_test_one_hot = torch.nn.functional.one_hot(labels_test_tensor, num_classes=2).to(torch.float32)

y_train = labels_train['is_experienced'].values[:, np.newaxis]
y_val = labels_val['is_experienced'].values[:, np.newaxis]
y_test = labels_test['is_experienced'].values[:, np.newaxis]

train_dataset = SimpleEEGDataset(torch.Tensor(data_train), torch.Tensor(y_train))
val_dataset = SimpleEEGDataset(torch.Tensor(data_val), torch.Tensor(y_val))
test_dataset = SimpleEEGDataset(torch.Tensor(data_test), torch.Tensor(y_test))


net = EEGNet(signal_length=data_train.shape[-1], num_class=1).cuda(0)
criterion = nn.BCELoss()
history = net.fit(data_train, y_train, data_val, y_val,
                  batch_size=64,
                  optimizer=optim.Adam(net.parameters()),
                  criterion=nn.BCELoss(),
                  epochs=100)

metric = net.evaluate(data_train, y_train, batch_size=64)
print("===============Train================")
print(metric)
metric = net.evaluate(data_val, y_val, batch_size=64)
print("===============Validation================")
print(metric)
metric = net.evaluate(data_test, y_test, batch_size=64)
print("===============Test================")
print(metric)

fig, axs = plt.subplots(6, 1, figsize=(6, 25))
for key, value in history.items():
    if 'loss' in key:
        axs[0].plot(value, label=key)
        axs[0].legend()
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].set_title('Loss')

    elif 'accuracy' in key:
        axs[1].plot(value, label=key)
        axs[1].legend()
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_title('Accuracy')
    elif 'precision' in key:
        axs[2].plot(value, label=key)
        axs[2].legend()
        axs[2].set_xlabel('Epochs')
        axs[2].set_ylabel('Precision')
        axs[2].set_title('Precision')

    elif 'recall' in key:
        axs[3].plot(value, label=key)
        axs[3].legend()
        axs[3].set_xlabel('Epochs')
        axs[3].set_ylabel('Recall')
        axs[3].set_title('Recall')
    elif 'f1' in key:
        axs[4].plot(value, label=key)
        axs[4].legend()
        axs[4].set_xlabel('Epochs')
        axs[4].set_ylabel('F1 Score')
        axs[4].set_title('F1 Score')
    elif 'roc_auc' in key:
        axs[5].plot(value, label=key)
        axs[5].legend()
        axs[5].set_xlabel('Epochs')
        axs[5].set_ylabel('ROC AUC')
        axs[5].set_title('ROC AUC')
plt.tight_layout()
plt.show()
