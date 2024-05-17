# lexical memory eeg task 

# Feature Analyses
Welcome to the GitHub repository for our model! 
<br/>

## Table of Contents
* [General Information](#general-information)
* [Getting Started](#getting-started)
* [Example](#example)
* [Reading in Data](#reading-in-edf-data)
* [Repository Structure](#repository-structure)
* [Citations](#citations)
* [Status](#status)
* [References](#references)
* [Contact](#contact)
<br/>

## General Information
...
<br/>

## Getting Started

1. Clone this repository to your local machine.

2. Install the required dependencies. (See `requirements.txt` for details)

3. Prepare your dataset

4. Create the `./configs/settings.json` according to `./cinfigs/settings_sample.json`

5. Create the `./configs/device_path.json` according to `./cinfigs/device_path_sample.json`

6. Run the `main.py` script to execute the BTsC model.

In this example we create some chirp data and run the multitaper spectrogram on it.
```
from src.data import EEGDataLoader
from src.features import FeatureExtractor
from src.models import BTsCModel, find_best_time, train_and_evaluate_model_kfold
from src.settings import Settings, Paths

# load settings from settings.json
settings = Settings()
settings.load_settings()

# load device paths from device.json and create result paths
paths = Paths(settings)
paths.load_device_paths()

# load raw data
data_loader = EEGDataLoader(paths, settings)
dataset = data_loader.load_data()

# Extract Features
feature_extractor = FeatureExtractor(dataset=dataset[1], settings=settings, feature_path=paths.feature_path)
time_features, features, labels, feature_name = feature_extractor.extract_BTsC_feature()
trial_idx = dataset[1].trial_info['Trial number']

# Find Best Time
# find_best_time(features, labels, trial_idx, feature_name, time_features, settings, paths)

# Train Model
model = BTsCModel(settings)
results, best_results, df_error_analysis = train_and_evaluate_model_kfold(model, features, labels, trial_idx,                                                                      feature_name, settings, paths)
```
<br/>

## Reading in Data
To load data, you need 
1. '_VerbMem_ecnt_af.mat' files. 
2. '.mat' files for x_new. 
<br/>

## Repository Structure
This repository is organized as follows:

- `/main.py`: The main script to run the analysis.

- `/data`: Contains scripts for data loading (`dataset.py`).

- `/data_preprocess`: Contains the `data_preprocess.py` script for data preprocessing.

- `/experiments`: Contains scripts for different experiments, such as `spectral_matrix_experiment.py`.

- `/feature_extraction`: Contains `feature_extraction.py` for feature extraction tasks.

- `/models`: Contains the ML models.

- `/settings`: Contains scripts to manage settings (`settings.py`) and paths (`paths.py`).

- `/utils`: Contains utility scripts (`utils.py`, `connectivity.py`) and a `multitapper` subfolder with multitaper spectrogram implementation (`multitaper_spectrogram_python.py`) and an example (`example.py`). 

- `/visualization`: Contains the `visualization.py` script for data and result visualization.
<br/>

## Citations
The code contained in this repository for BTsC is companion to the paper:  

which should be cited for academic use of this code.  
<br/>

## Contributing

We encourage you to contribute to model! 

## License

This project is licensed under the terms of the MIT license.