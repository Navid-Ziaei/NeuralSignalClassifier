from src.settings import Paths, Settings
from src.data_loader import VerbMemEEGDataLoader, PilotEEGDataLoader, CLEARDataLoader
from src.feature_extraction import FeatureExtractor
from src.data_preprocess import DataPreprocessor
from src.utils import *
from src.model.training_utils import *
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load settings from settings.json
settings = Settings()  # Initialize settings object
settings.load_settings()  # Load settings from a JSON file

# Set up paths for data
paths = Paths(settings)  # Initialize paths object with loaded settings
paths.load_device_paths()  # Load device-specific paths
paths.create_paths()  # Create any necessary file paths

# Check if features are available
features_raw_df_list = []
if isinstance(settings.patient, list):
    for patient in settings.patient:
        if os.path.exists(paths.feature_path + f"feature_{settings.dataset}_{patient}.csv"):
            features_raw_df = pd.read_csv(paths.feature_path + f"feature_{settings.dataset}_{patient}.csv")
        else:
            # Load EEG dataset using configured settings and paths
            if settings.dataset == 'pilot01':
                dataset = PilotEEGDataLoader(paths=paths, settings=settings)  # Initialize EEG dataset loader
            elif settings.dataset == 'verbmem':
                dataset = VerbMemEEGDataLoader(paths=paths, settings=settings)  # Initialize EEG dataset loader
            elif settings.dataset == 'clear':
                dataset = CLEARDataLoader(paths=paths, settings=settings)  # Initialize EEG dataset loader
            else:
                raise ValueError("dataset in model_settings should be verbmem or pilot01 or clear")

            dataset.load_data(patient_ids=patient)  # Load EEG data for specified patients

            # Preprocess the loaded dataset
            preprocessing_configs = {
                'remove_baseline': {'normalize': False, 'baseline_t_min': -1000},
                'low_pass_filter': {'cutoff': 45, 'order': 5}
            }
            data_preprocessor = DataPreprocessor(paths=paths, settings=settings)  # Initialize data preprocessor
            dataset = data_preprocessor.preprocess(dataset, preprocessing_configs)  # Apply preprocessing steps to the dataset

            """
            for idx in range(len(list(dataset.all_patient_data.keys()))):
                single_patient_data = dataset.all_patient_data[list(dataset.all_patient_data.keys())[idx]]
                visualize_erp(patient_data=single_patient_data, channel_idx=45,
                              label=single_patient_data.labels['is_experienced'], label_names=['Not Experienced', 'Experienced'])
                visualize_erp(patient_data=single_patient_data, channel_idx=45,
                              label=single_patient_data.labels['go_nogo'], label_names=['NoGo', 'Go'])
            """

            # Extract features from the preprocessed dataset
            feature_extraction_configs = {
                'time_n200': {'start_time': 150, 'end_time': 250},
                'time_p300': {'start_time': 250, 'end_time': 550},
                'time_post_p300': {'start_time': 550, 'end_time': 750},
                'frequency1': {'time_start': 0, 'end_time': 500},
                'frequency2': {'time_start': 250, 'end_time': 700}
            }
            feature_extractor = FeatureExtractor(paths=paths, settings=settings)  # Initialize feature extractor
            feature_extractor.extract_features(dataset,
                                               feature_extraction_configs)  # Extract relevant features from the dataset
            features_raw_df, *_ = feature_extractor.get_feature_array(dataset)
            features_raw_df.to_csv(paths.feature_path + f"feature_{settings.dataset}_{patient}.csv", index=False)
        features_raw_df_list.append(features_raw_df)
# Get the features matrix and labels from the raw features DataFrame
drop_columns = get_drop_colums(settings)
results_logger = ResultList(method_list=settings.method_list, metric_list=settings.metric_list)

# Define the KFold cross-validator
if isinstance(settings.cross_validation_mode, int):
    kf = StratifiedKFold(n_splits=settings.cross_validation_mode, shuffle=True, random_state=42)

elif isinstance(settings.cross_validation_mode, str) and settings.cross_validation_mode == 'block':
    kf = None
elif isinstance(settings.cross_validation_mode, str) and settings.cross_validation_mode == 'order':
    kf = None
else:
    raise ValueError("cross_validation_mode should be number of folds or be 'block' for block based")


certain = []
all_patient_group_results = {method: [] for method in settings.method_list}
for patient_id, features_raw_df in enumerate(features_raw_df_list):
    print(f"============ Subject {patient_id} ({settings.patient[patient_id]}) from {len(features_raw_df_list)} ============ \n")
    # select patient
    # select patient
    columns_to_remove = [col for col in features_raw_df.columns if "EX" in col]
    features_df = features_raw_df.drop(columns=columns_to_remove)

    # select labels
    y_one_hot, labels_array, unique_pids, patients_files, features_df = get_labels(features_df, settings)

    # Get the features matrix and labels from the raw features DataFrame
    results_logger.add_subject(unique_pids, patients_files)

    # Add the result path for this subject
    paths.create_subject_paths(patients_files)

    # Perform cross-validation
    fold_results = {method: [] for method in settings.method_list}
    fold_results_group = {method: [] for method in settings.method_list}

    if kf is None and settings.cross_validation_mode == 'block':
        block_nums = features_df['block_number'].unique()
        folds = [(np.where(features_df['block_number'] != block)[0],
                  np.where(features_df['block_number'] == block)[0]) for block in block_nums]

        fold_blocks = [(features_df[features_df['block_number'] != block]['block_number'].unique(),
                        features_df[features_df['block_number'] == block]['block_number'].unique()) for block in
                       block_nums]
    if kf is None and settings.cross_validation_mode == 'order':
        num_trials = labels_array.shape[0]
        trial_idx = np.arange(num_trials)
        fold_idx = np.int16(10*trial_idx / num_trials)
        folds = [(np.where(fold_idx != fold)[0], np.where(fold_idx == fold)[0]) for fold in np.unique(fold_idx)]
    else:
        folds = kf.split(features_df, labels_array)

    for fold_idx, (train_index, test_index) in enumerate(folds):

        paths.create_fold_path(fold_idx)

        # select features
        features_matrix, selected_features, patients_ids, patients_files = \
            get_selected_features(features_df.copy(), settings, paths,
                                  fold_idx, train_index, train_index,
                                  target_columns_drop=drop_columns)

        data_train, data_test = features_matrix[train_index], features_matrix[test_index]
        labels_train, labels_test = labels_array[train_index], labels_array[test_index]
        y_train, y_test = y_one_hot[train_index], y_one_hot[test_index]
        pid_train, pid_test = patients_ids[train_index], patients_ids[test_index]

        for method in settings.method_list:
            print(f"=========== Train Subject {patient_id} ({settings.patient[patient_id]}) "
                  f"from {len(features_raw_df_list)} Fold {fold_idx} Model {method} =========== \n")
            if method.lower() == 'xgboost':
                results, group_result = train_xgb(data_train, labels_train, data_test, labels_test, paths)
            elif method.lower() == 'ldgd':
                results, group_result = train_ldgd(data_train, labels_train, data_test, labels_test,
                                     y_train, y_test,
                                     settings, paths)
            elif method.lower() == 'fast_ldgd':
                results, group_result = train_fast_ldgd(data_train, labels_train, data_test, labels_test,
                                          y_train, y_test,
                                          settings, paths,
                                          use_validation=True)
            else:
                raise ValueError("Method should be 'xgboost' or 'ldgd'")

            fold_results[method].append(results)
            fold_results_group[method].append(group_result)

        plt.close('all')

    # Compute average scores
    for method in settings.method_list:
        for metric in settings.metric_list:
            avg_score = np.mean([result[metric] for result in fold_results[method]])
            std_score = np.std([result[metric] for result in fold_results[method]])
            results_logger.update_result(method, metric, avg_score, std_score)
            print(f"Method {method}: {metric}: {avg_score} +- {std_score}")

    for key in fold_results.keys():
        df = pd.DataFrame(fold_results[key])
        df.to_csv(paths.results_base_path + f'{key}_results.csv', index=False)

        # group base
        # Extract true values and predictions from the given data
        true_values = []
        predictions = []

        for gp_result in fold_results_group[key]:
            for true_value, predicted_value in gp_result.items():
                true_values.append(true_value)
                predictions.append(predicted_value)

        # Convert lists to numpy arrays
        true_values = np.array(true_values)
        predictions = np.array(predictions)

        # Calculate accuracy
        grp_accuracy = accuracy_score(true_values, predictions.round())
        grp_precision = precision_score(true_values, predictions.round(), average='binary')
        grp_recall = recall_score(true_values, predictions.round(), average='binary')
        grp_f1 = f1_score(true_values, predictions.round(), average='binary')
        grp_auc = roc_auc_score(true_values, predictions)

        group_result = {
            'accuracy': grp_accuracy,
            'precision': grp_precision,
            'recall': grp_recall,
            'f1': grp_f1,
            'auc': grp_auc
        }
        with open(paths.results_base_path + f'group_results_{key}.txt', 'w') as f:
            f.write(str(group_result))

        all_patient_group_results[key].append(group_result)



result_df = results_logger.to_dataframe()
result_df.to_csv(paths.base_path + paths.folder_name + '\\results.csv')

for key in all_patient_group_results.keys():
    df_gp = pd.DataFrame(all_patient_group_results[key], index=settings.patient)
    df_gp.to_csv(paths.base_path + paths.folder_name + f'\\group_results_{key}.csv')






