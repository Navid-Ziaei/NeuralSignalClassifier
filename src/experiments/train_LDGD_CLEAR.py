import pandas as pd

from src.settings import Paths, Settings

from collections import Counter
import itertools
from src.experiments.utils.train_gplvm_utils import *
from sklearn.model_selection import KFold


def convert(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


model_settings = {
    'dataset': 'clear',
    'data_dim': None,
    'latent_dim': 7,
    'num_inducing_points': 15,
    'num_epochs_train': 2000,
    'num_epochs_test': 2000,
    'batch_size': 300,
    'cls_weight': 1,
    'load_trained_model': False,
    'use_gpytorch': True,
    'patient': None,
    'cv': 5, # 'block',
    'max_num_patient': 3,
    'binary_column': 'ColorLev',  # decision  old_new is_go is_correct is_resp is_experienced
    'features': 'corr',  # 'all', 'corr', 'pre_selected'
    'dataset_file': "feature_clear_p01.csv",
    # "feature_dataset.csv", "feature_dataset_detrand.csv", "feature_dataset_normalized.csv"
    'feature_transformation': 'normalize',  # 'normalize', 'standardize', None
    'stim': 'w'

}

if model_settings['dataset'] == 'verbmem':
    drop_columns = ['id', 'old_new', 'decision', 'subject_file']
elif model_settings['dataset'] == 'pilot01':
    drop_columns = ['id', 'subject_file', 'block_number', 'block_type', 'stim_indicator', 'go_nogo', 'is_experienced',
                    'is_resp', 'is_correct', 'stim']
elif model_settings['dataset'] == 'clear':
    drop_columns = ['id', 'subject_file', 'ColorLev']
else:
    raise ValueError("dataset in model_settings should be verbmem or pilot01")

method_list = ['xgboost', 'ldgd']  # 'ldgd' 'xgboost'

result_list = {
    'subject id': [],
    'filename': []
}

for method in method_list:
    result_list[method + ' avg_f1_score'] = []
    result_list[method + ' std_f1_score'] = []
    result_list[method + ' avg_accuracy'] = []
    result_list[method + ' std_accuracy'] = []
    result_list[method + ' avg_recall'] = []
    result_list[method + ' std_recall'] = []
    result_list[method + ' avg_precision'] = []
    result_list[method + ' std_precision'] = []

# Load settings from settings.json
settings = Settings()  # Initialize settings object
settings.load_settings()  # Load settings from a JSON file

# Define the KFold cross-validator
if isinstance(model_settings['cv'], int):
    kf = KFold(n_splits=model_settings['cv'], shuffle=True, random_state=42)
elif isinstance(model_settings['cv'], str) and model_settings['cv'] == 'block':
    kf = None
else:
    raise ValueError("cv should be number of folds or be 'block' for block based")

# ###################### Set up paths for data  ######################
paths = Paths(settings)  # Initialize paths object with loaded settings
paths.load_device_paths()  # Load device-specific paths
paths.create_paths()  # Create any necessary file paths

features_raw_df = pd.read_csv(paths.feature_path + model_settings['dataset_file'])
patients = list(np.unique(features_raw_df['id']))
# [0, 1, 2, 3, 10, 11, 12, 13, 17, 41, 42, 47, 52, 54, 58, 65, 69, 72, 76, 83, 95]
certain = []
for patient_id in patients:
    print(f"============ Subject {patient_id} from {len(patients)} ============ \n")
    # select patient
    features_df = features_raw_df[features_raw_df['id'] == patient_id]

    labels_array = features_df[model_settings['binary_column']].values
    patients_files = features_df['subject_file'].values[0]

    y_one_hot = np.zeros((labels_array.size, 2))
    y_one_hot[np.arange(labels_array.size), labels_array.astype(int)] = 1
    y_one_hot = y_one_hot.astype(int)

    model_settings['subject id'] = patients_files
    result_list['filename'].append(patients_files.split('_')[0])

    paths.create_subject_paths(patients_files)

    # Perform cross-validation
    fold_acc, fold_f1_score = {method: [] for method in method_list}, {method: [] for method in method_list}
    fold_precision, fold_recall = {method: [] for method in method_list}, {method: [] for method in method_list}

    folds = kf.split(features_df)

    folds_info = {
        'number of train_samples (Black)': [],
        'number of test_samples (White)': [],
        'Label imbalance (Black/White) train': [],
        'Label imbalance (Black/White) test': []

    }
    for fold_idx, (train_index, test_index) in enumerate(folds):
        paths.create_fold_path(fold_idx)
        experienced_train = features_df.iloc[np.array(train_index)][model_settings['binary_column']]
        experienced_test = features_df.iloc[np.array(test_index)][model_settings['binary_column']]

        folds_info['number of train_samples (Black)'].append((experienced_train.sum(), np.sum(1-experienced_train)))
        folds_info['number of test_samples (White)'].append((experienced_test.sum(), np.sum(1-experienced_test)))
        folds_info['Label imbalance (Black/White) train'].append(experienced_train.sum()/np.sum(1-experienced_train))
        folds_info['Label imbalance (Black/White) test'].append(experienced_test.sum()/ np.sum(1-experienced_test))

        # select features
        features_matrix, selected_features, patients_ids, patients_files = \
            get_selected_features(features_df.copy(), model_settings, paths,
                                  fold_idx, train_index, train_index,
                                  target_columns_drop=drop_columns)

        data_train, data_test = features_matrix[train_index], features_matrix[test_index]
        labels_train, labels_test = labels_array[train_index], labels_array[test_index]
        y_train, y_test = y_one_hot[train_index], y_one_hot[test_index]
        pid_train, pid_test = patients_ids[train_index], patients_ids[test_index]

        for method in method_list:
            if method == 'xgboost':
                results = train_xgb(data_train, labels_train, data_test, labels_test, paths,
                                    selected_features=selected_features)
            elif method == 'ldgd':
                results = train_ldgd(data_train, labels_train, data_test, labels_test,
                                     y_train, y_test,
                                     model_settings, paths,
                                     shared_inducing_points=False,
                                     use_shared_kernel=False,
                                     cls_weight=model_settings['cls_weight'],
                                     reg_weight=1.0,
                                     early_stop=None)

            fold_acc[method].append(results['accuracy'])
            fold_f1_score[method].append(results['f1_score'])
            fold_precision[method].append(results['precision'])
            fold_recall[method].append(results['recall'])

        plt.close('all')
    pd.DataFrame(folds_info).to_csv(f"fold_info_{patients_files.split('_')[0]}.csv")
    # Compute average scores
    for method in method_list:
        # Compute average scores
        avg_accuracy = np.mean(fold_acc[method])
        avg_f1_score = np.mean(fold_f1_score[method])
        avg_precision = np.mean(fold_precision[method])
        avg_recall = np.mean(fold_recall[method])

        std_accuracy = np.std(fold_acc[method])
        std_f1_score = np.std(fold_f1_score[method])
        std_precision = np.std(fold_precision[method])
        std_recall = np.std(fold_recall[method])

        print(
            f"Method {method}: f1-score: {avg_f1_score} +- {std_f1_score}  \t accuracy: {avg_accuracy} += {std_accuracy}")

        result_list[method + ' avg_f1_score'].append(avg_f1_score)
        result_list[method + ' avg_accuracy'].append(avg_accuracy)
        result_list[method + ' avg_precision'].append(avg_precision)
        result_list[method + ' avg_recall'].append(avg_recall)

        result_list[method + ' std_f1_score'].append(std_f1_score)
        result_list[method + ' std_accuracy'].append(std_accuracy)
        result_list[method + ' std_precision'].append(std_precision)
        result_list[method + ' std_recall'].append(std_recall)

try:
    result_df = pd.DataFrame(result_list)
    result_df.to_csv(paths.base_path + paths.folder_name + '\\results.csv')
except:
    with open(paths.base_path + paths.folder_name + '\\results_all.json', 'w') as file:
        json.dump(result_list, file, default=convert, indent=2)

try:
    with open(paths.base_path + paths.folder_name + '\\model_setting.json', 'w') as file:
        json.dump(model_settings, file, default=convert, indent=2)
except:
    pass
try:
    # Flatten the list of lists
    flat_list = list(itertools.chain(*result_list['used features']))

    # Count the frequency of each item
    # Count the frequency of each item
    item_counts = Counter(flat_list)

    # Sort items by count and select the top 25
    top_25_items = item_counts.most_common(25)

    # Separate the items and their counts for plotting
    items, counts = zip(*top_25_items)

    # Plotting
    plt.figure(figsize=(10, 8))  # Adjust the size as needed
    plt.bar(items, counts)
    plt.xticks(rotation=90)  # Rotate labels to make them readable
    plt.xlabel('Feature Names')
    plt.ylabel('Frequency')
    plt.title('Top 25 Feature Names by Frequency')
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
    plt.savefig(paths.base_path + paths.folder_name + '\\hist.png')
except:
    pass
