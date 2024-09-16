from src.settings import Paths, Settings

from collections import Counter
import itertools
from src.experiments.utils.train_gplvm_utils import *
from sklearn.model_selection import KFold
from scipy.stats import ttest_ind
import seaborn as sns
from src.experiments.utils import *


def convert(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


model_settings = {
    'dataset': 'pilot01',
    'data_dim': None,
    'load_trained_model': False,
    'binary_column': 'is_experienced',  # decision  old_new is_go is_correct is_resp is_experienced
    'features': 'all',  # 'all', 'corr', 'pre_selected'
    'dataset_file': "feature_pilot01.csv",
    # "feature_dataset.csv", "feature_dataset_detrand.csv", "feature_dataset_normalized.csv"
    'feature_transformation': 'normalize'  # 'normalize', 'standardize', None

}

if model_settings['dataset'] == 'verbmem':
    drop_columns = ['id', 'old_new', 'decision', 'subject_file']
elif model_settings['dataset'] == 'pilot01':
    drop_columns = ['id', 'subject_file', 'block_number', 'block_type', 'stim_indicator', 'go_nogo', 'is_experienced',
                    'is_resp', 'is_correct', 'stim']
else:
    raise ValueError("dataset in model_settings should be verbmem or pilot01")

result_list = {
    'subject id': [],
    'filename': []
}

# Load settings from settings.json
settings = Settings()  # Initialize settings object
settings.load_settings()  # Load settings from a JSON file

# ###################### Set up paths for data  ######################
paths = Paths(settings)  # Initialize paths object with loaded settings
paths.load_device_paths()  # Load device-specific paths
paths.create_paths()  # Create any necessary file paths

features_raw_df = pd.read_csv(paths.feature_path[:-22] + model_settings['dataset_file'])
patients = list(np.unique(features_raw_df['id']))
# [0, 1, 2, 3, 10, 11, 12, 13, 17, 41, 42, 47, 52, 54, 58, 65, 69, 72, 76, 83, 95]
certain = []
for patient_id in patients:
    print(f"============ Subject {patient_id} from {len(patients)} ============ \n")
    # select patient
    columns_to_remove = [col for col in features_raw_df.columns if "EX" in col]
    features_raw_df = features_raw_df.drop(columns=columns_to_remove)
    features_df = features_raw_df[features_raw_df['id'] == patient_id]
    # features_df = features_raw_df

    # select labels
    y_one_hot, labels_array, unique_pids, patients_files, features_df = get_labels(features_df, model_settings)

    model_settings['subject id'] = patients_files

    result_list['subject id'].append(unique_pids)
    result_list['filename'].append(patients_files.split('_')[0])

    paths.create_subject_paths(patients_files)

    feature_cols = [column for column in features_df.columns if column not in drop_columns]

    # Calculate p-values for each numeric column
    p_values = {}

    for col in feature_cols:
        # Split the data based on "is_experienced"
        group_true = features_df[features_df['is_experienced'] == True][col]
        group_false = features_df[features_df['is_experienced'] == False][col]

        # Perform t-test
        t_stat, p_val = ttest_ind(group_true.dropna(), group_false.dropna(), equal_var=False)

        # Store the p-value
        p_values[col] = p_val

    # Sort features by p-value
    sorted_p_values = {k: v for k, v in sorted(p_values.items(), key=lambda item: item[1])}

    # Select top 20 features
    top_20_features = list(sorted_p_values.keys())[:20]

    plt.figure(figsize=(30, 15))
    for i, feature in enumerate(top_20_features, 1):
        plt.subplot(4, 5, i)
        sns.boxplot(x='is_experienced', y=feature, data=features_df)
        sns.stripplot(x='is_experienced', y=feature, data=features_df, color='black', alpha=0.5)
        plt.title(feature, fontsize=22)

    plt.tight_layout()
    plt.show()

    # analyze_line_fit_p_value(feature_list, path_results, p_value_thresh, save_path, target='rec_old')
