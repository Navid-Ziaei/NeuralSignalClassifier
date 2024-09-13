import os
import random

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.settings import Paths, Settings
from src.data_loader import VerbMemEEGDataLoader, PilotEEGDataLoader, CLEARDataLoader
from src.feature_extraction import FeatureExtractor
from src.data_preprocess import DataPreprocessor
from src.utils import *
from src.model.utils.training_utils import *

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def main():
    # Load settings from settings.json
    settings = Settings()
    settings.load_settings()

    # Set up paths for data
    paths = Paths(settings)
    paths.load_device_paths()
    paths.create_paths()

    # Load or extract features
    features_raw_df_dict = load_or_extract_features(settings, paths)

    # Perform training with cross-validation or single fold
    train_model_with_folds(features_raw_df_dict, settings, paths, RANDOM_SEED)


def load_or_extract_features(settings, paths):
    """
        Load or extract features for all patients specified in the settings.

        This function checks if the features for each patient are already saved as a CSV file.
        If the features are found and the `load_features` setting is enabled, the function
        loads the features from the CSV file. Otherwise, it extracts the features from the raw
        EEG data, saves them (if `save_features` is enabled), and appends the DataFrame to the list.

        Args:
            settings (Settings): The settings object containing configurations.
            paths (Paths): The paths object containing file paths.

        Returns:
            List[pd.DataFrame]: A list of DataFrames, each containing the features for a patient.
    """
    features_raw_df_dict = {}
    for patient in settings.patient:
        file_list = [file for file in os.listdir(paths.feature_path) if
                     file.endswith('.csv') and patient in file and settings.dataset_task in file]
        if settings.load_features and len(file_list)>0:
            features_raw_df = {}
            for file in file_list:
                features_raw_df['_'.join(file.split('_')[:-1])] = pd.read_csv(paths.feature_path+file)
        else:
            features_raw_df = extract_features_for_patient(patient, settings, paths)
            # if settings.save_features:
            #     features_raw_df.to_csv(feature_file, index=False)
        features_raw_df_dict.update(features_raw_df)
    return features_raw_df_dict


def extract_features_for_patient(patient, settings, paths):
    """
    Load, preprocess, and extract features from EEG data for a specific patient.

    This function loads the EEG dataset for a given patient based on the dataset type
    specified in the settings. The loaded data is then preprocessed according to the
    predefined preprocessing configurations. After preprocessing, relevant features are
    extracted from the data.

    Args:
        patient (str): The identifier of the patient whose data is being processed.
        settings (Settings): The settings object containing configurations.
        paths (Paths): The paths object containing file paths.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted features for the specified patient.
    """
    # Load EEG dataset using configured settings and paths
    if settings.dataset == 'pilot01':
        dataset = PilotEEGDataLoader(paths=paths, settings=settings)
    elif settings.dataset == 'verbmem':
        dataset = VerbMemEEGDataLoader(paths=paths, settings=settings)
    elif settings.dataset == 'clear':
        dataset = CLEARDataLoader(paths=paths, settings=settings)
    else:
        raise ValueError("Dataset should be 'verbmem', 'pilot01', or 'clear'")

    dataset.load_data(patient_ids=patient)

    # Preprocess the loaded dataset
    data_preprocessor = DataPreprocessor(paths=paths, settings=settings)
    dataset = data_preprocessor.preprocess(dataset, settings.preprocessing_configs)

    # Extract features from the preprocessed dataset
    feature_extractor = FeatureExtractor(paths=paths, settings=settings)
    features_raw_df = feature_extractor.extract_features(dataset, settings.feature_extraction_configs)
    # features_raw_df, *_ = feature_extractor.get_feature_array(dataset)

    return features_raw_df




if __name__ == "__main__":
    main()
