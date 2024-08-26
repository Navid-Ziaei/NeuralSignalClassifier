from src.feature_extraction import FeatureExtractor


def load_or_extract_features(settings, paths):
    features_raw_df_list = []
    for patient in settings.patient:
        feature_file = os.path.join(paths.feature_path, f"feature_{settings.dataset}_{patient}.csv")
        if settings.load_features and os.path.exists(feature_file):
            features_raw_df = pd.read_csv(feature_file)
        else:
            features_raw_df = extract_features_for_patient(patient, settings, paths)
            if settings.save_features:
                features_raw_df.to_csv(feature_file, index=False)
        features_raw_df_list.append(features_raw_df)
    return features_raw_df_list

def extract_features_for_patient(patient, settings, paths):
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
    preprocessing_configs = {'low_pass_filter': {'cutoff': 45, 'order': 5}}
    data_preprocessor = DataPreprocessor(paths=paths, settings=settings)
    dataset = data_preprocessor.preprocess(dataset, preprocessing_configs)

    # Extract features from the preprocessed dataset
    feature_extraction_configs = {
        'time_n200': {'start_time': 150, 'end_time': 250},
        'time_p300': {'start_time': 250, 'end_time': 550},
        'time_post_p300': {'start_time': 550, 'end_time': 750},
        'frequency1': {'time_start': 0, 'end_time': 500},
        'frequency2': {'time_start': 250, 'end_time': 700}
    }
    feature_extractor = FeatureExtractor(paths=paths, settings=settings)
    feature_extractor.extract_features(dataset, feature_extraction_configs)
    features_raw_df, *_ = feature_extractor.get_feature_array(dataset)

    return features_raw_df