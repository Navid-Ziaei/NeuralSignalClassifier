import random
from src.feature_extraction.feature_extractor_utils import *
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


if __name__ == "__main__":
    main()
