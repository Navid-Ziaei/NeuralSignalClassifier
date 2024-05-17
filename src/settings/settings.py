import json
import os
import warnings


class Settings:
    def __init__(self, patient='all', verbose=True):

        self.__supported_feature_transformation = ['normalize', 'standardize']
        self.__supported_datasets = ['verbmem', 'pilot01', 'clear']

        self.patient = None
        self.verbose = verbose

        self.__patient = patient
        self.__debug_mode = False
        self.__save_features = False
        self.__load_pretrained_model = False
        self.__num_fold = 5
        self.__test_size = 0.2
        self.__file_format = '.pkl'
        self.__load_features = False
        self.__feature_transformation = None
        self.__dataset = None

        self.__load_epoched_data = False
        self.__save_epoched_data = False
        self.__load_preprocessed_data = False
        self.__save_preprocessed_data = False

    def load_settings(self):
        """
        This function loads the json files for settings and network settings from the working directory and
        creates a Settings object based on the fields in the json file. It also loads the local path of the dataset
        from device_path.json
        return:
            settings: a Settings object
            network_settings: a dictionary containing settings of the model
            device_path: the path to the datasets on the local device
        """

        """ working directory """
        working_folder = os.path.dirname(os.path.realpath(__file__))
        parent_folder = os.path.dirname(os.path.dirname(working_folder)) + '/'

        """ loading settings from the json file """
        try:
            with open(parent_folder + "/configs/settings.json", "r") as file:
                settings_json = json.loads(file.read())
        except:
            raise Exception('Could not load settings.json from the working directory!')

        """ creating settings """
        if "patient" not in settings_json.keys():
            raise Exception('"patient" was not found in settings.json!')
        if "dataset" not in settings_json.keys():
            raise Exception('"dataset" was not found in settings.json!')

        self.patient = settings_json["patient"]

        for key, value in settings_json.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise Exception('{} is not an attribute of the Settings class!'.format(key))
        if self.verbose is True:
            print("Patient: {}".format(self.patient))

    @property
    def load_epoched_data(self):
        return self.__load_epoched_data

    @load_epoched_data.setter
    def load_epoched_data(self, value):
        if isinstance(value, bool):
            self.__load_epoched_data = value
        else:
            raise ValueError("load_epoched_data should be true or false")

    @property
    def save_epoched_data(self):
        return self.__save_epoched_data

    @save_epoched_data.setter
    def save_epoched_data(self, value):
        if isinstance(value, bool):
            self.__save_epoched_data = value
        else:
            raise ValueError("save_epoched_data should be true or false")

    @property
    def load_preprocessed_data(self):
        return self.__load_preprocessed_data

    @load_preprocessed_data.setter
    def load_preprocessed_data(self, value):
        if isinstance(value, bool):
            self.__load_preprocessed_data = value
        else:
            raise ValueError("load_preprocessed_data should be true or false")

    @property
    def save_preprocessed_data(self):
        return self.__save_preprocessed_data

    @save_preprocessed_data.setter
    def save_preprocessed_data(self, value):
        if isinstance(value, bool):
            self.__save_preprocessed_data = value
        else:
            raise ValueError("save_preprocessed_data should be true or false")

    @property
    def dataset(self):
        return self.__dataset

    @dataset.setter
    def dataset(self, dataset_nam):
        if isinstance(dataset_nam, str) and dataset_nam in self.__supported_datasets:
            self.__dataset = dataset_nam
        else:
            raise ValueError(f"The dataset should be selected from {self.__supported_datasets}")

    @property
    def feature_transformation(self):
        return self.__feature_transformation

    @feature_transformation.setter
    def feature_transformation(self, value):
        if value is None:
            self.__feature_transformation = None
        elif isinstance(value, str) and value.lower() in self.__supported_feature_transformation:
            self.__feature_transformation = value.lower()
        else:
            raise ValueError(
                f"The feature_transformation should be selected from {self.__supported_feature_transformation}")

    @property
    def load_features(self):
        return self.__load_features

    @load_features.setter
    def load_features(self, value):
        if isinstance(value, bool):
            self.__load_features = value
        else:
            raise ValueError("The load_features should be a boolean (true or false)")

    @property
    def file_format(self):
        return self.__file_format

    @file_format.setter
    def file_format(self, format_name):
        if isinstance(format_name, str) and format_name[0] == '.':
            self.__file_format = format_name
        else:
            raise ValueError(f"file_format should be a string starting with . (.pkl, .mat, and etc.) "
                             f"but we got {format_name}")

    @property
    def num_fold(self):
        return self.__num_fold

    @num_fold.setter
    def num_fold(self, k):
        if isinstance(k, int) and k > 0:
            self.__num_fold = k
        else:
            raise ValueError("num_fold should be integer bigger than 0")

    @property
    def test_size(self):
        return self.__test_size

    @test_size.setter
    def test_size(self, value):
        if 0 < value < 1:
            self.__test_size = value
        else:
            raise ValueError("test_size should be float number between 0 to 1")

    @property
    def load_pretrained_model(self):
        return self.__load_pretrained_model

    @load_pretrained_model.setter
    def load_pretrained_model(self, value):
        if isinstance(value, bool):
            self.__load_pretrained_model = value
        else:
            raise ValueError("load_pretrained_model should be True or False")

    @property
    def save_features(self):
        return self.__save_features

    @save_features.setter
    def save_features(self, value):
        if isinstance(value, bool):
            self.__save_features = value
        else:
            raise ValueError("save_features should be True or False")

    @property
    def debug_mode(self):
        return self.__debug_mode

    @debug_mode.setter
    def debug_mode(self, value):
        if isinstance(value, bool):
            self.__debug_mode = value
        else:
            raise ValueError("The v should be boolean")
