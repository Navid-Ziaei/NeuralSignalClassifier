import json
import datetime
from pathlib import Path
import os

import yaml


class Paths:
    def __init__(self, settings):
        self.base_path = None
        self.path_model = None
        self.path_result = None

        self.raw_dataset_path = './'
        self.preprocessed_dataset_path = './'
        self.dgd_output_path = './'
        self.feature_path = './'
        self.model_path = './'
        self.channel_group_file = None
        self.path_subject_result = {}

        self.patient = settings.patient
        self.debug_mode = settings.debug_mode
        self.settings = settings

    def load_device_paths(self):

        """ working directory """
        working_folder = os.path.abspath(__file__)
        if 'Neural_Signal_Classifier' in working_folder:
            # Find the index of 'Suspicious_Message_Detection'
            index = working_folder.find('Neural_Signal_Classifier') + len('Neural_Signal_Classifier')
            # Extract the path up to 'Suspicious_Message_Detection'
            working_folder = working_folder[:index]
        else:
            print("The path does not contain 'Neural_Signal_Classifier'")
            working_folder = ''

        """ loading device path from the json file """
        try:
            with open(working_folder + "/configs/device_path.yaml", "r") as file:
                device = yaml.safe_load(file)
        except:
            raise Exception('Could not load device_path.yaml from the working directory!')

        for key, value in device.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise Exception('{} is not an attribute of the Settings class!'.format(key))

        if self.channel_group_file is None:
            self.channel_group_file = working_folder + "/configs/channel_groups.mat"

    def create_paths(self):
        working_folder = os.path.abspath(__file__)
        if 'Neural_Signal_Classifier' in working_folder:
            # Find the index of 'Suspicious_Message_Detection'
            index = working_folder.find('Neural_Signal_Classifier') + len('Neural_Signal_Classifier')
            # Extract the path up to 'Suspicious_Message_Detection'
            dir_path = working_folder[:index]
        else:
            print("The path does not contain 'Neural_Signal_Classifier'")
            dir_path = ''

        self.base_path = dir_path + '/results/'

        if self.settings.dataset == 'clear':
            self.base_path += f'/{self.settings.dataset_task}/'

        self.eda_results = self.base_path + '/eda_results/'
        if self.debug_mode is False:
            self.folder_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        else:
            self.folder_name = 'debug'
        results_base_path = self.base_path + self.folder_name + '/'
        """if Path(results_base_path).is_dir():
            shutil.rmtree(results_base_path)"""

        Path(results_base_path).mkdir(parents=True, exist_ok=True)
        Path(self.eda_results).mkdir(parents=True, exist_ok=True)
        Path(results_base_path + 'model/').mkdir(parents=True, exist_ok=True)
        self.path_model = os.path.join(results_base_path + 'model/')
        self.path_result = os.path.join(results_base_path)

    def create_subject_paths(self, subject_name):
        self.results_base_path = self.base_path + self.folder_name + f'/{subject_name}/'
        print(f"Creating path {self.results_base_path} ...")
        Path(self.results_base_path).mkdir(parents=True, exist_ok=True)
        Path(self.results_base_path + 'model/').mkdir(parents=True, exist_ok=True)
        self.path_model = os.path.join(self.results_base_path + 'model/')
        self.path_result = os.path.join(self.results_base_path)

    def create_fold_path(self, fold):
        self.fold_path = self.results_base_path + f'/fold{fold}/'
        Path(self.fold_path).mkdir(parents=True, exist_ok=True)
        Path(self.fold_path + 'model/').mkdir(parents=True, exist_ok=True)
        self.path_model = os.path.join(self.fold_path + 'model/')
        self.path_result = os.path.join(self.fold_path)

    def create_paths_subject(self, patient_id):
        self.path_subject_result[patient_id] = self.path_result + patient_id + "/"
        Path(self.path_subject_result[patient_id]).mkdir(parents=True, exist_ok=True)

    def update_path(self, time_index):
        self.path_model_updated, self.path_result_updated = [], []
        for idx in range(len(self.path_result)):
            self.path_model_updated.append(self.path_result[idx] + '/t_{}/model/'.format(time_index))
            self.path_result_updated.append(self.path_result[idx] + '/t_{}/'.format(time_index))
            Path(self.path_model_updated[idx]).mkdir(parents=True, exist_ok=True)
            Path(self.path_result_updated[idx]).mkdir(parents=True, exist_ok=True)
