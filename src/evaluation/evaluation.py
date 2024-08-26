import pandas as pd


class ResultList():
    def __init__(self, metric_list, method_list):
        self.result_list = {
            'subject id': [],
            'filename': []
        }

        for method in method_list:
            for metric in metric_list:
                self.result_list[method + ' avg_' + metric] = []
                self.result_list[method + ' std_' + metric] = []

    def add_result(self, method, metric, avg, std):
        self.result_list[method + ' avg_' + metric].append(avg)
        self.result_list[method + ' std_' + metric].append(std)

    def add_subject(self, unique_pids, patients_files):
        self.result_list['subject id'].append(unique_pids)
        self.result_list['filename'].append('_'.join(patients_files.split('.')[:-1]))

    def update_result(self, method, metric, avg_score, std_score):
        self.result_list[method + ' avg_' + metric].append(avg_score)
        self.result_list[method + ' std_' + metric].append(std_score)

    def to_dataframe(self):
        return pd.DataFrame(self.result_list)
