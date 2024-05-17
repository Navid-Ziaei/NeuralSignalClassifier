from src.experiments.utils import *

path_results = "D:\\Navid\\Projects2\\lexical_memory_task_eeg\\results\\debug\\"
save_path = "D:\\Navid\\Projects2\\lexical_memory_task_eeg\\results\\"
folder_list = os.listdir(path_results)
p_value_thresh = 0.3
feature_list = ['coh_features_vec', 'coh_features', 'freq_features', 'time_features']

#analyze_p_value(feature_list, path_results, p_value_thresh, save_path, target='rec_old')
#analyze_p_value(feature_list, path_results, p_value_thresh, save_path, target='decision')

analyze_line_fit_p_value(feature_list, path_results, p_value_thresh, save_path, target='rec_old')
analyze_line_fit_p_value(feature_list, path_results, p_value_thresh, save_path, target='x_new')


