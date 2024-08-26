import numpy as np


def get_labels_pilot1(features_df, model_settings):
    binary_column = model_settings.binary_column
    if binary_column == 'old_new':
        labels_array = features_df['old_new'].values
    elif binary_column == 'decision':
        labels_array = features_df['decision'].values
        if len(np.unique(labels_array)) > 2:
            labels_array *= 2
    elif binary_column == 'is_experienced':
        labels_array = process_is_experienced(features_df, model_settings)
    elif binary_column == 'go_nogo':
        labels_array = map_column(features_df, 'go_nogo', {'go': 1, 'nogo': 0})
    elif binary_column == 'h5':
        labels_array = features_df['TargetValue'].values
    else:
        labels_array = features_df[binary_column].values

    return features_df, labels_array


def process_is_experienced(features_df, model_settings):
    mapping = {'ctrl': 2, 'exp': 1, 'noexp': 0}
    data = initialize_block_data()

    for block in features_df['block_number'].unique():
        block_data = extract_block_data(features_df, block)
        data = update_block_data(data, block_data)
        print_block_statistics(block_data)

    features_df = filter_correct_trials(features_df, model_settings)
    labels_array = features_df['is_experienced'].values
    return labels_array


def initialize_block_data():
    return {
        'block': [], 'type': [], 'num_ctl': [], 'num_experienced': [],
        'num_not_experienced': [], 'num_ctl_correct': [],
        'num_experienced_correct': [], 'num_not_experienced_correct': []
    }


def extract_block_data(features_df, block):
    block_data = {}
    block_data['type'] = features_df[features_df['block_number'] == block]['block_type'].unique()
    block_data['num_trials'] = len(features_df[features_df['block_number'] == block])

    block_data['ctl'] = features_df[(features_df['block_number'] == block) & (features_df['stim'] == 'ctl')]
    block_data['experienced'] = features_df[
        (features_df['block_number'] == block) &
        (features_df['stim'] != 'ctl') &
        (features_df['is_experienced'] != True)
        ]
    block_data['notexperienced'] = features_df[
        (features_df['block_number'] == block) &
        (features_df['stim'] != 'ctl') &
        (features_df['is_experienced'] != False)
        ]

    block_data['ctl_correct'] = block_data['ctl'][block_data['ctl']['is_correct']]
    block_data['experienced_correct'] = block_data['experienced'][block_data['experienced']['is_correct']]
    block_data['notexperienced_correct'] = block_data['notexperienced'][block_data['notexperienced']['is_correct']]

    return block_data


def update_block_data(data, block_data):
    data['block'].append(block_data['block'])
    data['type'].append(block_data['type'])
    data['num_ctl'].append(len(block_data['ctl']))
    data['num_experienced'].append(len(block_data['experienced']))
    data['num_not_experienced'].append(len(block_data['notexperienced']))
    data['num_ctl_correct'].append(len(block_data['ctl_correct']))
    data['num_experienced_correct'].append(len(block_data['experienced_correct']))
    data['num_not_experienced_correct'].append(len(block_data['notexperienced_correct']))
    return data


def print_block_statistics(block_data):
    print(
        f"Block {block_data['block']}: type {block_data['type']}\n"
        f"\tNumber of control stims: {len(block_data['ctl'])} "
        f"({100 * len(block_data['ctl']) / block_data['num_trials']}%) "
        f"(Correct: {len(block_data['ctl_correct'])} "
        f"({100 * len(block_data['ctl_correct']) / len(block_data['ctl'])}%))\n"
        f"\tNumber of experienced stims: {len(block_data['experienced'])} "
        f"({100 * len(block_data['experienced']) / block_data['num_trials']}%) "
        f"(Correct: {len(block_data['experienced_correct'])} "
        f"({100 * len(block_data['experienced_correct']) / len(block_data['experienced'])}%))\n"
        f"\tNumber of not experienced stims: {len(block_data['notexperienced'])} "
        f"({100 * len(block_data['notexperienced']) / block_data['num_trials']}%) "
        f"(Correct: {len(block_data['notexperienced_correct'])} "
        f"({100 * len(block_data['notexperienced_correct']) / len(block_data['notexperienced'])}%))"
    )


def filter_correct_trials(features_df, model_settings):
    correct_trials = features_df[features_df['is_correct'] == True]
    features_df = correct_trials[
        ((correct_trials['block_type'] == f"{model_settings['stim']}+e") |
         (correct_trials['block_type'] == f"{model_settings['stim']}-e") |
         (correct_trials['block_type'] == f"{model_settings['stim']}+e+x") |
         (correct_trials['block_type'] == f"{model_settings['stim']}-e+x")) &
        (correct_trials['stim'] != 'ctl')
        ]
    return features_df

def map_column(df, column, mapping):
    df['exp_label'] = df[column].map(mapping)
    return df['exp_label'].values


def get_labels_verbmem(features_df, settings):
    return features_df[settings.binary_column].values
