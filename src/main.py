import pandas as pd

from src.settings import Paths, Settings
from src.data_loader import VerbMemEEGDataLoader, PilotEEGDataLoader, CLEARDataLoader
from src.feature_extraction import FeatureExtractor
from src.data_preprocess import DataPreprocessor
from src.visualization.feature_visualizer import FeatureVisualizer
from src.visualization.model_result_visualizer import *

from src.model.Bayesian_JointGPLVM import *
from src.model.utils.kernels import ARDRBFKernel
from src.visualization.EEG_vissualizer import visualize_erp

from gpytorch.likelihoods import GaussianLikelihood, BernoulliLikelihood

import numpy as np
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Load settings from settings.json
settings = Settings()  # Initialize settings object
settings.load_settings()  # Load settings from a JSON file

# Set up paths for data
paths = Paths(settings)  # Initialize paths object with loaded settings
paths.load_device_paths()  # Load device-specific paths
paths.create_paths()  # Create any necessary file paths

# Load EEG dataset using configured settings and paths
if settings.dataset == 'pilot01':
    dataset = PilotEEGDataLoader(paths=paths, settings=settings)  # Initialize EEG dataset loader
elif settings.dataset == 'verbmem':
    dataset = VerbMemEEGDataLoader(paths=paths, settings=settings)  # Initialize EEG dataset loader
elif settings.dataset == 'clear':
    dataset = CLEARDataLoader(paths=paths, settings=settings)  # Initialize EEG dataset loader

dataset.load_data(patient_ids=None)  # Load EEG data for specified patients

# Preprocess the loaded dataset
data_preprocessor = DataPreprocessor(paths=paths, settings=settings)  # Initialize data preprocessor
dataset = data_preprocessor.preprocess(dataset)  # Apply preprocessing steps to the dataset

"""
for idx in range(len(list(dataset.all_patient_data.keys()))):
    single_patient_data = dataset.all_patient_data[list(dataset.all_patient_data.keys())[idx]]
    visualize_erp(patient_data=single_patient_data, channel_idx=45,
                  label=single_patient_data.labels['is_experienced'], label_names=['Not Experienced', 'Experienced'])
    visualize_erp(patient_data=single_patient_data, channel_idx=45,
                  label=single_patient_data.labels['go_nogo'], label_names=['NoGo', 'Go'])
"""

# Extract features from the preprocessed dataset
feature_extractor = FeatureExtractor(paths=paths, settings=settings)  # Initialize feature extractor
feature_extractor.extract_features(dataset)  # Extract relevant features from the dataset

# Prepare your labels and features (including response time) for training
features_matrix, labels, patients_ids, features_list_name = feature_extractor.get_feature_array(dataset)

labels_array = np.array(labels[list(labels.keys())[1]])
y_one_hot = np.zeros((labels_array.shape[0], len(np.unique(labels_array))))
y_one_hot[np.arange(labels_array.shape[0]), np.uint(labels_array - 1)] = 1

unique_pids = np.unique(patients_ids)

X_train, X_test, labels_train, labels_test, y_train, y_test, pid_train, pid_test = \
    train_test_split(features_matrix, labels_array - 1, y_one_hot, patients_ids, test_size=0.2, random_state=42)

model_settings = {
    'data_dim': X_train.shape[-1],
    'latent_dim': 20,
    'num_inducing_points': 30,
    'num_epochs_train': 200,
    'num_epochs_test': 200,
    'batch_size': 100,
    'load_trained_model': False,
    'use_gpytorch': False
}

batch_shape = torch.Size([model_settings['data_dim']])
if model_settings['use_gpytorch'] is False:
    kernel_cls = ARDRBFKernel(input_dim=model_settings['latent_dim'])
    kernel_reg = ARDRBFKernel(input_dim=model_settings['latent_dim'])
else:
    kernel_reg = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=model_settings['latent_dim']))
    kernel_cls = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=model_settings['latent_dim']))

likelihood_reg = GaussianLikelihood(batch_shape=batch_shape)
likelihood_cls = BernoulliLikelihood()

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train_onehot = torch.tensor(y_train)
y_test_onehot = torch.tensor(y_test)

model = JointGPLVM_Bayesian(torch.tensor(X_train, dtype=torch.float32),
                            kernel_reg=kernel_reg,
                            kernel_cls=kernel_cls,
                            num_classes=y_train_onehot.shape[-1],
                            latent_dim=model_settings['latent_dim'],
                            num_inducing_points=model_settings['num_inducing_points'],
                            likelihood_reg=likelihood_reg,
                            likelihood_cls=likelihood_cls,
                            use_gpytorch=model_settings['use_gpytorch'])

if model_settings['load_trained_model'] is False:
    losses = model.train_model(yn=X_train, ys=y_train_onehot,
                               epochs=model_settings['num_epochs_train'],
                               batch_size=model_settings['batch_size'])
    model.save_wights(path_save=paths.path_model[0])

    with open(paths.path_model[0] + 'model_settings.json', 'w') as f:
        json.dump(model_settings, f, indent=2)
else:
    losses = []
    model.load_weights()

predictions, metrics = model.evaluate(yn_test=X_test, ys_test=labels_test,
                                      epochs=model_settings['num_epochs_test'], save_path=paths.path_result)

if model_settings['use_gpytorch'] is False:
    alpha_reg = model.kernel_reg.alpha.detach().numpy()
    alpha_cls = model.kernel_cls.alpha.detach().numpy()
    X = model.x.q_mu.detach().numpy()
    std = model.x.q_sigma.detach().numpy()
else:
    alpha_reg = 1 / model.kernel_reg.base_kernel.lengthscale.detach().numpy()
    alpha_cls = 1 / model.kernel_cls.base_kernel.lengthscale.detach().numpy()
    X = model.x.q_mu.detach().numpy()
    std = torch.nn.functional.softplus(model.x.q_log_sigma).detach().numpy()

plot_results_gplvm(X, std, labels=labels_train, losses=losses, inverse_length_scale=alpha_reg,
                   latent_dim=model_settings['latent_dim'],
                   save_path=paths.path_result, file_name=f'gplvm_train_reg_result_all')
plot_results_gplvm(X, std, labels=labels_train, losses=losses, inverse_length_scale=alpha_cls,
                   latent_dim=model_settings['latent_dim'],
                   save_path=paths.path_result, file_name=f'gplvm_train_cls_result_all')

X_test = model.x_test.q_mu.detach().numpy()
std_test = model.x_test.q_sigma.detach().numpy()
plot_results_gplvm(X_test, std_test, labels=labels_test, losses=losses, inverse_length_scale=alpha_cls,
                   latent_dim=model_settings['latent_dim'],
                   save_path=paths.path_result, file_name=f'gplvm_test_result_all')

plt.show()

f1_scores = []

for pid_value in unique_pids:
    print(f"subject{pid_value} from {len(unique_pids)}")
    # Create a mask to exclude the current patient from the training data
    train_mask = (patients_ids != pid_value)

    # Select training data for all patients except the current patient
    X_train = features_matrix[train_mask]
    y_train = labels_array[train_mask] - 1
    y_train_onehot = y_one_hot[train_mask]

    # Select test data for the current patient
    test_mask = (patients_ids == pid_value)
    X_test_patient = features_matrix[test_mask]
    y_test_patient = labels_array[test_mask] - 1
    y_test_patient_onehot = y_one_hot[test_mask]

    model_settings = {
        'data_dim': X_train.shape[-1],
        'latent_dim': 20,
        'num_inducing_points': 30,
        'num_epochs_train': 200,
        'num_epochs_test': 200,
        'batch_size': 100,
        'load_trained_model': False,
        'use_gpytorch': False
    }

    batch_shape = torch.Size([model_settings['data_dim']])
    if model_settings['use_gpytorch'] is False:
        kernel_cls = ARDRBFKernel(input_dim=model_settings['latent_dim'])
        kernel_reg = ARDRBFKernel(input_dim=model_settings['latent_dim'])
    else:
        kernel_reg = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=model_settings['latent_dim']))
        kernel_cls = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=model_settings['latent_dim']))

    likelihood_reg = GaussianLikelihood(batch_shape=batch_shape)
    likelihood_cls = BernoulliLikelihood()

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test_patient, dtype=torch.float32)
    y_train_onehot = torch.tensor(y_train_onehot)
    y_test_onehot = torch.tensor(y_test_patient_onehot)

    model = JointGPLVM_Bayesian(torch.tensor(X_train, dtype=torch.float32),
                                kernel_reg=kernel_reg,
                                kernel_cls=kernel_cls,
                                num_classes=y_train_onehot.shape[-1],
                                latent_dim=model_settings['latent_dim'],
                                num_inducing_points=model_settings['num_inducing_points'],
                                likelihood_reg=likelihood_reg,
                                likelihood_cls=likelihood_cls,
                                use_gpytorch=model_settings['use_gpytorch'])

    if model_settings['load_trained_model'] is False:
        losses = model.train_model(yn=X_train, ys=y_train_onehot,
                                   epochs=model_settings['num_epochs_train'],
                                   batch_size=model_settings['batch_size'])
        model.save_wights(path_save=paths.path_model[0])

        with open(paths.path_model[0] + 'model_settings.json', 'w') as f:
            json.dump(model_settings, f, indent=2)
    else:
        losses = []
        model.load_weights()

    predictions, metrics = model.evaluate(yn_test=X_test, ys_test=y_test_patient,
                                          epochs=model_settings['num_epochs_test'],
                                          save_path=paths.path_result + f"p{pid_value}")

    if model_settings['use_gpytorch'] is False:
        alpha_reg = model.kernel_reg.alpha.detach().numpy()
        alpha_cls = model.kernel_cls.alpha.detach().numpy()
        X = model.x.q_mu.detach().numpy()
        std = model.x.q_sigma.detach().numpy()
    else:
        alpha_reg = 1 / model.kernel_reg.base_kernel.lengthscale.detach().numpy()
        alpha_cls = 1 / model.kernel_cls.base_kernel.lengthscale.detach().numpy()
        X = model.x.q_mu.detach().numpy()
        std = torch.nn.functional.softplus(model.x.q_log_sigma).detach().numpy()

    plot_results_gplvm(X, std, labels=y_train, losses=losses, inverse_length_scale=alpha_reg,
                       latent_dim=model_settings['latent_dim'],
                       save_path=paths.path_result, file_name=f'gplvm_train_reg_result_{pid_value}')
    plot_results_gplvm(X, std, labels=y_train, losses=losses, inverse_length_scale=alpha_cls,
                       latent_dim=model_settings['latent_dim'],
                       save_path=paths.path_result, file_name=f'gplvm_train_cls_result_{pid_value}')

    X_test = model.x_test.q_mu.detach().numpy()
    std_test = model.x_test.q_sigma.detach().numpy()
    plot_results_gplvm(X_test, std_test, labels=y_test_patient, losses=losses, inverse_length_scale=alpha_cls,
                       latent_dim=model_settings['latent_dim'],
                       save_path=paths.path_result, file_name=f'gplvm_test_result_{pid_value}')

    f1_scores.append(metrics['f1_score'])

    """
    class_weights = len(y_train) / (2 * np.bincount(y_train))
    model = xgb.XGBClassifier(objective="multi:softmax", num_class=2, scale_pos_weight=class_weights)
    model.fit(X_train, y_train)
    
    # Predict on the test data for the current patient
    y_pred_patient = model.predict(X_test_patient)

    # Calculate F1-score for the current patient
    f1_score_patient = f1_score(y_test_patient, y_pred_patient, average='weighted')
    f1_scores.append(f1_score_patient)
    print(f"{f1_score_patient}")
    
    """

result_box_plot(result=f1_scores, metric='F1-Score')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, pid_train, pid_test = train_test_split(features_matrix, labels_array - 1,
                                                                         patients_ids, test_size=0.2, random_state=42)

# Create and train the XGBoost model with class weights
class_weights = len(y_train) / (2 * np.bincount(y_train))
model = xgb.XGBClassifier(objective="multi:softmax", num_class=2, scale_pos_weight=class_weights)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the F1-score
f1 = f1_score(y_test, y_pred, average='macro')
print(f"F1-score: {f1 * 100:.2f}%")

# Get feature importances from the XGBoost model
feature_importances = model.feature_importances_

# Get the indices of the top 10 most important features
top_10_indices = np.argsort(feature_importances)[-10:]

flatten_list = []
for item in features_list_name[1:]:
    flatten_list.extend(item)
# Get the names of the top 10 features
top_10_features = [features_list_name[i] for i in top_10_indices]

# Plot a histogram for the top 10 feature importances
plt.figure(figsize=(10, 6))
plt.barh(range(10), feature_importances[top_10_indices], align='center', color='skyblue')
plt.yticks(range(10), top_10_features)
plt.xlabel('Feature Importance')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()  # Invert the y-axis to show the most important feature at the top
plt.tight_layout()
plt.show()

unique_pids = np.unique(pid_test)
f1_scores = []

for pid_value in unique_pids:
    # Select test data for the current patient
    test_mask = (pid_test == pid_value)
    X_test_patient = X_test[test_mask]
    y_test_patient = y_test[test_mask]

    # Predict on the test data for the current patient
    y_pred_patient = model.predict(X_test_patient)

    # Calculate F1-score for the current patient
    f1_score_patient = f1_score(y_test_patient, y_pred_patient, average='weighted')
    f1_scores.append(f1_score_patient)

result_box_plot(result=f1_scores, metric='F1-Score')
