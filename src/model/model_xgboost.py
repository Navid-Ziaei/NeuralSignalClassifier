import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import os
from sklearn.metrics import f1_score, classification_report, accuracy_score, precision_score, recall_score
import numpy as np
import xgboost as xgb

import json
from imblearn.over_sampling import SMOTE



def train_xgb(data_train, labels_train, data_test, labels_test, paths, balance_method='weighting',
              selected_features=None, target_columns=None):
    save_path = paths.path_result + '/xgb/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    # Create and train the XGBoost model with class weights

    if len(np.unique(labels_train)) > 2:
        model = xgb.XGBClassifier(objective="multi:softmax", num_class=2)
    else:
        scale_pos_weight = 1
        if balance_method == 'smote':
            smote = SMOTE(random_state=42)
            data_train, labels_train = smote.fit_resample(data_train, labels_train)
        elif balance_method == 'weighting':
            scale_pos_weight = 2 * np.sum(1 - labels_train) / np.sum(labels_train)

        model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight,
                                  max_depth=6,
                                  num_parallel_tree=2)

    model.fit(data_train, labels_train)
    if selected_features is not None:
        key_list = list(selected_features.keys())
        feature_list = selected_features[key_list[0]]
        feature_importance = {feature_list[i]: model.feature_importances_[i] for i in
                              range(len(feature_list))}
        print("top 10 important features are: ",
              sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])

    # Make predictions
    predictions = model.predict(data_test)

    if len(target_columns) == 1:
        target_names = [f'{target_columns[0]}_1', f'{target_columns[0]}_1']
    else:
        target_names = target_columns
    report = classification_report(y_true=labels_test, y_pred=predictions, target_names=target_names)
    print(report)

    report_results = classification_report(y_true=labels_test, y_pred=predictions, output_dict=True,
                                           target_names=target_names)

    metrics = {
        'accuracy': accuracy_score(labels_test, predictions),
        'precision': precision_score(labels_test, predictions, average='weighted'),
        'recall': recall_score(labels_test, predictions, average='weighted'),
        'f1_score': f1_score(labels_test, predictions, average='weighted')
    }

    with open(save_path + 'xgb_classification_report.txt', "w") as file:
        file.write(report)

    with open(save_path + 'xgb_classification_result.json', "w") as file:
        json.dump(metrics, file, indent=2)

    return metrics, report_results


def optimize_xgboost_hyperparameters(X, y):
    # Define the hyperparameter grid to search
    param_grid = {
        'n_estimators': [100, 200, 300],  # Number of boosting rounds
        'max_depth': [3, 4, 5, 6, 7],  # Maximum depth of trees
        'learning_rate': [0.01, 0.1, 0.2, 0.3],  # Learning rate
        'subsample': [0.7, 0.8, 0.9],  # Fraction of samples used for training each tree
        'colsample_bytree': [0.7, 0.8, 0.9],  # Fraction of features used for training each tree
        'min_child_weight': [1, 2, 3],  # Minimum sum of instance weight (hessian) needed in a child
        'gamma': [0, 0.1, 0.2, 0.3],  # Minimum loss reduction required to make a further partition on a leaf node
        'reg_lambda': [0, 1, 2],  # L2 regularization term on weights
        'reg_alpha': [0, 1, 2]  # L1 regularization term on weights
    }

    # Create the XGBoost classifier
    xgb_classifier = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss")

    # Use RandomizedSearchCV for hyperparameter optimization with cross-validation
    randomized_search = RandomizedSearchCV(
        xgb_classifier, param_distributions=param_grid, n_iter=50,
        scoring="accuracy", cv=5, verbose=2, n_jobs=-1, random_state=42
    )

    # Fit the randomized search to the data
    randomized_search.fit(X, y)

    # Get the best hyperparameters
    best_params = randomized_search.best_params_
    best_score = randomized_search.best_score_

    return best_params, best_score