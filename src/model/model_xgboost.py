import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV


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