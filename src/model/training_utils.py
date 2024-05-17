import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def leave_one_patient_out_evaluation(model, features_matrix, labels_array, patients_ids):
    unique_pids = np.unique(patients_ids)
    f1_scores = []

    for pid_value in unique_pids:
        train_mask = (patients_ids != pid_value)
        test_mask = (patients_ids == pid_value)
        X_train = features_matrix[train_mask]
        y_train = labels_array[train_mask]
        X_test_patient = features_matrix[test_mask]
        y_test_patient = labels_array[test_mask]

        model.fit(X_train, y_train)
        y_pred_patient = model.predict(X_test_patient)
        f1_score_patient = f1_score(y_test_patient, y_pred_patient, average='weighted')
        f1_scores.append(f1_score_patient)

    return f1_scores


def patient_based_split_evaluation(model, features_matrix, labels_array, patients_ids, test_size=0.2):
    f1_scores = []

    for _ in range(10):  # Repeat 10 times for stable results
        X_train, X_test, y_train, y_test, _, patients_ids_test = train_test_split(
            features_matrix, labels_array, patients_ids, test_size=test_size, random_state=None)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1_score_patient = f1_score(y_test, y_pred, average='weighted')
        f1_scores.append(f1_score_patient)

    return f1_scores


def individual_patient_evaluation(model, features_matrix, labels_array, patients_ids, train_ratio=0.8):
    unique_pids = np.unique(patients_ids)
    f1_scores = []

    for pid_value in unique_pids:
        mask = (patients_ids == pid_value)
        X_patient = features_matrix[mask]
        y_patient = labels_array[mask]

        num_train_samples = int(len(X_patient) * train_ratio)
        X_train = X_patient[:num_train_samples]
        y_train = y_patient[:num_train_samples]
        X_test_patient = X_patient[num_train_samples:]
        y_test_patient = y_patient[num_train_samples:]

        model.fit(X_train, y_train)
        y_pred_patient = model.predict(X_test_patient)
        f1_score_patient = f1_score(y_test_patient, y_pred_patient, average='weighted')
        f1_scores.append(f1_score_patient)

    return f1_scores
