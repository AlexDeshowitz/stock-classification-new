"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.5
"""

from typing import Dict, Tuple

import pandas as pd
import numpy as np
import importlib
from sklearn.model_selection import train_test_split, StratifiedKFold

# visualization tools:
import matplotlib.pyplot as plt
import seaborn as sns

# models:
from xgboost import XGBClassifier

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator

# evaluation functions:
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix




def custom_recall_score(confusion_matrix: np.array) -> np.int64 :

    recall_value = confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[1,0])

    return recall_value


def custom_precision_score(confusion_matrix: np.array) -> np.int64 : 

    precision_value = confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[0,1])

    return precision_value


def extract_feature_importances(X: np.array, model: BaseEstimator) -> pd.DataFrame:
    
    #TODO: Add documentation

    # Extract feature importances: 
    feature_importances = model.feature_importances_
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    feature_importance_df = pd.DataFrame({
                                          'Feature': feature_names,
                                          'Importance': feature_importances
                                        }
                                        ).sort_values(by='Importance', ascending=False)


    return feature_importance_df


def dynamic_import(class_path: str):
    """Dynamically imports a class from its full path."""
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def train_models(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, parameters: dict) -> pd.DataFrame:
    
    '''WRITE DOCUMENTATION'''
    
    # Convert y_train to a 1D array if it's a DataFrame
    y_train = y_train.iloc[:, 0].values if isinstance(y_train, pd.DataFrame) else y_train.values
    
    # Store feature names from the DataFrame
    feature_names = X_train.columns
    
    # Initialize metrics storage
    model, classifier_details, fold = [], [], []
    train_precisions, test_precisions = [], []
    train_recalls, test_recalls = [], []
    train_f_scores, test_f_scores = [], []
    train_accuracies, test_accuracies = [], []
    train_true_positives, test_true_positives = [], []
    train_true_negatives, test_true_negatives = [], []
    train_false_positives, test_false_positives = [], []
    train_false_negatives, test_false_negatives = [], []

    feature_importances_df = pd.DataFrame()

    # Iterate through classifiers specified in the parameters
    for clf_name, clf_info in parameters['classifiers'].items():
        # Dynamically import and instantiate classifier
        clf_class = dynamic_import(clf_info['class_path'])
        clf_params = clf_info['params']
        clf = clf_class(**clf_params)  # Initialize classifier with params

        print(f'Training: {clf_name} classifier')

        # Cross-validation loop
        cv = StratifiedKFold(n_splits=parameters['cross_val_splits'], shuffle=True, random_state=parameters['seed']).split(X_train, y_train)

        for k, (fold_train, fold_test) in enumerate(cv):
            clf.fit(X_train.iloc[fold_train].values, y_train[fold_train])
            
            # Predictions
            train_pred = clf.predict(X_train.iloc[fold_train].values)
            test_pred = clf.predict(X_train.iloc[fold_test].values)

            # Confusion matrices
            train_confusion_matrix = confusion_matrix(y_train[fold_train], train_pred)
            test_confusion_matrix = confusion_matrix(y_train[fold_test], test_pred)

            # Accuracy
            train_accuracy = clf.score(X_train.iloc[fold_train].values, y_train[fold_train])
            test_accuracy = clf.score(X_train.iloc[fold_test].values, y_train[fold_test])

            # Precision
            train_precision = precision_score(y_train[fold_train], train_pred)
            test_precision = precision_score(y_train[fold_test], test_pred)

            # Recall
            train_recall = recall_score(y_train[fold_train], train_pred)
            test_recall = recall_score(y_train[fold_test], test_pred)

            # F1 Score
            train_f = f1_score(y_train[fold_train], train_pred)
            test_f = f1_score(y_train[fold_test], test_pred)

            # True/False Positives/Negatives
            train_tp = train_confusion_matrix[1,1]
            test_tp = test_confusion_matrix[1,1]
            train_tn = train_confusion_matrix[0,0]
            test_tn = test_confusion_matrix[0,0]
            train_fp = train_confusion_matrix[0,1]
            test_fp = test_confusion_matrix[0,1]
            train_fn = train_confusion_matrix[1,0]
            test_fn = test_confusion_matrix[1,0]

            # Append metrics
            model.append(clf_name)
            classifier_details.append(clf)
            fold.append(k)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            train_precisions.append(train_precision)
            test_precisions.append(test_precision)
            train_recalls.append(train_recall)
            test_recalls.append(test_recall)
            train_f_scores.append(train_f)
            test_f_scores.append(test_f)
            train_true_positives.append(train_tp)
            test_true_positives.append(test_tp)
            train_true_negatives.append(train_tn)
            test_true_negatives.append(test_tn)
            train_false_positives.append(train_fp)
            test_false_positives.append(test_fp)
            train_false_negatives.append(train_fn)
            test_false_negatives.append(test_fn)

            # Extract feature importances or coefficients
            if hasattr(clf, "coef_"):  # For LogisticRegression
                feature_importances = clf.coef_.flatten()
                temp_df = pd.DataFrame({
                    "model": [clf_name]*len(feature_importances),
                    "fold": [k]*len(feature_importances),
                    "feature": feature_names,
                    "importance": feature_importances
                })
                feature_importances_df = pd.concat([feature_importances_df, temp_df], ignore_index=True)
            
            elif hasattr(clf, "feature_importances_"):  # For RandomForest, XGBoost
                feature_importances = clf.feature_importances_
                temp_df = pd.DataFrame({
                    "model": [clf_name]*len(feature_importances),
                    "fold": [k]*len(feature_importances),
                    "feature": feature_names,
                    "importance": feature_importances
                })
                feature_importances_df = pd.concat([feature_importances_df, temp_df], ignore_index=True)

    # Summary of feature importances
    feature_importances_summary_df = feature_importances_df.groupby(['model', 'feature']).agg(
        importance_value=('importance', 'mean')).reset_index()

    # Detailed results DataFrame
    detailed_results_df = pd.DataFrame({
        "model": model,
        "classifier_details": classifier_details,
        "fold": fold,
        "train_accuracy": train_accuracies,
        "test_accuracy": test_accuracies,
        "train_precision": train_precisions,
        "test_precision": test_precisions,
        "train_recall": train_recalls,
        "test_recall": test_recalls,
        "train_true_positives": train_true_positives,
        "test_true_positives": test_true_positives,
        "train_true_negatives": train_true_negatives,
        "test_true_negatives": test_true_negatives,
        "train_false_positives": train_false_positives,
        "test_false_positives": test_false_positives,
        "train_false_negatives": train_false_negatives,
        "test_false_negatives": test_false_negatives
    })

    # Aggregated results
    results_df = detailed_results_df.groupby('model').agg({
        'train_accuracy': 'mean',
        'test_accuracy': 'mean',
        'train_precision': 'mean',
        'test_precision': 'mean',
        'train_recall': 'mean',
        'test_recall': 'mean',
        'train_true_positives': 'sum',
        'test_true_positives': 'sum',
        'train_true_negatives': 'sum',
        'test_true_negatives': 'sum',
        'train_false_positives': 'sum',
        'test_false_positives': 'sum',
        'train_false_negatives': 'sum',
        'test_false_negatives': 'sum'
    }).reset_index()

    # Add in overall positve-class rates to the datasets:
    results_df['train_positive_rate'] = (results_df['train_true_positives'] + results_df['train_false_negatives']) / (results_df['train_true_positives'] + results_df['train_false_negatives'] + results_df['train_false_positives'] + results_df['train_true_negatives'])
    results_df['test_positive_rate'] = (results_df['test_true_positives'] + results_df['test_false_negatives']) / (results_df['test_true_positives'] + results_df['test_false_negatives'] + results_df['test_false_positives'] + results_df['test_true_negatives'])

    # share message with complete:
    print('training evaluation completed')
    return detailed_results_df, results_df, feature_importances_df, feature_importances_summary_df


def select_champion_model(models: pd.DataFrame, parameters: str) -> pd.DataFrame:
    """
    Selects the champion model based on the highest or lowest value of a given optimization target.
    
    Args:
    - models: A DataFrame containing model performance metrics.
    - optimization_target: The metric by which to select the champion model (e.g., 'test_precision', 'test_recall', 'test_false_positives').
    
    Returns:
    - champion_model: The row corresponding to the selected champion model.
    """
    
    if parameters['optimization_target'] not in models.columns:
        raise ValueError(f"'{parameters['optimization_target'] }' not found in model metrics. Choose from: {list(models.columns)}")
    
    # Determine whether we want the highest or lowest value
    # Assuming metrics like 'precision', 'recall', 'accuracy' need the highest, and 'false positives', 'false negatives', etc. need the lowest
    if 'precision' in parameters['optimization_target']  or 'recall' in parameters['optimization_target']  or 'accuracy' in parameters['optimization_target'] :
        ascending = False  # We want to maximize these metrics
   
    else:
        ascending = True  # For counts (e.g., false positives), we want the minimum value
    
    # Sort the models based on the optimization target
    sorted_models = models.sort_values(by=parameters['optimization_target'] , ascending=ascending)
    
    # Return the top model (first row after sorting)
    champion_model = pd.DataFrame(sorted_models.iloc[0]).reset_index()

    champion_model.columns = ['element', 'value']
    

    return champion_model