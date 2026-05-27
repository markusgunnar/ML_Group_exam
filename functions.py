import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report,
                             make_scorer,
                             f1_score,
                             average_precision_score,
                             precision_score,
                             recall_score,
                             confusion_matrix)
from sklearn.dummy import DummyClassifier

r_state = 42


def evaluate_model(_name, _model, _x_train, _y_train, _x_val, _y_val, _cv_strat):

    # Train model
    _model.fit(_x_train, _y_train)

    # Validation predictions
    y_pred = _model.predict(_x_val)
    y_proba = _model.predict_proba(_x_val)[:, 1]

    # Validation metrics
    f1_val = f1_score(_y_val, y_pred)
    pr_auc_val = average_precision_score(_y_val, y_proba)

    # Cross-validation metrics
    cv_f1 = cross_val_score(
        _model,
        _x_train,
        _y_train,
        cv=_cv_strat,
        scoring="f1"
    ).mean()

    cv_pr = cross_val_score(
        _model,
        _x_train,
        _y_train,
        cv=_cv_strat,
        scoring="average_precision"
    ).mean()

    return {
        "Model": _name,
        "F1_val": f1_val,
        "PR_AUC_val": pr_auc_val,
        "F1_CV": cv_f1,
        "PR_AUC_CV": cv_pr
    }


def transform_preprocess(_num_features, _binary_features, _cat_features):
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    binary_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value="Unknown"
        )),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, _num_features),
        ("binary", binary_transformer, _binary_features),
        ("cat", cat_transformer, _cat_features)
    ])

    return preprocessor


def extract_feature_coeff(_pipeline):
    feature_names = _pipeline.named_steps[
        "preprocess"
    ].get_feature_names_out()

    coefficients = _pipeline.named_steps[
        "model"
    ].coef_[0]

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients
    })

    importance["Absolute"] = (
        importance["Coefficient"]
        .abs()
    )

    return importance