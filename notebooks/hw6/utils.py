import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
)

from app.antifraud import AntifraudService


logger = logging.getLogger(__name__)


def load_application(db) -> pd.DataFrame:
    """
    Загружает таблицу application из БД
    """
    df = db.get_df("SELECT * FROM application")
    df.columns = df.columns.str.upper()
    return df


def load_bureau(db) -> pd.DataFrame:
    """
    Загружает bureau из БД
    """
    query = """
    SELECT
        SK_ID_CURR,
        CREDIT_ACTIVE,
        DAYS_CREDIT
    FROM bureau
    """
    df = db.get_df(query)
    df.columns = df.columns.str.upper()
    return df


def load_previous_application(db) -> pd.DataFrame:
    """
    Загружает previous_application из БД
    """
    query = """
    SELECT
        SK_ID_CURR
    FROM previous_application
    """
    df = db.get_df(query)
    df.columns = df.columns.str.upper()
    return df


def build_hw22_features(
    application_df: pd.DataFrame,
    bureau_df: pd.DataFrame,
    previous_application_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Строит 5 признаков из hw2
    """
    features_app = application_df[
        ["SK_ID_CURR", "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY"]
    ].copy()

    income_nonzero = features_app["AMT_INCOME_TOTAL"].replace(0, np.nan)

    features_app["credit_to_income_ratio"] = (
        features_app["AMT_CREDIT"] / income_nonzero
    ).round(4)

    features_app["annuity_to_income_ratio"] = (
        features_app["AMT_ANNUITY"] / income_nonzero
    ).round(4)

    features_app = features_app[
        ["SK_ID_CURR", "credit_to_income_ratio", "annuity_to_income_ratio"]
    ]

    active_credit_count = (
        bureau_df.loc[bureau_df["CREDIT_ACTIVE"] == "Active", ["SK_ID_CURR"]]
        .groupby("SK_ID_CURR", as_index=False)
        .size()
        .rename(columns={"size": "active_credit_count"})
    )

    credit_history_days = (
        bureau_df.groupby("SK_ID_CURR", as_index=False)["DAYS_CREDIT"]
        .min()
        .rename(columns={"DAYS_CREDIT": "credit_history_days"})
    )
    credit_history_days["credit_history_days"] = (
        credit_history_days["credit_history_days"].abs()
    )

    prev_application_count = (
        previous_application_df.groupby("SK_ID_CURR", as_index=False)
        .size()
        .rename(columns={"size": "prev_application_count"})
    )

    features = features_app.merge(active_credit_count, on="SK_ID_CURR", how="left")
    features = features.merge(credit_history_days, on="SK_ID_CURR", how="left")
    features = features.merge(prev_application_count, on="SK_ID_CURR", how="left")

    fill_zero_cols = [
        "active_credit_count",
        "credit_history_days",
        "prev_application_count",
    ]
    features[fill_zero_cols] = features[fill_zero_cols].fillna(0)

    return features


def prepare_dataset(
    application_df: pd.DataFrame,
    bureau_df: pd.DataFrame,
    previous_application_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Собирает train:
    добавляет признаки из HW2
    оставляет только строки с DATASET_SOURCE='train'
    формирует матрицу признаков X и целевой вектор 
    """
    hw22_features = build_hw22_features(
        application_df=application_df,
        bureau_df=bureau_df,
        previous_application_df=previous_application_df,
    )

    full_df = application_df.merge(hw22_features, on="SK_ID_CURR", how="left")

    train_df = full_df[full_df["DATASET_SOURCE"].eq("train")].copy()
    antifraud_service = AntifraudService()
    antifraud_result = train_df.apply(antifraud_service.check, axis=1)
    accepted_mask = antifraud_result.eq(AntifraudService.ACCEPTED)

    removed_rows = int((~accepted_mask).sum())
    logger.info("Антифрод-фильтр исключил строк из обучения: %d", removed_rows)

    train_df = train_df.loc[accepted_mask].copy()

    y = train_df["TARGET"].astype(int)
    X = train_df[existing_columns].copy()

    selected_columns = [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "DAYS_LAST_PHONE_CHANGE",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "OBS_30_CNT_SOCIAL_CIRCLE",
        "DEF_30_CNT_SOCIAL_CIRCLE",
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
        "CODE_GENDER",
        "NAME_CONTRACT_TYPE",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE",
        "ORGANIZATION_TYPE",
        "credit_to_income_ratio",
        "annuity_to_income_ratio",
        "active_credit_count",
        "credit_history_days",
        "prev_application_count",
    ]

    existing_columns = [col for col in selected_columns if col in train_df.columns]
    missing_columns = sorted(set(selected_columns) - set(existing_columns))

    if missing_columns:
        logger.warning("Отсутствуют признаки: %s", missing_columns)

    X = train_df[existing_columns].copy()

    logger.info("Финальная матрица признаков: %s", X.shape)
    logger.info("Количество выбранных признаков: %d", len(existing_columns))

    return X, y


def get_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Разделяет признаки на числовые и категориальные.
    """
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessors(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[ColumnTransformer, ColumnTransformer]:
    """
    Возвращает два препроцессора:
    для линейной модели
    для моделей деревка, бустинга
    """
    linear_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    tree_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    return linear_preprocessor, tree_preprocessor


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Считает остановные метрики на holdout test
    """
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_score),
        "pr_auc": average_precision_score(y_test, y_score),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    return metrics, y_score


def save_pickle(model, output_path: Path) -> None:
    """
    Сохраняет модель в pickle
    """
    with open(output_path, "wb") as file:
        pickle.dump(model, file)


def plot_roc_curves(roc_data: dict, y_test: pd.Series, output_path: Path) -> None:
    """
    Сохраняет ROC-кривые всех моделей 
    """
    plt.figure(figsize=(10, 7))

    for model_name, y_score in roc_data.items():
        RocCurveDisplay.from_predictions(
            y_test,
            y_score,
            name=model_name,
            ax=plt.gca(),
        )

    plt.title("ROC-кривые на отложенной тестовой выборке")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrices(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
) -> None:
    """
    Сохраняет матрицы ошибок всех моделей 
    """
    n_models = len(models)
    n_cols = 2
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (model_name, model) in zip(axes, models.items()):
        ConfusionMatrixDisplay.from_estimator(
            model,
            X_test,
            y_test,
            ax=ax,
            cmap="Blues",
            colorbar=False,
        )
        ax.set_title(f"Матрица ошибок: {model_name}")

    for ax in axes[n_models:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
