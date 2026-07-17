import sys
import logging
from pathlib import Path
import argparse
import warnings


warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

import optuna
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from config.db_config import DB_ARGS
from app.utils.db_manager import PostgresDB
from utils import (
    load_application,
    load_bureau,
    load_previous_application,
    prepare_dataset,
    get_feature_types,
    build_preprocessors,
    evaluate_model,
    save_pickle,
    plot_roc_curves,
    plot_confusion_matrices,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


def metrics_at_threshold(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)

    return {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
    }



def find_best_threshold(
    y_true,
    y_score,
    min_precision=0.18,
    min_accuracy=0.70,
):
    thresholds = np.arange(0.01, 0.51, 0.01)

    rows = [
        metrics_at_threshold(y_true, y_score, threshold)
        for threshold in thresholds
    ]

    threshold_df = pd.DataFrame(rows)

    candidates = threshold_df[
        (threshold_df["precision"] >= min_precision) &
        (threshold_df["accuracy"] >= min_accuracy)
    ].copy()

    if candidates.empty:
        candidates = threshold_df.copy()

    best_row = (
        candidates
        .sort_values(["f2", "recall", "f1"], ascending=False)
        .iloc[0]
    )

    return best_row, threshold_df

RANDOM_STATE = 42

logger = logging.getLogger(__name__)


def configure_logging(log_file: Path | None, level: int = logging.INFO) -> None:
    """
    Настраивает логирование в консоль и в файл.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def build_estimator(model_name: str, linear_preprocessor, tree_preprocessor, params: dict):
    """
    Создаёт pipeline для выбранной модели с заданными гиперпараметрами.
    """
    if model_name == "logistic_regression":
        model = LogisticRegression(
            C=params["C"],
            class_weight=params["class_weight"],
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
        return Pipeline(
            steps=[
                ("preprocessor", linear_preprocessor),
                ("model", model),
            ]
        )

    if model_name == "decision_tree":
        model = DecisionTreeClassifier(
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight=params["class_weight"],
            random_state=RANDOM_STATE,
        )
        return Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                ("model", model),
            ]
        )

    if model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            class_weight=params["class_weight"],
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        return Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                ("model", model),
            ]
        )

    if model_name == "gradient_boosting":
        model = HistGradientBoostingClassifier(
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            max_leaf_nodes=params["max_leaf_nodes"],
            min_samples_leaf=params["min_samples_leaf"],
            l2_regularization=params["l2_regularization"],
            max_iter=params["max_iter"],
            random_state=RANDOM_STATE,
        )
        return Pipeline(
            steps=[
                ("preprocessor", tree_preprocessor),
                ("model", model),
            ]
        )

    raise ValueError(f"Неизвестное название модели: {model_name}")


def suggest_params(trial: optuna.Trial, model_name: str) -> dict:
    """
    Пространство поиска Optuna для каждой модели
    """
    if model_name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 1e-2, 10.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }

    if model_name == "decision_tree":
        return {
            "max_depth": trial.suggest_categorical("max_depth", [3, 5, 7, 10, None]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 100),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }

    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 350, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [5, 10, 15, None]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight": trial.suggest_categorical(
                "class_weight",
                [None, "balanced", "balanced_subsample"],
            ),
        }

    if model_name == "gradient_boosting":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15),
            "max_depth": trial.suggest_categorical("max_depth", [3, 5, 7, None]),
            "max_leaf_nodes": trial.suggest_categorical("max_leaf_nodes", [15, 31, 63]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 100),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
            "max_iter": trial.suggest_categorical("max_iter", [150, 250, 350]),
        }

    raise ValueError(f"Неизвестное название модели: {model_name}")


def tune_model_with_optuna(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    linear_preprocessor,
    tree_preprocessor,
    cv,
    scoring: str,
    n_trials: int,
):
    """
    Подбирает гиперпараметры через Optuna и возвращает лучшую обученную модель
    """
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_name)
        estimator = build_estimator(
            model_name=model_name,
            linear_preprocessor=linear_preprocessor,
            tree_preprocessor=tree_preprocessor,
            params=params,
        )
        scores = cross_val_score(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_model = build_estimator(
        model_name=model_name,
        linear_preprocessor=linear_preprocessor,
        tree_preprocessor=tree_preprocessor,
        params=study.best_params,
    )
    best_model.fit(X_train, y_train)

    return study, best_model

def plot_confusion_matrices_with_thresholds(
    models: dict,
    thresholds: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
) -> None:
    n_models = len(models)
    n_cols = 2
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (model_name, model) in zip(axes, models.items()):
        threshold = thresholds[model_name]
        y_score = model.predict_proba(X_test)[:, 1]
        y_pred = (y_score >= threshold).astype(int)

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            ax=ax,
            cmap="Blues",
            colorbar=False,
        )

        ax.set_title(
            f"Confusion matrix: {model_name}, threshold={threshold:.2f}"
        )

    for ax in axes[n_models:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    default_log_file = Path(__file__).resolve().parent / "logs" / "train_models.log"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "models"),
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "images"),
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=str(default_log_file),
    )
    parser.add_argument(
        "--disable-file-logging",
        action="store_true",
        help="Отключить запись логов в файл.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--valid-size",
        type=float,
        default=0.25,
        help="Доля validation внутри train_valid. ",
    )
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--scoring", type=str, default="roc_auc")
    parser.add_argument("--n-trials", type=int, default=12)

    args = parser.parse_args()

    log_file = None if args.disable_file_logging else Path(args.log_file)
    configure_logging(log_file=log_file)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    images_dir = Path(args.images_dir)

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Запуск обучения моделей для HW6")
    if log_file is not None:
        logger.info("Логи сохраняются в файл: %s", log_file)

    db = PostgresDB(DB_ARGS)
    db.connect()

    try:
        logger.info("Загрузка данных из PostgreSQL")
        application_df = load_application(db)
        bureau_df = load_bureau(db)
        previous_application_df = load_previous_application(db)

        logger.info("Подготовка итогового датасета")
        X, y = prepare_dataset(
            application_df=application_df,
            bureau_df=bureau_df,
            previous_application_df=previous_application_df,
        )
    finally:
        db.close()
        logger.info("Соединение с базой данных закрыто")

    logger.info("Доля положительного класса: %.4f", y.mean())

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=args.valid_size,
        random_state=RANDOM_STATE,
        stratify=y_train_valid,
    )

    logger.info("Размер train: %d", len(X_train))
    logger.info("Размер valid: %d", len(X_valid))
    logger.info("Размер test: %d", len(X_test))

    numeric_cols, categorical_cols = get_feature_types(X_train)
    logger.info("Числовых признаков: %d", len(numeric_cols))
    logger.info("Категориальных признаков: %d", len(categorical_cols))

    linear_preprocessor, tree_preprocessor = build_preprocessors(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    cv = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    model_names = [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
    ]

    results = []
    roc_data = {}
    trained_models = {}
    thresholds_by_model = {}

    for model_name in model_names:
        logger.info("%s", "=" * 80)
        logger.info("Обучение модели: %s", model_name)

        study, best_model = tune_model_with_optuna(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            linear_preprocessor=linear_preprocessor,
            tree_preprocessor=tree_preprocessor,
            cv=cv,
            scoring=args.scoring,
            n_trials=args.n_trials,
        )


        metrics, y_test_score = evaluate_model(best_model, X_test, y_test)

        y_valid_score = best_model.predict_proba(X_valid)[:, 1]
        best_threshold_row, threshold_report = find_best_threshold(
            y_true=y_valid,
            y_score=y_valid_score,
            min_precision=0.18,
            min_accuracy=0.70,
        )

        threshold_value = float(best_threshold_row["threshold"])
        thresholds_by_model[model_name] = threshold_value

        test_threshold_metrics = metrics_at_threshold(
            y_true=y_test,
            y_score=y_test_score,
            threshold=threshold_value,
        )

        metrics = {
            **metrics,
            "best_threshold": threshold_value,
            "valid_threshold_precision": best_threshold_row["precision"],
            "valid_threshold_recall": best_threshold_row["recall"],
            "valid_threshold_f1": best_threshold_row["f1"],
            "valid_threshold_f2": best_threshold_row["f2"],
            "valid_threshold_accuracy": best_threshold_row["accuracy"],
            "threshold_precision": test_threshold_metrics["precision"],
            "threshold_recall": test_threshold_metrics["recall"],
            "threshold_f1": test_threshold_metrics["f1"],
            "threshold_f2": test_threshold_metrics["f2"],
            "threshold_accuracy": test_threshold_metrics["accuracy"],
        }

        logger.info(
            "Лучший threshold для %s выбран на valid: %.2f | precision=%.4f | recall=%.4f | f1=%.4f | f2=%.4f | accuracy=%.4f",
            model_name,
            threshold_value,
            best_threshold_row["precision"],
            best_threshold_row["recall"],
            best_threshold_row["f1"],
            best_threshold_row["f2"],
            best_threshold_row["accuracy"],
        )

        logger.info(
            "Метрики на test для %s при threshold=%.2f | precision=%.4f | recall=%.4f | f1=%.4f | f2=%.4f | accuracy=%.4f",
            model_name,
            threshold_value,
            test_threshold_metrics["precision"],
            test_threshold_metrics["recall"],
            test_threshold_metrics["f1"],
            test_threshold_metrics["f2"],
            test_threshold_metrics["accuracy"],
        )

        roc_data[model_name] = y_test_score
        trained_models[model_name] = best_model

        results.append(
            {
                "model": model_name,
                "best_cv_score": study.best_value,
                "best_params": str(study.best_params),
                **metrics,
            }
        )

        logger.info("Лучшие параметры для %s: %s", model_name, study.best_params)
        logger.info(
            (
                "Метрики для %s | roc_auc=%.6f | pr_auc=%.6f | "
                "f1=%.6f | precision=%.6f | recall=%.6f | accuracy=%.6f"
            ),
            model_name,
            metrics["roc_auc"],
            metrics["pr_auc"],
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
            metrics["accuracy"],
        )

        model_path = models_dir / f"{model_name}.pkl"
        save_pickle(best_model, model_path)
        logger.info("Модель сохранена: %s", model_path)

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)

    logger.info(
        "Итоговая таблица метрик:\n%s",
        results_df[
            ["model", "roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy"]
        ].to_string(index=False),
    )

    metrics_path = results_dir / "model_metrics.csv"
    results_df.to_csv(metrics_path, index=False)
    logger.info("Метрики сохранены: %s", metrics_path)

    roc_path = images_dir / "roc_curves.png"
    plot_roc_curves(roc_data, y_test, roc_path)
    logger.info("ROC-кривые сохранены: %s", roc_path)

    cm_path = images_dir / "confusion_matrices.png"
    plot_confusion_matrices(trained_models, X_test, y_test, cm_path)
    logger.info("Матрицы ошибок сохранены: %s", cm_path)
    threshold_cm_path = images_dir / "confusion_matrices_threshold.png"
    plot_confusion_matrices_with_thresholds(
        models=trained_models,
        thresholds=thresholds_by_model,
        X_test=X_test,
        y_test=y_test,
        output_path=threshold_cm_path,
    )
    logger.info(
        "Матрицы ошибок с подобранными threshold сохранены: %s",
        threshold_cm_path,
    )

    logger.info("Обучение завершено успешно")


if __name__ == "__main__":
    main()
