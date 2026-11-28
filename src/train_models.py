"""
Train and compare era-prediction classifiers on Spotify audio features using
balanced class weights, engineered interaction features, ordinal era framing,
and an MLP neural network benchmark.

Usage:
    python src/train_models.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.modeling_utils import (
    ALL_MODEL_FEATURES,
    ALL_NUMERIC_FEATURES,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    ERA_ORDER,
    RAW_INPUT_FEATURES,
    AudioFeatureEngineer,
    BalancedMLPClassifier,
    OrdinalGBDTClassifier,
)

TARGET_COL = "era"
RANDOM_STATE = 42
ERA_TO_INDEX = {era: idx for idx, era in enumerate(ERA_ORDER)}


def load_datasets(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre-split training and test CSV datasets."""
    train_df = pd.read_csv(data_dir / "train_data.csv")
    test_df = pd.read_csv(data_dir / "test_data.csv")

    missing_cols = set(RAW_INPUT_FEATURES + [TARGET_COL]) - set(train_df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns in dataset: {missing_cols}")

    return train_df, test_df


def _build_scaled_ohe_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ALL_NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ]
    )


def build_logistic_regression_pipeline() -> Pipeline:
    """Balanced multinomial Logistic Regression with engineered audio features."""
    return Pipeline(
        steps=[
            ("features", AudioFeatureEngineer()),
            ("preprocessor", _build_scaled_ohe_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    C=0.1,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_random_forest_pipeline() -> Pipeline:
    """Balanced Random Forest with regularized depth and engineered audio features."""
    return Pipeline(
        steps=[
            ("features", AudioFeatureEngineer()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=15,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_gbdt_pipeline() -> Pipeline:
    """Balanced Histogram-based Gradient Boosted Decision Trees with engineered features."""
    return Pipeline(
        steps=[
            ("features", AudioFeatureEngineer()),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    class_weight="balanced",
                    categorical_features=CATEGORICAL_FEATURES,
                    learning_rate=0.05,
                    max_iter=400,
                    max_leaf_nodes=63,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_ordinal_gbdt_pipeline() -> Pipeline:
    """Frank & Hall cumulative Ordinal GBDT exploiting `<2000` -> `2021+` ordering."""
    return Pipeline(
        steps=[
            ("features", AudioFeatureEngineer()),
            (
                "classifier",
                OrdinalGBDTClassifier(
                    era_order=tuple(ERA_ORDER),
                    categorical_features=tuple(CATEGORICAL_FEATURES),
                    learning_rate=0.05,
                    max_iter=400,
                    max_leaf_nodes=63,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_mlp_pipeline() -> Pipeline:
    """Balanced Multi-Layer Perceptron (MLP) neural network benchmark."""
    return Pipeline(
        steps=[
            ("features", AudioFeatureEngineer()),
            ("preprocessor", _build_scaled_ohe_preprocessor()),
            (
                "classifier",
                BalancedMLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    alpha=1e-2,
                    batch_size=512,
                    learning_rate_init=1e-3,
                    max_iter=50,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_model(
    name: str,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate a trained pipeline on the held-out test set (classification + ordinal metrics)."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    macro_roc_auc = float(
        roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    )

    # Ordinal error metrics across chronological eras (0, 1, 2, 3)
    true_idx = y_test.map(ERA_TO_INDEX).to_numpy()
    pred_idx = pd.Series(y_pred).map(ERA_TO_INDEX).to_numpy()
    era_mae = float(mean_absolute_error(true_idx, pred_idx))
    within_1_era = float((np.abs(true_idx - pred_idx) <= 1).mean())

    report_dict = classification_report(
        y_test, y_pred, labels=ERA_ORDER, digits=4, output_dict=True
    )
    report_str = classification_report(
        y_test, y_pred, labels=ERA_ORDER, digits=4
    )
    cm = confusion_matrix(y_test, y_pred, labels=ERA_ORDER)
    cm_norm = confusion_matrix(y_test, y_pred, labels=ERA_ORDER, normalize="true")

    print(f"\n{'=' * 72}")
    print(f"TEST SET EVALUATION: {name}")
    print(f"{'=' * 72}")
    print(f"Accuracy             : {acc:.4f}")
    print(f"Balanced Accuracy    : {bal_acc:.4f}  (random 4-class baseline = 0.2500)")
    print(f"Macro F1-Score       : {macro_f1:.4f}")
    print(f"Weighted F1-Score    : {weighted_f1:.4f}")
    print(f"Macro ROC-AUC        : {macro_roc_auc:.4f}")
    print(f"Within-1-Era Acc     : {within_1_era:.4f}  (|pred_era - true_era| <= 1)")
    print(f"Ordinal Era MAE      : {era_mae:.4f}  (lower = fewer multi-decade jumps)")
    print("\nPer-Era Classification Report (Chronological):")
    print(report_str)

    return {
        "model_name": name,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_roc_auc": macro_roc_auc,
        "within_1_era_accuracy": within_1_era,
        "era_mae": era_mae,
        "per_era": {
            era: {
                "precision": float(report_dict[era]["precision"]),
                "recall": float(report_dict[era]["recall"]),
                "f1_score": float(report_dict[era]["f1-score"]),
                "support": int(report_dict[era]["support"]),
            }
            for era in ERA_ORDER
        },
        "confusion_matrix_labels": ERA_ORDER,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_recall_normalized": np.round(cm_norm, 4).tolist(),
        "report_str": report_str,
    }


def plot_confusion_matrices(evals: list[dict], output_path: Path) -> None:
    """Save 2x3 dashboard of 5 confusion matrices + overall metric comparison bar chart."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(19, 11.2))
    flat_axes = axes.flatten()

    cmaps = ["Blues", "Greens", "Oranges", "YlOrBr", "Purples"]
    for ax, res, cmap in zip(flat_axes[:5], evals, cmaps):
        cm = np.array(res["confusion_matrix"])
        cm_norm = np.array(res["confusion_matrix_recall_normalized"]) * 100.0

        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm_norm[i, j]:.1f}%\n({cm[i, j]:,})"

        sns.heatmap(
            cm_norm,
            annot=annot,
            fmt="",
            cmap=cmap,
            vmin=0,
            vmax=100,
            xticklabels=ERA_ORDER,
            yticklabels=ERA_ORDER,
            cbar_kws={"label": "Row Recall (%)"},
            ax=ax,
        )
        ax.set_title(
            f"{res['model_name']}\n"
            f"Acc: {res['accuracy']:.3f} | Bal Acc: {res['balanced_accuracy']:.3f} | "
            f"F1: {res['macro_f1']:.3f} | Adj: {res['within_1_era_accuracy']:.3f}",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Predicted Era", fontsize=10.5)
        ax.set_ylabel("True Era", fontsize=10.5)

    # Panel 6: Grouped bar summary across all 5 models
    ax_bar = flat_axes[5]
    short_names = ["LogReg", "RandForest", "GBDT", "Ordinal GBDT", "MLP (NN)"]
    x = np.arange(len(short_names))
    w = 0.20
    ax_bar.bar(x - 1.5 * w, [r["accuracy"] for r in evals], w, label="Accuracy", color="#4682B4")
    ax_bar.bar(x - 0.5 * w, [r["balanced_accuracy"] for r in evals], w, label="Balanced Acc", color="#1DB954")
    ax_bar.bar(x + 0.5 * w, [r["macro_f1"] for r in evals], w, label="Macro F1", color="#E67E22")
    ax_bar.bar(x + 1.5 * w, [r["within_1_era_accuracy"] for r in evals], w, label="Within-1-Era Acc", color="#8E44AD")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(short_names, rotation=15, fontsize=10)
    ax_bar.set_ylim(0, 1.0)
    ax_bar.set_ylabel("Score", fontsize=10.5)
    ax_bar.set_title("Summary Metric Comparison Across All 5 Models", fontsize=11, fontweight="bold")
    ax_bar.legend(loc="upper left", fontsize=9.5)

    plt.suptitle(
        "Test Set Confusion Matrices & Benchmark Comparison (With Engineered Audio Features)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importances(
    lr_pipeline: Pipeline,
    rf_pipeline: Pipeline,
    gbdt_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
) -> None:
    """Plot feature importances including engineered interaction features."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.8))

    # 1. Logistic Regression mean |coef| across eras (on engineered feature space)
    pre: ColumnTransformer = lr_pipeline.named_steps["preprocessor"]
    lr_clf: LogisticRegression = lr_pipeline.named_steps["classifier"]
    abs_coef_mean = np.abs(lr_clf.coef_).mean(axis=0)

    grouped_importance: dict[str, float] = {}
    for fname, val in zip(pre.get_feature_names_out(), abs_coef_mean):
        clean = fname.split("__", 1)[1]
        if clean.startswith("key_"):
            grouped_importance.setdefault("key (one-hot avg)", 0.0)
            grouped_importance["key (one-hot avg)"] += float(val) / 11.0
        else:
            grouped_importance[clean] = float(val)

    lr_imp = pd.Series(grouped_importance).sort_values(ascending=True)
    axes[0].barh(lr_imp.index, lr_imp.values, color="#4682B4", edgecolor="#121212")
    axes[0].set_title("Logistic Regression\nMean |Standardized Weight|", fontsize=11.5, fontweight="bold")
    axes[0].set_xlabel("Mean |Coefficient|", fontsize=10.5)

    # 2. Random Forest Gini feature importances (all 17 engineered + raw features)
    rf_clf: RandomForestClassifier = rf_pipeline.named_steps["classifier"]
    rf_imp = pd.Series(rf_clf.feature_importances_, index=ALL_MODEL_FEATURES).sort_values(ascending=True)
    axes[1].barh(rf_imp.index, rf_imp.values, color="#1DB954", edgecolor="#121212")
    axes[1].set_title("Random Forest\nImpurity Importance (17 Features)", fontsize=11.5, fontweight="bold")
    axes[1].set_xlabel("Mean Decrease in Impurity", fontsize=10.5)

    # 3. GBDT internal tree permutation importance on engineered features
    X_test_fe = gbdt_pipeline.named_steps["features"].transform(X_test)
    gbdt_clf = gbdt_pipeline.named_steps["classifier"]
    perm = permutation_importance(
        gbdt_clf,
        X_test_fe,
        y_test,
        scoring="balanced_accuracy",
        n_repeats=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    gbdt_imp = pd.Series(perm.importances_mean, index=ALL_MODEL_FEATURES).sort_values(ascending=True)
    axes[2].barh(gbdt_imp.index, gbdt_imp.values, color="#E67E22", edgecolor="#121212")
    axes[2].set_title("Gradient Boosted Trees\nPermutation Importance (17 Features)", fontsize=11.5, fontweight="bold")
    axes[2].set_xlabel("Mean Balanced Accuracy Decrease", fontsize=10.5)

    plt.suptitle(
        "Feature Importance Including Engineered Audio Interactions",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data_dir = REPO_ROOT / "data"
    models_dir = REPO_ROOT / "models"
    results_dir = REPO_ROOT / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading train and test datasets...")
    train_df, test_df = load_datasets(data_dir)
    X_train = train_df[RAW_INPUT_FEATURES]
    y_train = train_df[TARGET_COL]
    X_test = test_df[RAW_INPUT_FEATURES]
    y_test = test_df[TARGET_COL]

    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    print(f"Engineered pipeline feature count: {len(ALL_MODEL_FEATURES)} features")

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(ERA_ORDER),
        y=y_train,
    )
    class_weight_summary = {}
    print("\nTraining Era Distribution & Balanced Class Weights:")
    for era, w in zip(ERA_ORDER, weights):
        count = int((y_train == era).sum())
        pct = 100.0 * count / len(y_train)
        class_weight_summary[era] = {
            "train_count": count,
            "train_pct": round(pct, 4),
            "test_count": int((y_test == era).sum()),
            "balanced_weight": round(float(w), 4),
        }
        print(f"  {era:10s}: {count:6,d} ({pct:5.2f}%) -> weight = {w:.4f}")

    model_specs = [
        ("logistic_regression", "Logistic Regression (Balanced)", build_logistic_regression_pipeline(), "logistic_regression_era.joblib"),
        ("random_forest", "Random Forest (Balanced)", build_random_forest_pipeline(), "random_forest_era.joblib"),
        ("gradient_boosted_trees", "Gradient Boosted Trees (Balanced)", build_gbdt_pipeline(), "gradient_boosted_era.joblib"),
        ("ordinal_gbdt", "Ordinal GBDT (Frank & Hall)", build_ordinal_gbdt_pipeline(), "ordinal_gbdt_era.joblib"),
        ("mlp_neural_net", "MLP Neural Network (Balanced)", build_mlp_pipeline(), "mlp_era.joblib"),
    ]

    trained_pipelines: dict[str, Pipeline] = {}
    eval_results: dict[str, dict] = {}
    ordered_evals: list[dict] = []

    for key, label, pipeline, filename in model_specs:
        print(f"\nTraining {label}...")
        pipeline.fit(X_train, y_train)
        res = evaluate_model(label, pipeline, X_test, y_test)
        model_path = models_dir / filename
        joblib.dump(pipeline, model_path)
        res["artifact_file"] = filename
        res["artifact_size_kb"] = round(model_path.stat().st_size / 1024.0, 1)
        trained_pipelines[key] = pipeline
        eval_results[key] = res
        ordered_evals.append(res)

    # Export recommended production model (`best_era_model.joblib`)
    # Following 'Best AI Model Selection': lightweight GBDT variant with best deployment footprint & balanced metrics
    best_key = "gradient_boosted_trees"
    if (
        eval_results["ordinal_gbdt"]["macro_f1"] >= eval_results["gradient_boosted_trees"]["macro_f1"]
        and eval_results["ordinal_gbdt"]["within_1_era_accuracy"] > eval_results["gradient_boosted_trees"]["within_1_era_accuracy"]
    ):
        best_key = "ordinal_gbdt"

    best_src = models_dir / eval_results[best_key]["artifact_file"]
    best_dst = models_dir / "best_era_model.joblib"
    shutil.copy2(best_src, best_dst)
    print(
        f"\nExported recommended production model ({eval_results[best_key]['model_name']}, "
        f"{eval_results[best_key]['artifact_size_kb']:.1f} KB) -> {best_dst}"
    )

    # Save plots
    cm_plot_path = results_dir / "confusion_matrices.png"
    fi_plot_path = results_dir / "feature_importance_comparison.png"
    plot_confusion_matrices(ordered_evals, cm_plot_path)
    plot_feature_importances(
        trained_pipelines["logistic_regression"],
        trained_pipelines["random_forest"],
        trained_pipelines["gradient_boosted_trees"],
        X_test,
        y_test,
        fi_plot_path,
    )
    print(f"Saved plots:\n  - {cm_plot_path}\n  - {fi_plot_path}")

    # Save reports and JSON metrics
    report_txt_path = results_dir / "classification_reports.txt"
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("TRAINING ERA COUNTS & BALANCED WEIGHTS\n")
        f.write("=" * 72 + "\n")
        for era, info in class_weight_summary.items():
            f.write(
                f"{era:10s}: train={info['train_count']:,} ({info['train_pct']:.2f}%), "
                f"test={info['test_count']:,}, weight={info['balanced_weight']:.4f}\n"
            )
        f.write("\n")
        for res in ordered_evals:
            f.write("=" * 72 + "\n")
            f.write(f"{res['model_name']} ({res['artifact_size_kb']:.1f} KB)\n")
            f.write("=" * 72 + "\n")
            f.write(f"Accuracy             : {res['accuracy']:.4f}\n")
            f.write(f"Balanced Accuracy    : {res['balanced_accuracy']:.4f}\n")
            f.write(f"Macro F1-Score       : {res['macro_f1']:.4f}\n")
            f.write(f"Weighted F1-Score    : {res['weighted_f1']:.4f}\n")
            f.write(f"Macro ROC-AUC        : {res['macro_roc_auc']:.4f}\n")
            f.write(f"Within-1-Era Acc     : {res['within_1_era_accuracy']:.4f}\n")
            f.write(f"Ordinal Era MAE      : {res['era_mae']:.4f}\n\n")
            f.write(res["report_str"] + "\n")

    metrics_json_path = results_dir / "metrics_summary.json"
    summary_payload = {
        "dataset": {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "raw_input_features": RAW_INPUT_FEATURES,
            "engineered_model_features": ALL_MODEL_FEATURES,
            "era_order": ERA_ORDER,
            "class_weights": class_weight_summary,
        },
        "recommended_production_model": {
            "key": best_key,
            "model_name": eval_results[best_key]["model_name"],
            "artifact": "best_era_model.joblib",
            "size_kb": eval_results[best_key]["artifact_size_kb"],
        },
        "models": {
            k: {mk: mv for mk, mv in v.items() if mk != "report_str"}
            for k, v in eval_results.items()
        },
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"Saved reports:\n  - {report_txt_path}\n  - {metrics_json_path}")


if __name__ == "__main__":
    main()
