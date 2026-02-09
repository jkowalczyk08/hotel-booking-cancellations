import re
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV

from src.set_utils import load_splits


def normalize_notebook_title(notebook_name: str) -> str:
    """
    Convert a notebook filename/path into a human-readable title.

    Rules:
    - remove directories and `.ipynb` suffix
    - remove digits
    - replace underscores with spaces
    - collapse repeated whitespace
    """
    base_name = notebook_name.split("/")[-1].split("\\")[-1]
    base_name = base_name.removesuffix(".ipynb")
    base_name = re.sub(r"\d+", "", base_name).replace("_", " ")
    normalized = re.sub(r"\s+", " ", base_name).strip()
    return normalized.title() if normalized else "Model"


def _build_plot_title(notebook_name: str | None, suffix: str, fallback: str) -> str:
    if notebook_name:
        return f"{normalize_notebook_title(notebook_name)} - {suffix}"
    return fallback


def prepare_encoded_splits(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    custom feature engineering and encoding approach:
    1. agent: Top-10 frequent agents (excluding 0 - NaN replacement after preprocessing) become binary cols,
        rest -> agent_other, 0 -> agent_none.
    2. country: Top-10 frequent countries become binary cols, rest -> country_other.
    3. company: binary has_company.
    4. arrival_date_week_number: cyclical week_sin and week_cos columns (week 53 and week 1 are close).
    5. Remaining object cols: One Hot Encoded.
    """

    # top 10 on training set to avoid data leakage
    top_agents = X_train.loc[X_train["agent"] != 0, "agent"].value_counts().nlargest(10).index.tolist()
    top_countries = X_train["country"].value_counts().nlargest(10).index.tolist()


    def process_split_high_cardinality_categorical(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for agent_id in top_agents:
            df[f"agent_{int(agent_id)}"] = (df["agent"] == agent_id).astype(int)
        
        df["agent_other"] = (~df["agent"].isin(top_agents) & (df["agent"] != 0)).astype(int)
        df["agent_none"] = (df["agent"] == 0).astype(int)
        df = df.drop(columns=["agent"])

        for country_code in top_countries:
            df[f"country_{country_code}"] = (df["country"] == country_code).astype(int)
        
        df["country_other"] = (~df["country"].isin(top_countries)).astype(int)
        df = df.drop(columns=["country"])

        df["has_company"] = (df["company"] != 0).astype(int)
        df = df.drop(columns=["company"])

        
        df["week_sin"] = np.sin(2 * np.pi * df["arrival_date_week_number"] / 53)
        df["week_cos"] = np.cos(2 * np.pi * df["arrival_date_week_number"] / 53)
        df = df.drop(columns=["arrival_date_week_number"])
        
        return df

    X_train_proc = process_split_high_cardinality_categorical(X_train)
    X_val_proc = process_split_high_cardinality_categorical(X_val)
    X_test_proc = process_split_high_cardinality_categorical(X_test)

    combined = pd.concat({"train": X_train_proc, "val": X_val_proc, "test": X_test_proc}, names=["split"])
    combined_encoded = pd.get_dummies(combined, drop_first=False)

    X_train_final = combined_encoded.xs("train", level="split")
    X_val_final = combined_encoded.xs("val", level="split")
    X_test_final = combined_encoded.xs("test", level="split")

    X_val_final = X_val_final.reindex(columns=X_train_final.columns, fill_value=0)
    X_test_final = X_test_final.reindex(columns=X_train_final.columns, fill_value=0)

    return X_train_final, X_val_final, X_test_final


def prepare_encoded_splits_no_leakage(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Baseline encoding with leakage-prone assigned room type removed.
    """
    X_train = X_train.drop(columns=["assigned_room_type"], errors="ignore")
    X_val = X_val.drop(columns=["assigned_room_type"], errors="ignore")
    X_test = X_test.drop(columns=["assigned_room_type"], errors="ignore")

    return prepare_encoded_splits(X_train, X_val, X_test)


def prepare_encoded_splits_enhanced(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    better feature engineering and encoding:
    1. [OLD] agent: Top-10 frequent agents (excluding 0 - NaN replacement after preprocessing) become binary cols,
        rest -> agent_other, 0 -> agent_none.
    2. [OLD] country: Top-10 frequent countries become binary cols, rest -> country_other.
    3. [OLD] company: binary has_company.
    4. [OLD] arrival_date_week_number: cyclical week_sin and week_cos columns (week 53 and week 1 are close).
    5. [NEW] arrival_date_month: cyclical month_sin and month_cos.
    6. [DILEMA] room_type_changed: binary (reserved != assigned) - this might be a data leakage.
        Assigned room is possibly known after checkin, so it is better to remove this feature entirely, although the room_type_changed would be tempting.
    7. [NEW] cancellation_ratio: previous_cancellations / (total_history + 1).
    8. [NEW] cost_per_person: adr / people.
    9. [NEW] total_cost: adr * total_nights.
    10. [NEW] is_family_special: is_family * total_of_special_requests - random, but seems like something that would not be cancelled
    11. Remaining object cols: One Hot Encoded.
    """

    # top 10 on training set to avoid data leakage
    top_agents = X_train.loc[X_train["agent"] != 0, "agent"].value_counts().nlargest(10).index.tolist()
    top_countries = X_train["country"].value_counts().nlargest(10).index.tolist()

    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
    }

    def process_split_enhanced(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Existing transformations
        for agent_id in top_agents:
            df[f"agent_{int(agent_id)}"] = (df["agent"] == agent_id).astype(int)
        
        df["agent_other"] = (~df["agent"].isin(top_agents) & (df["agent"] != 0)).astype(int)
        df["agent_none"] = (df["agent"] == 0).astype(int)
        df = df.drop(columns=["agent"])

        for country_code in top_countries:
            df[f"country_{country_code}"] = (df["country"] == country_code).astype(int)
        
        df["country_other"] = (~df["country"].isin(top_countries)).astype(int)
        df = df.drop(columns=["country"])

        df["has_company"] = (df["company"] != 0).astype(int)
        df = df.drop(columns=["company"])
        
        df["week_sin"] = np.sin(2 * np.pi * df["arrival_date_week_number"] / 53)
        df["week_cos"] = np.cos(2 * np.pi * df["arrival_date_week_number"] / 53)
        df = df.drop(columns=["arrival_date_week_number"])

        # New transformations
        if "arrival_date_month" in df.columns:
            if df["arrival_date_month"].dtype == "O":
                 df["month_num"] = df["arrival_date_month"].map(month_map)
            else:
                 df["month_num"] = df["arrival_date_month"]
            
            df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
            df = df.drop(columns=["arrival_date_month", "month_num"])
        
       # Remove leakage feature
        if "assigned_room_type" in df.columns:
            df = df.drop(columns=["assigned_room_type"])
        
        total_history = df["previous_cancellations"] + df["previous_bookings_not_canceled"]
        df["cancellation_ratio"] = df["previous_cancellations"] / (total_history + 1)
        
        total_people = df["adults"] + df["children"] + df["babies"]
        df["cost_per_person"] = df["adr"] / total_people
        
        df["total_cost"] = df["adr"] * df["total_nights"]
        
        df["is_family_special"] = df["is_family"] * df["total_of_special_requests"]

        return df

    X_train_proc = process_split_enhanced(X_train)
    X_val_proc = process_split_enhanced(X_val)
    X_test_proc = process_split_enhanced(X_test)

    combined = pd.concat({"train": X_train_proc, "val": X_val_proc, "test": X_test_proc}, names=["split"])
    combined_encoded = pd.get_dummies(combined, drop_first=False)

    X_train_final = combined_encoded.xs("train", level="split")
    X_val_final = combined_encoded.xs("val", level="split")
    X_test_final = combined_encoded.xs("test", level="split")

    X_val_final = X_val_final.reindex(columns=X_train_final.columns, fill_value=0)
    X_test_final = X_test_final.reindex(columns=X_train_final.columns, fill_value=0)

    return X_train_final, X_val_final, X_test_final


def load_encoded_splits(
    data_dir: str,
    version: str = "baseline"
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(data_dir)
    
    if version == "baseline":
        X_train, X_val, X_test = prepare_encoded_splits(X_train, X_val, X_test)
    elif version == "no-leakage":
        X_train, X_val, X_test = prepare_encoded_splits_no_leakage(X_train, X_val, X_test)
    elif version == "enhanced":
        X_train, X_val, X_test = prepare_encoded_splits_enhanced(X_train, X_val, X_test)
    else:
        raise ValueError(f"Unknown version: {version}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate_classifier(model: Any, X: pd.DataFrame, y: pd.Series, split_name: str) -> dict:
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:
        y_score = y_pred

    results = {
        "split": split_name,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_score),
    }

    return results


def plot_roc_curve(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str,
    ax: plt.Axes | None = None,
    color: str = "#1f77b4",
    notebook_name: str | None = None,
) -> float:
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X)
    else:
        y_score = model.predict(X)

    fpr, tpr, _ = roc_curve(y, y_score)
    auc_value = roc_auc_score(y, y_score)

    created_figure = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        created_figure = True

    ax.plot(fpr, tpr, color=color, linewidth=2, label=f"AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1.5, label="Random")
    ax.set_title(
        _build_plot_title(
            notebook_name=notebook_name,
            suffix=f"{split_name.capitalize()} ROC curve",
            fallback=f"{split_name.capitalize()} ROC curve",
        )
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right")

    if created_figure:
        fig.tight_layout()
        if "agg" in matplotlib.get_backend().lower():
            plt.close(fig)
        else:
            plt.show()

    return auc_value


def get_confusion_matrix_df(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str,
    normalize: str | None = None,
    plot: bool = False,
    cmap: str = "Blues",
    notebook_name: str | None = None,
) -> pd.DataFrame:
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred, labels=[0, 1], normalize=normalize)

    if normalize is None:
        cm = cm.astype(int)

    cm_df = pd.DataFrame(
        cm,
        index=["Not Canceled (Actual)", "Canceled (Actual)"],
        columns=["Not Canceled (Predicted)", "Canceled (Predicted)"],
    )
    cm_df.index.name = f"{split_name}_actual"
    cm_df.columns.name = f"{split_name}_predicted"

    if plot:
        fig, ax = plt.subplots(figsize=(6, 4))
        image = ax.imshow(cm_df.values, interpolation="nearest", cmap=cmap)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks([0, 1], labels=cm_df.columns)
        ax.set_yticks([0, 1], labels=cm_df.index)
        ax.set_xlabel(cm_df.columns.name)
        ax.set_ylabel(cm_df.index.name)
        ax.set_title(
            _build_plot_title(
                notebook_name=notebook_name,
                suffix=f"{split_name.capitalize()} confusion matrix",
                fallback=f"{split_name.capitalize()} confusion matrix",
            )
        )

        value_format = ".3f" if normalize is not None else "d"
        threshold = cm_df.values.max() / 2.0 if cm_df.values.size else 0
        for row in range(cm_df.shape[0]):
            for col in range(cm_df.shape[1]):
                value = cm_df.iloc[row, col]
                text_color = "white" if value > threshold else "black"
                ax.text(col, row, format(value, value_format), ha="center", va="center", color=text_color)

        fig.tight_layout()
        if "agg" in matplotlib.get_backend().lower():
            plt.close(fig)
        else:
            plt.show()

    return cm_df


def plot_metric_scorecards(
    val_results: dict,
    test_results: dict,
    metrics: tuple[str, ...] = ("f1", "recall", "roc_auc", "precision", "accuracy"),
    decimals: int = 3,
    notebook_name: str | None = None,
) -> None:
    metric_labels = {
        "f1": "F1",
        "recall": "Recall",
        "roc_auc": "ROC-AUC",
        "precision": "Precision",
        "accuracy": "Accuracy",
    }
    split_rows = [
        ("Validation", val_results, "#E8F1FF", "#1F4E79"),
        ("Test", test_results, "#EAF8EE", "#2D6A4F"),
    ]

    fig, axes = plt.subplots(nrows=2, ncols=len(metrics), figsize=(3.2 * len(metrics), 4.8))
    if len(metrics) == 1:
        axes = axes.reshape(2, 1)

    for row_idx, (split_name, split_results, face_color, edge_color) in enumerate(split_rows):
        for col_idx, metric_name in enumerate(metrics):
            axis = axes[row_idx, col_idx]
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)

            card = FancyBboxPatch(
                (0.05, 0.08),
                0.9,
                0.84,
                boxstyle="round,pad=0.02,rounding_size=0.04",
                linewidth=1.5,
                edgecolor=edge_color,
                facecolor=face_color,
                transform=axis.transAxes,
            )
            axis.add_patch(card)

            metric_value = split_results.get(metric_name)
            display_value = "N/A" if metric_value is None else f"{float(metric_value):.{decimals}f}"

            axis.text(
                0.5,
                0.65,
                metric_labels.get(metric_name, metric_name.upper()),
                ha="center",
                va="center",
                fontsize=11,
                color=edge_color,
                fontweight="bold",
                transform=axis.transAxes,
            )
            axis.text(
                0.5,
                0.36,
                display_value,
                ha="center",
                va="center",
                fontsize=18,
                color="#111111",
                fontweight="bold",
                transform=axis.transAxes,
            )

            if col_idx == 0:
                axis.text(
                    0.08,
                    0.9,
                    split_name,
                    ha="left",
                    va="center",
                    fontsize=10,
                    color=edge_color,
                    fontweight="bold",
                    transform=axis.transAxes,
                )

    fig.suptitle(
        _build_plot_title(
            notebook_name=notebook_name,
            suffix="Model Scorecards",
            fallback="Model Scorecards",
        ),
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if "agg" in matplotlib.get_backend().lower():
        plt.close(fig)
    else:
        plt.show()


def run_random_search(
    estimator: Any,
    param_distributions: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 10,
    cv: int = 3,
    scoring: str = "f1",
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 1,
) -> RandomizedSearchCV:
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search
