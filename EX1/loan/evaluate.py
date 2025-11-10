import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score


def plot_confusion_matrix(y_test, y_pred, target_names, save_path=None):
    labels = list(target_names)
    cm = confusion_matrix(y_test, y_pred, labels=labels, normalize="true")

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"color": "yellow"},
        cbar=True,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix simple")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_feature_importance(model, feature_names, top_n=10, save_path=None):
    # Check if model has feature_importances_ (like RF)
    if not hasattr(model, "feature_importances_"):
        print("Model does not have 'feature_importances_' attribute.")
        return

    importances = model.feature_importances_
    df_importances = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values(by="importance", ascending=False)

    plt.figure(figsize=(10, 6))
    df_importances.head(top_n).plot(kind="barh", x="feature", y="importance")
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} Feature Importances")

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_parameter_sensitivity(
    param_values, scores, param_name, save_path=None, tick_labels=None
):
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, scores, marker="o")
    # Annotate each point with its exact value
    ax = plt.gca()
    for x, y in zip(param_values, scores):
        ax.annotate(
            f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center"
        )
    # Optional categorical-style x labels (e.g., include 'None')
    if tick_labels is not None:
        ax.set_xticks(param_values)
        ax.set_xticklabels([str(lbl) for lbl in tick_labels])
    plt.xlabel(param_name)
    plt.ylabel("F1-Score (or Accuracy)")
    plt.title(f"Model Performance vs. {param_name}")

    if save_path:
        plt.savefig(save_path)
    plt.show()


def get_results(
    y_actual, y_pred, target_names, plot_cm: bool = True, save_path: str | None = None
):
    print(classification_report(y_actual, y_pred, target_names=target_names))
    if plot_cm:
        plot_confusion_matrix(
            y_actual, y_pred, target_names=target_names, save_path=save_path
        )


def get_cross_validation_scores(model, X_train, y_train, cv=10, scoring="f1_weighted"):
    validation_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
    print(f"{cv}-Fold CV Scores: {validation_scores}")
    print(f"Average {scoring} Score: {validation_scores.mean():.4f}")
    print(f"Standard Deviation: {validation_scores.std():.4f}")
