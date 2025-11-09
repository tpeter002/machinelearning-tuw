import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from scikeras.wrappers import KerasClassifier


def get_results(y_actual, y_pred, target_names):
    print(classification_report(y_actual, y_pred, target_names=target_names))
    plot_confusion_matrix(
    y_actual, 
    y_pred, 
    target_names=target_names
)

def plot_confusion_matrix(y_test, y_pred, target_names, save_path=None):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def get_cross_validation_scores(model, X_train, y_train, cv=10, scoring = 'f1_weighted'):
    validation_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring = scoring)
    print(f"{cv}-Fold CV Scores: {validation_scores}")
    print(f"Average {scoring} Score: {validation_scores.mean():.4f}")
    print(f"Standard Deviation: {validation_scores.std():.4f}")

def get_cross_validation_scores_keras(model_builder, X_train, y_train, input_shape, num_classes, cv=10, scoring='f1_weighted'):
    """
    Cross-validation for Keras models using KerasClassifier wrapper
    """
    # Wrap the Keras model for sklearn compatibility
    keras_model = KerasClassifier(
        model=model_builder,
        input_shape=input_shape,
        num_classes=num_classes,
        epochs=30,
        batch_size=32,
        verbose=0
    )
    
    validation_scores = cross_val_score(keras_model, X_train, y_train, cv=cv, scoring=scoring)
    print(f"{cv}-Fold CV Scores: {validation_scores}")
    print(f"Average {scoring} Score: {validation_scores.mean():.4f}")
    print(f"Standard Deviation: {validation_scores.std():.4f}")