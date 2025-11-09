import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

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


def plot_feature_importance(model, feature_names, top_n=10, save_path=None):
    # Check if model has feature_importances_ (like RF)
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have 'feature_importances_' attribute.")
        return

    importances = model.feature_importances_
    df_importances = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    df_importances.head(top_n).plot(kind='barh', x='feature', y='importance')
    plt.gca().invert_yaxis()
    plt.title(f'Top {top_n} Feature Importances')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_parameter_sensitivity(param_values, scores, param_name, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, scores, marker='o')
    plt.xlabel(param_name)
    plt.ylabel('F1-Score (or Accuracy)')
    plt.title(f'Model Performance vs. {param_name}')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def get_results(y_actual, y_pred, target_names):
    print(classification_report(y_actual, y_pred, target_names=target_names))
    plot_confusion_matrix(
    y_actual, 
    y_pred, 
    target_names=target_names
)
    
def get_cross_validation_scores(model, X_train, y_train, cv=10, scoring = 'f1_weighted'):
    validation_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring = scoring)
    print(f"{cv}-Fold CV Scores: {validation_scores}")
    print(f"Average {scoring} Score: {validation_scores.mean():.4f}")
    print(f"Standard Deviation: {validation_scores.std():.4f}")
    
def save_results_to_csv(model, pp, test_path, target_path, scaled = True, ridge = False):
    df_test = pd.read_csv(test_path)
    # Some columns have leading whitspaces
    df_test.columns = df_test.columns.str.strip()
    # Removing columns that cause multicollinearity
    df_test = df_test.drop(columns=['perimeterMean', 'areaMean', 'perimeterWorst', 'areaWorst', 'perimeterStdErr', 'areaStdErr', 'concavePointsMean', 'textureWorst', 'radiusMean'], axis = 1)
    
    ids = df_test['ID']
    X = df_test.drop(['ID'], axis=1)
    if(scaled):
        # Apply the scaler to the test set
        X = pp.scaler.transform(X)
    # Get predictions
    preds = model.predict(X)

    # df for the results
    df_result = pd.DataFrame()
    df_result['ID'] = ids
    df_result['class'] = preds
    
    if ridge:
        df_result['class'] = df_result['class'].map({-1: 'false', 1: 'true'})
    else:    
        df_result['class'] = df_result['class'].map({0: 'false', 1: 'true'})
        
    df_result.to_csv(target_path, index=False)