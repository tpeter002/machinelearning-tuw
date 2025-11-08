from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix


def test_model(model, X_test, y_test, classes, y_pred=None):
    # fix labels, add back 
    if y_pred is None:
        y_pred = model.predict(X_test)

    print(f'test acc: {accuracy_score(y_test, y_pred)}')
    print(f'test recall: {recall_score(y_test, y_pred, labels=classes, average='macro')}')
    print(f'test f1 score: {f1_score(y_test, y_pred, labels=classes, average='macro')}')
    print(f'test confusion matrix: \n {confusion_matrix(y_test, y_pred, labels=classes)}')

#prediction, evulation and comparison pipeline

# dont forget cv and normal 

# accuracy, f1, recall, confusion matrix, auroc