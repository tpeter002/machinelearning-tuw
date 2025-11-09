import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
import tensorflow as tf
import numpy as np

from preprocess import Preprocessing, PATH
from evaluation import test_model


def random_forest_pipeline():
    """
    Simple random forest pipeline

    Loads data, fits random forest, tests relevant metrics
    """
    pp = Preprocessing()

    # random forest doesn't require scaling
    X_train, X_test, y_train, y_test = pp.pipeline(PATH, split=True, scale=False)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    labels = pp.le.transform(pp.le.classes_)
    test_model(rf_model, X_test, y_test, labels)

def ridge_pipeline():
    """
    Ridge classifier pipeline

    Loads data, fits ridge classifier, tests relevant metrics
    """
    pp = Preprocessing()

    X_train, X_test, y_train, y_test = pp.pipeline(PATH, split=True, scale=True)
    ridge_model = RidgeClassifier(random_state=42)
    ridge_model.fit(X_train, y_train)
    labels = pp.le.transform(pp.le.classes_)
    test_model(ridge_model, X_test, y_test, labels)

def build_neural_net(input_shape, num_classes):
    """
    Builds a keras MLP with dropout
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax') 
    ])
    
    # We use 'sparse_categorical_crossentropy' because our y labels are integers (0, 1, 2, 3)
    # not one-hot encoded vectors.
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def neural_net_pipeline():
    """
    Runs neural net pipeline

    Loads data, builds and trains MLP, tests metrics
    """
    pp = Preprocessing()
    X_train, X_test, y_train, y_test = pp.pipeline(PATH, split=True, scale=True)
    labels = pp.le.transform(pp.le.classes_)
    
    nn_model = build_neural_net(input_shape=X_train.shape[1], num_classes=len(labels))
    
    print("Training MLP...")
    nn_model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    y_prob_nn = nn_model.predict(X_test)
    y_pred_nn = np.argmax(y_prob_nn, axis=1)
    test_model(nn_model, X_test, y_test, labels, y_pred_nn)

    
def main():
    random_forest_pipeline()
    ridge_pipeline()
    neural_net_pipeline()

if __name__=="__main__":
    main()

