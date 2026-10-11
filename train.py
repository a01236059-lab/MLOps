# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 19:29:13 2026

@author: stefa
"""

# cargar librerias
import pandas as pd
import os
import kagglehub
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report


import argparse
import mlflow
import mlflow.sklearn

# generacion de path
def load_data(path):
    train = pd.read_csv(os.path.join(path, "train.csv"))
    test = pd.read_csv(os.path.join(path, "test.csv"))
    return train, test

# procesamiento de datos
def data_cleaning(test,train):
    cols2drop=["Unnamed: 0", "id", "Arrival Delay in Minutes"]
    test=test.drop(columns=cols2drop)
    train=train.drop(columns=cols2drop)
    train = train.drop_duplicates()
    test = test.drop_duplicates()
    train["satisfaction"] = train["satisfaction"].map({
    "neutral or dissatisfied": 0,
    "satisfied": 1
    })

    test["satisfaction"] = test["satisfaction"].map({
    "neutral or dissatisfied": 0,
    "satisfied": 1
    })

    class_map = {
    "Eco": 0,
    "Eco Plus": 1,
    "Business": 2
    }

    train["Class"] = train["Class"].map(class_map)
    test["Class"] = test["Class"].map(class_map)

    col_nominal=["Gender", "Customer Type", "Type of Travel"]

    train = pd.get_dummies(
    train,
    columns=col_nominal,
    drop_first=True,
    dtype=int
    )

    test = pd.get_dummies(
    test,
    columns=col_nominal,
    drop_first=True,
    dtype=int
    )
    
    return (test, train)

# separación de conjuntos
def split_features_target(df):
    X = df.drop("satisfaction", axis=1)
    y = df["satisfaction"]

    
    return (X, y )

# Entrenamiento de modelo
def train_model(X_train, y_train, args):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    #X_test_scaled = scaler.transform(X_test)
    penalty=None if args.penalty =="None" else args.penalty
    modelo=LogisticRegression(
        penalty=penalty,
        C=args.C,
        max_iter=args.max_iter,
        random_state= args.random_state)
    
    modelo.fit(X_train_scaled, y_train)
    
    return (modelo, scaler)

# Evaluación de modelo
def evaluate_model(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred= model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    
    plt.savefig("confusion_matrix.png")
    plt.close()
    
    
    return (accuracy, report, y_pred)
    
    
def input_param():
    parser= argparse.ArgumentParser()
    
    parser.add_argument("--C",
                        type=float,
                        default=1.5,
                        help= "Regularización de Logistic Regression")
    
    parser.add_argument("--max_iter",
                        type=int,
                        default=5000,
                        help= "Número de máximo de iteraciones")
    
    parser.add_argument("--penalty",
                        type=str,
                        choices=["l1", "l2", "elasticnet", "None"],
                        default="l2",
                        help= "Tipo de penalización")
    
    parser.add_argument("--random_state",
                        type=int,
                        default=42,
                        help= "Semilla aleatoria")
    
    return parser.parse_args()
    
# MAIN

def main():
    path = kagglehub.dataset_download("teejmahal20/airline-passenger-satisfaction")
    
    train, test = load_data(path)
    
    args=input_param()
    
    train, test = data_cleaning(train, test)
    
    X_train, y_train = split_features_target(train)
    X_test, y_test = split_features_target(test)
    
    #test
    mlflow.set_experiment("Airline satisfaction regression - act2")
    
    # Inicia run con MLflow
    with mlflow.start_run():
        mlflow.log_param("C", args.C)
        mlflow.log_param("penalty", args.penalty)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("random_state", args.random_state)
        
        model, scaler = train_model(X_train, y_train, args)
    
        accuracy, report, y_pred = evaluate_model(model, scaler, X_test, y_test)
        
        # registro de hiperparámetros
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", report["weighted avg"]["precision"])
        mlflow.log_metric("recall", report["weighted avg"]["recall"])
        
        # registro de artefactos
        mlflow.log_artifact("confusion_matrix.png")
        
        # registro de modelo
        mlflow.sklearn.log_model(model, "modelo")
        
    
if __name__ == "__main__":
    main()
    
    