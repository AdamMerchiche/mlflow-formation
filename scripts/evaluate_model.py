""" Model evaluation module """
import joblib 
import pathlib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import argparse
import mlflow

def evaluate_model(
       test_data: str, 
       tracking_uri:str,
       model_registry_name: str): 
    """ Evaluate model"""

    # Load model and data - on type le modele pour être sûr d'utiliser la bonne fonction predict
    mlflow.set_tracking_uri(uri=tracking_uri)

    #  On récupère la dernière version du modèle importé sur MLFLOW dans notre registry name
    version = mlflow.search_model_versions(filter_string=f"name='{model_registry_name}'")[0].version
    print("VERSION", version)
    model = mlflow.pyfunc.load_model(model_uri=f'models:/{model_registry_name}/{version}')
    test_df = pd.read_parquet(test_data)

    # Extract labels 
    Xtest = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:,-1]

    # Evaluate 
    y_pred = model.predict(Xtest)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation Accuracy: {acc}")


if __name__== "__main__":

    # Parsing command line arguments
    parser=argparse.ArgumentParser(description='Test Data and model MLFLOW  Parser')
    # On ajoute les arguments (on en a 2)
    parser.add_argument(
        "--test_data", 
        type=str
    )

    parser.add_argument(
        "--model_registry_name", 
        type=str
    )

    parser.add_argument(
        "--tracking_uri", 
        type=str
    )


    #On ecoute les arguments
    args = parser.parse_args()


    evaluate_model(
        test_data=args.test_data, 
        model_registry_name=args.model_registry_name, 
        tracking_uri = args.tracking_uri    )
    

    #python3 scripts/evaluate_model.py --test_data 'data/processed/test.parquet' --tracking_uri 'http://127.0.0.1:5000'    --model_registry_name  'TRACKING_RF_FINAL'