""" Model train module
"""
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import joblib
import pathlib
import argparse
import mlflow


def train_model(
        training_data: str, 
        model_registry_name: str, 
        tracking_uri: str
) -> None: 
    """ Model training function
    """

    # On connecte le serveur de tracking ici. Si le serveur n'est pas sur notre machine, en local, il faut bien le spécifier
    mlflow.set_tracking_uri(uri=tracking_uri)
    mlflow.set_experiment("Exo 2.3 - Pipeline with Mlflow")
    # mlflow.set_tag("Training Info", "RandomForest model for iris data with AUTOLOG")

    mlflow.sklearn.autolog(registered_model_name=model_registry_name, 
                           log_input_examples=True, log_model_signatures=True)

    # Load training dataset 
    train_df = pd.read_parquet(training_data)

    # Extract features and labels 
    Xtrain = train_df.iloc[:,:-1]
    y_train = train_df.iloc[:, -1]

    # Training model
    model = RandomForestClassifier(random_state=42)
    model.fit(Xtrain,y_train)
    
    

    # Si le nom du module (name) == nom du fichier executé (main)
if __name__== "__main__":

    # Parsing command line arguments
    parser=argparse.ArgumentParser(description='Training Data and model registry  Parser')
    # On ajoute les arguments (on en a 2)
    parser.add_argument(
        "--training_data", 
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


    train_model(
        training_data=args.training_data, 
        model_registry_name=args.model_registry_name, 
        tracking_uri=args.tracking_uri
    )