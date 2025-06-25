# Docker c'est bien en fin de production. Mais fondamental de créer un environnement virtuel pour partager le travail. 
# C'est mieux d'avoir un environnement virtuel par projet - 1 enviro par projet 
# on peut utiliser l'outil PipEnv, conçu sur VirtualEnv

import subprocess, sys, os

# Execution & installation des librairies 
packages = ['scikit-learn', 'pandas', 'joblib', 'datetime']
for package in packages: 
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# Import des librairies
from sklearn import datasets
import pandas as pd 
import joblib 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split 
from datetime import datetime


# Etre le plus modulaire possible

# Versionning des données absente : en ne téléchargeant pas les données dans un dossier local (ou cloud), on ne pourra prendre 
# en compte les nouvelles versions de données
folder = '../data/raw'
os.makedirs(folder, exist_ok=True)

# Import des données 
def load_data(): 
    iris = datasets.load_iris()
    df = pd.DataFrame(data=iris['data'], columns= iris['feature_names'])
    df['type']= iris['target']
    df.to_parquet(os.path.join(folder, f'iris_{datetime.now()}.parquet'))
    return df 

# Process des données 
def process_data(df): 
    X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
       'petal width (cm)']]
    y = df['type'] 
    return X, y

# Entrainement 
def train_model(X, y): 
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=42, shuffle=True, train_size=0.2)
    model = RandomForestClassifier()
    model.fit(Xtrain, ytrain)
    return Xtrain, Xtest, ytrain, ytest, model 

# Evaluation 
def evaluate(model, Xtest, ytest):
    ypred = model.predict(Xtest)
    print('Model accuracy', accuracy_score(ytest, ypred))
    print(classification_report(ytest, ypred))
    return accuracy_score(ytest, ypred)

# Pipeline générale
def pipeline(): 
    df = load_data()
    X, y = process_data(df)
    _, Xtest, _, ytest, model = train_model(X, y)
    acc = evaluate(model, Xtest, ytest)
    joblib.dump(model, 'model_exo1.pkl')

# Lancement de la pipeline - création du modèle de données, entraînement et save
pipeline()