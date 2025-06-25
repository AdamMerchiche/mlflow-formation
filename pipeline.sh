#!/bin/bash 

# Exit immediately if error (avec le set -e) si non zero status
set -e

# Setting variables (donner des valeurs par défaut) -- Variables à retirer-peu utiles si on a mlflow Project file 
input_data=${input_data:="data/raw/iris_2025-06-23 15:46:23.488049.parquet"}
output_folder=${output_folder:='data/processed'}
training_data=${training_data:='data/processed/train.parquet' }
tracking_uri=${tracking_uri:="http://127.0.0.1:5000"}
model_registry_name=${model_registry_name:="TRACKING_RF_FINAL"}
test_data=${test_data:='data/processed/test.parquet'}

# On regarde si les variables sont bien mentionnées
if [ "$#" -lt 1 ] ; then 
    echo "Need input data and output folder"
    exit 1
fi




# Step 1 : Prepare data
echo "Preparing data..."
# python3 scripts/prepa_data.py --input_data $input_data --output_folder $output_folder
#python3 scripts/prepare_data.py --input_data "$1" --output_folder "$2" -> On met des numéros si on ne set pas les variables au dessus par exemple

# Step 2 : Train and save model
echo "Training model..."
python3 scripts/train_model.py --training_data $training_data --model_registry_name $model_registry_name --tracking_uri $tracking_uri

# Step 3 : Test model
echo "Testing model..."
python3 scripts/evaluate_model.py --test_data $test_data --model_registry_name $model_registry_name --tracking_uri $tracking_uri

echo "Pipeline execution complete"

# Pour le rendre executable, il faut faire chmod u+x pipeline.sh dans le terminal
# On pourrait aussi lancer la pipeline en changeant uniquement un paramètre: par exemple tracking_uri 
# ./pipeline.sh --tracking_uri "URL" dans le cas où on veut changer de port ou de serveur 