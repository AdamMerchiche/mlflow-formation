""" Data Preparation module"""

import pathlib
import argparse
import numpy as np
import pandas as pd

def prepare_data(input_data: str, output_folder:str    
) -> None: 
    """ Apply preparation process on input data and save output in output folder
    Args: 
        input_data(str): _description_
        output_folder(str): _description_

    """
    iris_df = pd.read_parquet(input_data)
    train, validate, test = np.split(iris_df.sample(frac=1, random_state=42), [
    int(0.6 * len(iris_df)), int(0.8 * len(iris_df))
])

    train.to_parquet(f'{output_folder}/train.parquet')

    # Avec pathlib 
    train.to_parquet(pathlib.Path(output_folder).joinpath('train.parquet'))
    test.to_parquet(f'{output_folder}/test.parquet')
    validate.to_parquet(f'{output_folder}/validate.parquet')

# Si le nom du module (name) == nom du fichier executé (main)
if __name__== "__main__":

    # Parsing command line arguments
    parser=argparse.ArgumentParser(description='Input Data Parser')
    # On ajoute les arguments (on en a 2)
    parser.add_argument(
        "--input_data", 
        type=str
    )
    parser.add_argument(
        "--output_folder", 
        type=str
    )

    #On ecoute les arguments
    args = parser.parse_args()


    prepare_data(
        input_data=args.input_data, 
        output_folder= args.output_folder
    )