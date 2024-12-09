import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from dataclasses import dataclass
import psycopg2

#ML libraries 
import pandas as pd
import numpy as np

#local importing
from data_transformation import DataTransformation

#Data ingestion config class to input the csv file
@dataclass #Directly define the class variable
class DataIngestionConfig:
    raw_data_path:str = os.path.join('artifacts', 'dataset.csv')
    predictor_variable_path: str = os.path.join('artifacts', 'predictors.csv')
    target_variable_path: str = os.path.join('artifacts','target.csv')

#Class to ingest the data
class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            #Connect to the database
            conn = psycopg2.connect(
                host = "localhost",
                port = 5432,
                database = "spam_message_db",
                user = "postgres",
                password = "@As998831"
            )
            
            query = 'SELECT * FROM mytable'
            
            # Pandas to execute the query and fetch data
            df = pd.read_sql_query(query, conn)
            logging.info('Imported the table in dataframe format')

            #Create the artifacts folder, where we have to store train.csv, test and raw
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)

            #saving the dataframe into csv file
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Data ingestion completed')
            
            return self.data_ingestion_config.raw_data_path

        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    # Data ingestion
    data_ingestion = DataIngestion()
    raw_data_path = data_ingestion.initiate_data_ingestion()

    #Data Transformation
    data_transformation = DataTransformation()
    cleaned_df = data_transformation.clean_data(data_path = raw_data_path)
    new_df = data_transformation.data_preprocessing(data_frame = cleaned_df)
    X_train,X_test,y_train,y_test = data_transformation.initiate_data_transformation(data = new_df)
    print(X_train.shape)