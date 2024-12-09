import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from dataclasses import dataclass
from utils import transform_text, save_obj

# ML libraries
import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
# from sklearn.compose import ColumnTransformer # type: ignore

# create data transformation config 
@dataclass
class DataTransformationConfig:
    preprocessor_obj : str = os.path.join('artifacts', 'feature_engineering.pkl')  
    cleaned_data_obj_path : str = os.path.join('artifacts', 'cleaned_data.csv')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def clean_data(self, data_path):
        try:
            df = pd.read_csv(data_path)
            df.drop(columns=['column_3','column_4','column_5'], inplace=True)
            df.rename(columns={'v1': 'Target', 'v2':'SMS'}, inplace=True)
            df  = df.drop_duplicates(keep='first')
            
            logging.info('Data cleaning completed')
            df.to_csv(self.data_transformation_config.cleaned_data_obj_path)
            
            return df
        except Exception as e:
            raise CustomException(e,sys)

    def data_preprocessing(self,data_frame):
        try:
            df = data_frame
            df['transformed_text'] = df['SMS'].apply(transform_text)
            encoder = LabelEncoder()
            df['Target'] = encoder.fit_transform(df['Target'])

            return df

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self, data):
        try:
            df = data
            preprocessor = TfidfVectorizer(max_features=3000)
            X = preprocessor.fit_transform(df['transformed_text']).toarray()
            y = df['Target'].values

            logging.info('TF idf transformation completed')
            
            #train test split
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 2)
            logging.info('Train test split completed')

            save_obj(
                file_path= self.data_transformation_config.preprocessor_obj,
                obj = preprocessor
            )
            logging.info('Preprocessor pkl file saved')
            
            return(
                X_train,
                X_test,
                y_train,
                y_test
            )

        except Exception as e:
            raise CustomException(e,sys)
        