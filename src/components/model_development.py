import os
import sys
from dataclasses import dataclass
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from logger import logging
from utils import save_obj

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score

# Model trainer config
@dataclass
class ModelTrainerConfig:
    model_trainer_path = os.path.join('artifacts', 'model_trainer.pkl')
    logging.info('Model trainer config class completed')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,X_train, X_test, y_train, y_test):
        try:
            svc = SVC(kernel='sigmoid', gamma=1.0)
            knc = KNeighborsClassifier()
            mnb = MultinomialNB()
            dtc = DecisionTreeClassifier(max_depth=5)
            lrc = LogisticRegression(solver='liblinear', penalty='l1')
            rfc = RandomForestClassifier(n_estimators=50, random_state=2)
            abc = AdaBoostClassifier(n_estimators=50, random_state=2)
            etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
            gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)

            models  = {
                'SVC' : svc,
                'KN' : knc, 
                'NB': mnb, 
                'DT': dtc, 
                'LR': lrc, 
                'RF': rfc, 
                'AdaBoost': abc,
                'ExtraClass': etc,
                "GradientBoosting": gbdt
            }

            # Model Training
            model_report = {}
            for i in range(len(models)):
                model = list(models.values())[i]
                model.fit(X_train,y_train)
                Y_pred = model.predict(X_test)
                acc_score = accuracy_score(y_test, Y_pred)
                prec_score = precision_score(y_test, Y_pred)

                model_report[list(models.keys())[i]] = acc_score, prec_score
            
            logging.info('model training complete')

            # Best model score
            best_model_score = max(sorted(model_report.values()))

            #Best model name - gives the key of dictionary
            model_keys_list = list(model_report.keys())
            model_values_list = list(model_report.values())
            index_of_best_model = model_values_list.index(best_model_score) # lets say 2
            best_model_name = model_keys_list[index_of_best_model]

             # best model - gives the value of dictionary
            best_model = models[best_model_name]
            logging.info('Best model name declared')

            save_obj(
                file_path = self.model_trainer_config.model_trainer_path,
                obj = best_model
            )
            logging.info('Best model saved')

            #Prediction of the best model
            ypred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, ypred)
            precision = precision_score(y_test, ypred)
            logging.info('best model name and score printed')


            return best_model_name, accuracy, precision
        
        except Exception as e:
            raise CustomException(e,sys)
        