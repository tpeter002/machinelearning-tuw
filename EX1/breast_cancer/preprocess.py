import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

COMPETITION_ID = '184-702-tu-ml-2025w-breast-cancer'
EXTRACTION_DIR = './data'

class Preprocessing():
    """
    class for preprocessing
    """
    
    def __init__(self):
        self.scaler = RobustScaler()
    
    def load_data(self):
        api = KaggleApi()
        api.authenticate()
        
        # Download the zip
        api.competition_download_files(
            competition=COMPETITION_ID, 
            path='.', 
            force=False, 
            quiet=True
        )
        
        # Extract the zip
        zip_filename = f'{COMPETITION_ID}.zip'
        if os.path.exists(zip_filename):
            if not os.path.exists(EXTRACTION_DIR):
                os.makedirs(EXTRACTION_DIR)

            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(EXTRACTION_DIR)
                
            os.remove(zip_filename)
                
        df = pd.read_csv(os.path.join(EXTRACTION_DIR, 'breast-cancer-diagnostic.shuf.lrn.csv'))
        return df
    
    def preprocess_data(self, df: pd.DataFrame, split:bool = True, scale:bool = True, ridge = False):
        """
        preprocessing data
        split: bool - if True, a train/test split is done
        scale: bool - if True, the predictors are scaled
        ridge: bool - if True, the target class labels are set to -1 and 1
        df: the DataFrame to work on
        """
        if ridge:
            df['class'] = df['class'].map({ True: 1, False: -1})
        
        else:
            df['class'] = df['class'].map({ True: 1, False: 0})
        
        # Some columns have leading whitspaces
        df.columns = df.columns.str.strip()
        # Removing columns that cause multicollinearity
        df = df.drop(columns=['perimeterMean', 'areaMean', 'perimeterWorst', 'areaWorst', 'perimeterStdErr', 'areaStdErr', 'concavePointsMean', 'textureWorst', 'radiusMean'], axis = 1)
        
        X = df.drop(['ID', 'class'], axis=1)
        y = df['class']
        
        if split:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)

            if scale: 
                X_train = self.scaler.fit_transform(X_train)
                X_test = self.scaler.transform(X_test)
                
            return X_train, X_test, y_train, y_test
        
        else:
            if scale:
                X = self.scaler.fit_transform(X)
            
            return X, y
        
        