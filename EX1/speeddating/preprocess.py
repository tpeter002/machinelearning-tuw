from pathlib import Path
from typing import Optional, Union
import pandas as pd
from scipy.io import arff
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

class Preprocessing():
     def __init__(self):
        self.scaler = RobustScaler()

     def _to_text(self,value):
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8", errors="ignore")
        return value


     def correct_float_to_int(self, df,):
        
        float_cols = df.select_dtypes(include=[np.float64, np.float32]).columns
        
        
        target_dtype = np.int32
        
        for col in float_cols:
            
            is_whole_number = (df[col].dropna() % 1 == 0).all()
            
            if is_whole_number:
                df[col] = df[col].astype(target_dtype)

        return df


     def load_preprocessed_speeddating(
        self,
        data_path: Optional[Union[str, Path]] = None,
        drop_ambitious:bool = False,
        split = True,
        scale = True
     ) -> pd.DataFrame:
        base_dir = Path(__file__).parent
        arff_path = (
            Path(data_path) if data_path is not None else base_dir / "speeddating.arff"
        )

        data, _meta = arff.loadarff(str(arff_path))
        dating = pd.DataFrame(data)

        # Decode object (bytes) columns to str
        object_cols = dating.select_dtypes(include=["object"]).columns.tolist()
        for col in object_cols:
            dating[col] = dating[col].map(self._to_text)

        # Drop columns
        for col in ["has_null", "wave"]:
            if col in dating.columns:
                dating.drop(columns=[col], inplace=True)

        # Type conversions and new feature
        if "samerace" in dating.columns:
            dating["samerace"] = dating["samerace"].astype(int)

        if "age" in dating.columns and "age_o" in dating.columns:
            dating["age_diff"] = dating["age"] - dating["age_o"]
            for col in ["age", "age_o", "d_age", "d_d_age"]:
                if col in dating.columns:
                    dating.drop(columns=[col], inplace=True)

        # Drop all columns starting with 'd_'
        d_cols = [c for c in dating.columns if c.startswith("d_")]
        if d_cols:
            dating.drop(columns=d_cols, inplace=True)

        # Drop 'field'
        if "field" in dating.columns:
            dating.drop(columns=["field"], inplace=True)

        # Binarize 'met'
        if "met" in dating.columns:
            dating["met"] = dating["met"].replace({3.0: 0, 5.0: 0, 6.0: 0, 7.0: 0, 8.0: 0})

        # Drop decisions
        for col in ["decision_o", "decision"]:
            if col in dating.columns:
                dating.drop(columns=[col], inplace=True)

        # Missing values
        if "expected_num_interested_in_me" in dating.columns:
            dating.drop(columns=["expected_num_interested_in_me"], inplace=True)
        
        
        cols_to_drop = ['shared_interests_partner', 'shared_interests_o', 'expected_num_matches']
        
        if drop_ambitious:
            cols_to_drop.extend(['ambition_partner', 'ambitous_o'])
        dating = dating.drop(cols_to_drop, axis = 1)
        nan_columns = dating.columns[dating.isnull().any()].tolist()
        dating = dating.dropna(subset = nan_columns)
        
        
        dating['match'] = dating['match'].astype(np.int32)
        
        object_columns = dating.select_dtypes(include=['object']).columns
        categorical = list(object_columns)
        if categorical:
            dating = pd.get_dummies(dating, columns=categorical, prefix=categorical, drop_first=False)

        # Convert boolean columns to 0/1
        bool_cols = dating.select_dtypes(include=['bool']).columns
        if len(bool_cols):
            dating[bool_cols] = dating[bool_cols].astype(np.int8)
            
        dating = self.correct_float_to_int(dating)
            
        if split:
            X = dating.drop(['match'], axis = 1)
            y = dating['match']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)
            if scale:
                X_train = self.scaler.fit_transform(X_train)
                X_test = self.scaler.transform(X_test)

            return X_train, X_test, y_train, y_test
        return dating
