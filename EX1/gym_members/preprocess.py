import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import kagglehub

# complete loading, preparation, optional feature engineering and preprocessing pipeline
PATH = 'valakhorasani/gym-members-exercise-dataset'

class Preprocessing():
    """
    class for preprocessing
    main purpose is so we can later backtransform if we'd like
    """

    def __init__(self):
        self.le = LabelEncoder()
        self.scaler = StandardScaler()

    def load_data(self, download_path:str = PATH):
        """
        loads data
        """
        path = kagglehub.dataset_download(download_path)
        df = pd.read_csv(f'{path}/gym_members_exercise_tracking.csv')
        return df

    def feature_engineering(self, df: pd.DataFrame):
        """
        adds the new features discussed in Ex0
        """
        df['Exertion'] = (df['Avg_BPM']-df['Resting_BPM']) * df['Session_Duration (hours)']
        df['Intesity_Ratio'] = (df['Avg_BPM'] / df['Max_BPM']).clip(0,1) # shouldn't need clip, but just to be safe

        return df

    def preprocess_data(self, df:pd.DataFrame, split:bool = True, scale:bool=False):
        """
        preprocessing data
        split: bool - whether split data or don't
        scale: bool - whether scale the data or don't, some methods don't require it.

        encodes gender, and seperates features from target
        """
        # encode gender to binary values
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        # one hot encode workout_type
        one_hot_workout = pd.get_dummies(df['Workout_Type'])
        df = df.drop(columns=['Workout_Type'])
        df = df.join(one_hot_workout)


        # seperate features from target
        X = df.drop('Experience_Level', axis=1)
        y = df['Experience_Level']

        # encode labels
        y = self.le.fit_transform(y)

        if split:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if scale:
                X_train = self.scaler.fit_transform(X_train)
                X_test = self.scaler.transform(X_test)
            
            return X_train, X_test, y_train, y_test
        
        return X, y

    def pipeline(self, download_path, split, scale):
        """
        run the whole thing at once
        """
        df = self.load_data(download_path)
        df = self.feature_engineering(df)
        return self.preprocess_data(df, split=split, scale=scale)

def main():
    pp = Preprocessing()
    pp.pipeline(PATH, False, False)

if __name__ == "__main__":
    main()