import numpy as np
import pandas as pd

def string_in_int(old_df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(old_df, drop_first=True)

def delete_nan(df: pd.DataFrame) -> pd.DataFrame:
    new_df=df.dropna()
    return new_df

class Loader:
    def __init__(self):
        self.df = pd.DataFrame
        self.df_not_nan =pd.DataFrame

    def load_data(self, path: str):
        try:
            self.df = pd.read_csv(path)
            print(self.df.head(5))
            self.df_not_nan = delete_nan(self.df)
        except FileNotFoundError:
            print('File not found')

    def get_df(self) -> pd.DataFrame:
        return self.df

    

    def load_user_analytics(self) -> pd.DataFrame:
        df_user = self.df[['age', 'gender', 'occupation_status', 'country_region', 'primary_device_used_at_night', 'uses_night_mode', 'keeps_phone_in_bedroom', 'consumes_negative_news_content', 'uses_sleep_tracking_app', 'doomscroller']]
        df_user=delete_nan(df_user)
        return df_user
    
    def load_specific_data(self, category: list):
        return self.df_not_nan[category]

    
    
