import numpy as np
import pandas as pd

class Loader:
    def __init__(self):
        self.df = pd.DataFrame

    def load_data(self, path: str):
        try:
            self.df = pd.read_csv(path)
            print(self.df.head(5))
        except FileNotFoundError:
            print('File not found')

    def get_df(self) -> pd.DataFrame:
        return self.df

    def delete_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        new_df=df.dropna()
        return new_df

    def load_user_analytics(self) -> pd.DataFrame:
        df_user = self.df[['age', 'gender', 'occupation_status', 'country_region', 'primary_device_used_at_night', 'uses_night_mode', 'keeps_phone_in_bedroom', 'consumes_negative_news_content', 'uses_sleep_tracking_app', 'doomscroller']]
        df_user=self.delete_nan(df_user)
        return df_user