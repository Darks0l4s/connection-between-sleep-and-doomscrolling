import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.figure as matfig
class Graph:
    def __init__(self):
        pass

    def visualization_user_analytics(self, df: pd.DataFrame) -> matfig.Figure:
        fig, ax = plt.subplots(1, 1, figsize = (15,15))
        ax = plt.pie(df['gender'].value_counts(), labels = df['gender'].value_counts().index, autopct='%1.1f%%')
        fig.tight_layout()
        return fig