import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.figure as matfig

class Graph:
    def __init__(self):
        pass

    def visualization_user_analytics(self, df: pd.DataFrame) -> matfig.Figure:
        fig, axes = plt.subplots(2, 2, figsize = (20,20))

        axes[0, 0].pie(df['gender'].value_counts(), labels = df['gender'].value_counts().index, autopct='%1.1f%%')
        axes[0, 0].set_title("Gender Distribution")

        axes[0, 1].bar(df['age'].value_counts().index, df['age'].value_counts().values)
        axes[0, 1].set_xlabel('Age')
        axes[0, 1].set_ylabel('Count users')
        axes[0, 1].set_title('Age analytics')
        axes[0, 1].grid()

        axes[1, 0].barh(df['occupation_status'].value_counts().index, df['occupation_status'].value_counts().values, color='Green')  
        axes[1, 0].set_xlabel('Users count')
        axes[1, 0].set_ylabel('Occupation status')
        axes[1, 0].set_title('Occupation status analytics')

        axes[1, 1].barh(df['country_region'].value_counts().index, df['country_region'].value_counts().values, color='red')
        axes[1, 1].set_xlabel('Users count')
        axes[1, 1].set_ylabel('Region')
        axes[1, 1].set_title('Region analytics')

        fig.tight_layout()
        return fig