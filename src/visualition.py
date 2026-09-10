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

    def predict_sleep(self, coef: list, coef_name: list) -> matfig.Figure:
        coef = coef.flatten()
        fig, axes = plt.subplots(1,1, figsize=(20,20))
        axes.barh(coef_name, coef)
        axes.invert_yaxis()  
        return fig

    def show_dependence_parameter(self, parameters: pd.DataFrame, target: pd.DataFrame) -> matfig.Figure:
        from math import ceil
        feature_count = parameters.shape[1]
        size = ceil(feature_count**(0.5))
        fig, axes = plt.subplots(size,size, figsize=(20,20))
        i=0
        axes= axes.flatten()
        for feature_name, feature_data in parameters.items():
            axes[i].scatter(feature_data, target)
            axes[i].set_xlabel(feature_name)
            axes[i].set_ylabel(target.columns[0])
            i+=1
        return fig

    def coincidence_values(self, y_predict: pd.DataFrame, y_real: pd.DataFrame) -> matfig.Figure:
        from numpy import linspace
        fig, axes = plt.subplots(figsize=(15,15))
        axes.scatter(y_real, y_predict)
        axes.set_xlabel('Real data')
        axes.set_ylabel('Predict data')
        y_linear= linspace(6, 10, 10)
        axes.plot(y_linear, y_linear, color='red')
        return fig