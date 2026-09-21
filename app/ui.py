import streamlit as st
from pandas import DataFrame
from src.visualization import Graph
from src.models import ModelTrainer
from matplotlib.figure import Figure
from numpy import ndarray
import os
import joblib
class WindowApp:
    def __init__(self):
        st.set_page_config(page_title='Sleep Analytic')
        st.title('Hello')

    def print_graph(self, fig: Figure):
        st.pyplot(fig)

    def print_df(self, df: DataFrame):
        st.dataframe(df, width='stretch')

    def print_text(self, text: str):
        st.text(text)

    def render_regression_tab(self, signs: DataFrame, target: DataFrame, model: ModelTrainer, graph: Graph, mode: str):
        self.print_text('Linear Regression')
        file_path = f"models/{'regression'}-{mode}-res.pkl"
        if os.path.exists(file_path):
            res =joblib.load(file_path)
        else:
            res = model.train_linear(signs, target, mode)
        self.print_graph(graph.coef_visual(res.k, res.signs_column))
        text = f'MAE={res.mae}, MSE={res.mse}, R2_score={res.r2}'
        self.print_text(text)
        self.print_graph(graph.prediction_vs_actual(res.y_real, res.y_predict))

    def render_model_selection(self, signs: DataFrame, target: DataFrame, model: ModelTrainer, graph: Graph):

        st.title('Regression')
        self.print_graph(graph.show_dependence_parameter(signs, target))
        tab_linear, tab_ridge, tab_lasso, tab_random, tab_gradient = st.tabs([
        "📈 Linear Regression", 
        "🛡️ Ridge Regression", 
        "🎯 Lasso Regression", 
        "🌲 Random Forest", 
        "⚡ Gradient Boosting"
    ])
        with tab_linear:
            self.render_regression_tab(signs, target, model, graph, 'linear')
        with tab_ridge:
            self.render_regression_tab(signs, target, model, graph, 'ridge')
        with tab_lasso:
            self.render_regression_tab(signs, target, model, graph, 'lasso')
        with tab_random:
            self.render_regression_tab(signs, target, model, graph, 'random')
        with tab_gradient:
            self.render_regression_tab(signs, target, model, graph, 'gradient')

    def render_classifier_tab(self, signs: DataFrame, target: DataFrame, model: ModelTrainer, graph: Graph, mode: str):

        file_path = f"models/{'classifier'}-{mode}-res.pkl"
        if os.path.exists(file_path):
            res =joblib.load(file_path)
        else:
            res = model.train_classifier(signs, target, mode)
        st.metric("Accuracy", f"{res.accuracy * 100:.2f}%")
        st.write("📊 Detailed report by class:")
        st.dataframe(res.report)
        st.pyplot(graph.coef_visual(res.importances, signs.columns))
        self.print_graph(graph.heatmap(res.matrix))

    def render_classifier(self, signs: DataFrame, target: ndarray, model: ModelTrainer, graph: Graph):
        st.title('Classifier')
        tab_logistic, tab_random, tab_histgradient = st.tabs([
                "LogisticRegression", 
                "RandomForestClassifier", 
                "HistGradientBoostingClassifier"
            ])
        with tab_logistic:
            self.render_classifier_tab(signs, target, model, graph, 'logistic')
        with tab_random:
            self.render_classifier_tab(signs, target, model, graph, 'random')
        with tab_histgradient:
            self.render_classifier_tab(signs, target, model, graph, 'histgradient')
