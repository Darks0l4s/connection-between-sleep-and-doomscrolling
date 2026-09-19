import streamlit as st
from pandas import DataFrame
from src.visualization import Graph
from src.models import LinearML
from matplotlib.figure import Figure

class WindowApp:
    def __init__(self):
        st.set_page_config(page_title='Sleep Analytic')
        st.title('Hello')

    def print_graph(self, fig: Figure):
        st.pyplot(fig)

    def print_df(self, df: DataFrame):
        st.dataframe(df, use_container_width=True)

    def print_text(self, text: str):
        st.text(text)

    def render_regression_tab(self, signs: DataFrame, target: DataFrame, model: LinearML, graph: Graph, mode: str):
        self.print_text('Linear Regression')
        signs_column, k, b, mae, mse, r2 = model.train_linear(signs, target, mode='linear')
        self.print_graph(graph.coef_visual(k, signs_column))
        text = f'MAE={mae}, MSE={mse}, R2_score={r2}'
        self.print_text(text)
        self.print_graph(graph.coincidence_values(model.y_real, model.y_predict))

    def render_model_selection(self, signs: DataFrame, target: DataFrame, model: LinearML, graph: Graph):

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

    def render_classifier_tab(self, signs: DataFrame, target: DataFrame, model: LinearML, graph: Graph, mode: str):
        importances, accuracy, report, matrix = model.train_classifier(signs, target, mode)
        st.metric("Accuracy", f"{accuracy * 100:.2f}%")
        st.write("📊 Detailed report by class:")
        st.dataframe(report)
        st.pyplot(graph.coef_visual(importances, signs.columns))
        self.print_graph(graph.heatmap(matrix))

    def render_classifier(self, signs: DataFrame, target: DataFrame, model: LinearML, graph: Graph):
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