import streamlit as st
from pandas import DataFrame
from src.visualition import Graph
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
            self.print_text('Linear Regression')
            signs_column, k, b, mae, mse, r2 = model.train_linear(signs, target, mode='linear')
            self.print_graph(graph.predict_sleep(k, signs_column))
            text = f'MAE={mae}, MSE={mse}, R2_score={r2}'
            self.print_text(text)
            self.print_graph(graph.coincidence_values(model.y_real, model.y_predict))
        with tab_ridge:
            self.print_text('Ridge Regression')
            signs_column, k, b, mae, mse, r2 = model.train_linear(signs, target, mode='linear')
            self.print_graph(graph.predict_sleep(k, signs_column))
            text = f'MAE={mae}, MSE={mse}, R2_score={r2}'
            self.print_text(text)
            self.print_graph(graph.coincidence_values(model.y_real, model.y_predict))
        with tab_lasso:
            self.print_text('Lasso Regression')
            signs_column, k, b, mae, mse, r2 = model.train_linear(signs, target, mode='linear')
            self.print_graph(graph.predict_sleep(k, signs_column))
            text = f'MAE={mae}, MSE={mse}, R2_score={r2}'
            self.print_text(text)
            self.print_graph(graph.coincidence_values(model.y_real, model.y_predict))
        with tab_random:
            self.print_text('Random Forest')
            signs_column, k, b, mae, mse, r2 = model.train_linear(signs, target, mode='linear')
            self.print_graph(graph.predict_sleep(k, signs_column))
            text = f'MAE={mae}, MSE={mse}, R2_score={r2}'
            self.print_text(text)
            self.print_graph(graph.coincidence_values(model.y_real, model.y_predict))
        with tab_gradient:
            self.print_text('Gradient Boosting')
            signs_column, k, b, mae, mse, r2 = model.train_linear(signs, target, mode='linear')
            self.print_graph(graph.predict_sleep(k, signs_column))
            text = f'MAE={mae}, MSE={mse}, R2_score={r2}'
            self.print_text(text)
            self.print_graph(graph.coincidence_values(model.y_real, model.y_predict))

    def render_classifier(self, signs: DataFrame, target: DataFrame, model: LinearML, graph: Graph):
        st.title('Classifier')
        importances, accuracy, report, matrix = model.train_classifier(signs, target)
        st.metric("Общая точность (Accuracy)", f"{accuracy * 100:.2f}%")

        st.write("📊 Детальный отчет по классам:")
        st.dataframe(report)
        st.pyplot(graph.predict_sleep(importances, signs.columns))