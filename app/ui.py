import streamlit as st
from pandas import DataFrame
from src.visualization import Graph
from src.models import ModelTrainer
from matplotlib.figure import Figure
from numpy import ndarray
import os
import joblib
from src.data_loader import Loader

@st.cache_data
def cache_render_regression_tab(mode, _model, signs, target):
    file_path = f"models/{'regression'}-{mode}-res.pkl"
    if os.path.exists(file_path):
        res =joblib.load(file_path)
    else:
        res = _model.train_linear(signs, target, mode)
    text = f'MAE={res.mae}, MSE={res.mse}, R2_score={res.r2}'
    return res, text

@st.cache_data
def cache_render_classifier_tab(mode, _model, signs, target):
    file_path = f"models/{'classifier'}-{mode}-res.pkl"
    if os.path.exists(file_path):
        res =joblib.load(file_path)
    else:
        res = _model.train_classifier(signs, target, mode)
    return res

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
        res, text = cache_render_regression_tab(mode, model, signs, target)
        self.print_text('Linear Regression')
        self.print_graph(graph.coef_visual(res.k, res.signs_column))
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
        res = cache_render_classifier_tab(mode, model, signs, target)
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

    @st.fragment
    def doomscraller_test(self):
        st.title('You are doomscraller?')
        with st.container(border=True):
            option = st.selectbox("Select a model for training:",
            ("Random Forest", "Logistic Regression", "Gradient Boosting"))
            match option:
                case 'Gradient Boosting':
                    mode='histgradient'
                case 'Logistic Regression':
                    mode='logistic'
                case 'Random Forest':
                    mode='random'
                case _:
                    mode='logistic'
            age = st.slider('Age', 0, 100, 18, 1)
            bedtime_screen_time_minutes = st.slider('Bedtime screen time minutes:',0, 240, 0, 1)
            total_daily_screen_time_hours = st.slider('Total daily screen time hours',0, 24, 0, 1)
            doomscroll_sessions_per_night = st.slider('Doomscroll sessions per night',0, 20, 0, 1)
            avg_doomscroll_session_minutes = st.slider('Avg doomscroll session minutes',0, 240, 0, 1)
            phone_checks_per_night = st.slider('Phone checks per night',0, 10, 0, 1)
            keeps_phone_in_bedroom = st.radio('Keeps phone in bedroom',['Yes', 'No'])

            data = DataFrame([{
                'age': age,
                'bedtime_screen_time_minutes': bedtime_screen_time_minutes,
                'total_daily_screen_time_hours': total_daily_screen_time_hours,
                'doomscroll_sessions_per_night': doomscroll_sessions_per_night,
                'avg_doomscroll_session_minutes': avg_doomscroll_session_minutes,
                'phone_checks_per_night': phone_checks_per_night,
                'keeps_phone_in_bedroom': keeps_phone_in_bedroom
            }])

            file_path = f"models/{'classifier'}-{mode}-model.pkl"
            if os.path.exists(file_path):
                model = joblib.load(file_path)
            else:
                st.cache_resource.clear()
                st.cache_data.clear()
                st.session_state.clear() 
                st.rerun()
            loader =Loader()
            data =loader.encode_categorical(data)
            data = data.reindex(
                columns=model.feature_names_in_,
                fill_value=0
            )
            res=ModelTrainer.predict_classifier(model, data)
            # print(res)
            st.subheader("📊 Habit analysis results:")
            if res[0] == 'Yes':
                st.error("🚨 The model has classified you as a **doomscroller**! It is recommended to put your phone away before bed.")
            else:
                st.success("✅ Everything looks great! The model believes you are **in control** of your nighttime habits.")

        