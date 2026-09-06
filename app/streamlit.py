import streamlit as st
from pandas import DataFrame
class WindowApp:
    def __init__(self):
        st.set_page_config(page_title='Sleep Analytic')
        st.title('Hello')

    def print_graph(self, fig):
        st.pyplot(fig)

    def print_df(self, df: DataFrame):
        st.dataframe(df, use_container_width=True)

    def print_text(self, text: str):
        st.text(text)