import streamlit as st

class WindowApp:
    def __init__(self):
        st.set_page_config(page_title='Sleep Analytic')
        st.title('Hello')

    def print_user_analytics(self, fig):
        st.pyplot(fig)