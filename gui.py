import streamlit as st
import data

data.load_data()

def setup():
    st.write("""
# My first app

**Hello** *world!* """)
    df=data.return_data(10)
    st.dataframe(df, use_container_width=True)