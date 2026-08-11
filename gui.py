import streamlit as st
import data
import graphs

data.load_data()

def toggle_text():
    st.session_state.show_user_analytics = not st.session_state.show_user_analytics

def setup():
    st.write("""
# Sleep analytics

**Hello** *world!* """)
    df=data.return_data(10)
    st.dataframe(df, use_container_width=True)

    if 'show_user_analytics' not in st.session_state:
        st.session_state.show_user_analytics = False

    st.button(
        label='User analytics',
        on_click=toggle_text
        )
    if st.session_state.show_user_analytics:
        st.pyplot(graphs.user_analytics_graph(data.df))
    