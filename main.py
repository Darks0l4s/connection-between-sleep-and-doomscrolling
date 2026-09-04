from app.streamlit import WindowApp
from src.data_loader import Loader
from src.visualition import Graph

def main():
    app = WindowApp()
    loader = Loader()
    loader.load_data('data/sleep_doomscrolling_habits.csv')
    df_user = loader.load_user_analytics()
    app.print_df(df_user)
    graph=Graph()
    app.print_graph(graph.visualization_user_analytics(df_user))


if __name__=='__main__':
    main()