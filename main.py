from app.ui import WindowApp
import src.data_loader as dl
from src.visualization import Graph
from src.models import LinearML
import pandas as pd

def regression_analytics(app: WindowApp, loader: dl.Loader, graph: Graph, model: LinearML):
    features = [
    'bedtime_screen_time_minutes', 'avg_doomscroll_session_minutes', 
    'doomscroll_sessions_per_night', 'phone_checks_per_night',
    'sleep_latency_minutes', 'caffeine_intake_mg_per_day', 
    'anxiety_score', 'stress_score', 'age', 'exercise_minutes_per_day',
    'occupation_status', 'keeps_phone_in_bedroom', 'uses_night_mode'
    ]   
    signs = loader.load_specific_data(features)
    target = loader.load_specific_data(['sleep_hours_per_night'])
    signs = loader.encode_categorical(signs)
    app.render_model_selection(signs, target, model, graph)

def classifier_analytics(app: WindowApp, loader: dl.Loader, graph: Graph, model: LinearML):
    feature = [
        'age', 'bedtime_screen_time_minutes', 'total_daily_screen_time_hours',
        'doomscroll_sessions_per_night', 'avg_doomscroll_session_minutes',
        'phone_checks_per_night', 'keeps_phone_in_bedroom'
    ]
    signs = loader.load_specific_data(feature)
    signs = loader.encode_categorical(signs)
    target = loader.load_specific_data(['doomscroller'])
    app.render_classifier(signs, target, model, graph)

def main():
    app = WindowApp()
    loader = dl.Loader()
    graph=Graph()
    model = LinearML()
    loader.load_data('data/sleep_doomscrolling_habits.csv')
    df_user = loader.load_user_analytics()
    df = loader.get_df()
    app.print_df(df)
    app.print_graph(graph.visualization_user_analytics(df_user))
    regression_analytics(app, loader,  graph, model)
    classifier_analytics(app, loader, graph, model)


if __name__=='__main__':
    main()

