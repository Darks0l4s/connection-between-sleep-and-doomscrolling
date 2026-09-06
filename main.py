from app.streamlit import WindowApp
from src.data_loader import Loader
from src.visualition import Graph
from src.models import LinearML
import pandas as pd
def main():
    app = WindowApp()
    loader = Loader()
    loader.load_data('data/sleep_doomscrolling_habits.csv')
    df_user = loader.load_user_analytics()
    app.print_df(df_user)
    graph=Graph()
    app.print_graph(graph.visualization_user_analytics(df_user))
    model = LinearML()
    features = [
    'bedtime_screen_time_minutes', 'avg_doomscroll_session_minutes', 
    'doomscroll_sessions_per_night', 'phone_checks_per_night',
    'sleep_latency_minutes', 'caffeine_intake_mg_per_day', 
    'anxiety_score', 'stress_score', 'age', 'exercise_minutes_per_day',
    'occupation_status', 'keeps_phone_in_bedroom', 'uses_night_mode'
    ]   
    signs = loader.load_specific_data(features)
    target = loader.load_specific_data(['sleep_hours_per_night'])
    signs_column, k, b, mae, mse, r2 = model.train_linear(signs, target)
    app.print_graph(graph.predict_sleep(k, signs_column))
    text = f'MAE={mae}, MSE={mse}, R2_scroe={r2}'
    app.print_text(text)

if __name__=='__main__':
    main()