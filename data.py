import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def doomscrolling_sleep():
    plt.figure(figsize=(20,15))

    signs = ['total_doomscroll_minutes',
    'bedtime_screen_time_minutes',
    'total_daily_screen_time_hours',
    'phone_checks_per_night',
    'number_of_night_wakeups',
    'anxiety_score',
    'stress_score',
    'exercise_minutes_per_day',
    'caffeine_intake_mg_per_day',
    'number_of_news_apps_used',
    'age',
    # 'weekly_sleep_debt_hours',
    'gender_encode',
    'occupation_status_encode'
    ]
    target_variable = 'sleep_latency_minutes'
    df_model = df
    df_model['gender_encode']=df_model['gender'].map({
                                   'Male': 1,
                                   'Female': 2,
                                   'Prefer not to say': 0
       })
    df_model['occupation_status_encode']=df_model['occupation_status'].map({
        'Unemployed': 1,
        'Student': 2,
        'Employed Part-time': 3,
        'Employed Full-time':4
    })
    df_model = df.dropna(subset = signs+[target_variable])
    
    X=df_model[signs]
    # y=df_model['sleep_hours_per_night']
    y = df_model[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2
    )
    model=LinearRegression()
    model.fit(X_train, y_train)
    result = np.column_stack((signs, model.coef_)).flatten()
    print(result)
    y_pred = model.predict(X_test)
    # y_pred = np.floor(y_pred+0.5)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)   
    print(f'MAE={mae} MSE={mse} r2={r2}')

    plt.subplot(2, 2, 1)
    # plt.grid()
    plt.scatter(x=y_test, y=y_pred, c='r')
    x_line=np.linspace(0, df[target_variable].max())
    plt.plot(x_line, x_line, c='b')
    plt.title('Model spread')
    plt.xlabel('Real Y')
    plt.ylabel('Predict Y')

    plt.subplot(2,2, 2)
    plt.bar(x=signs, height=model.coef_)

    target_variable = 'sleep_hours_per_night'
    df_model = df.dropna(subset = signs+[target_variable])
    
    X=df_model[signs]
    # y=df_model['sleep_hours_per_night']
    y = df_model[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2
    )
    model=LinearRegression()
    model.fit(X_train, y_train)
    result = np.column_stack((signs, model.coef_)).flatten()
    print(result)
    y_pred = model.predict(X_test)
    # y_pred = np.floor(y_pred+0.5)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)   
    print(f'MAE={mae} MSE={mse} r2={r2}')

    plt.subplot(2, 2, 3)
    # plt.grid()
    plt.scatter(x=y_test, y=y_pred, c='r')
    x_line=np.linspace(0, df[target_variable].max())
    plt.plot(x_line, x_line, c='b')
    plt.title('Model spread')
    plt.xlabel('Real Y')
    plt.ylabel('Predict Y')

    plt.subplot(2,2, 4)
    plt.bar(x=signs, height=model.coef_)
    
    plt.show()
    plt.cla()

def user_analytics():
    plt.figure(figsize=(20,15))

    plt.subplot(2,2, 1)
    plt.title('Gender of People')
    plt.pie(df['gender'].value_counts(), labels=df['gender'].unique(), autopct='%1.1f%%')
    plt.legend(loc='upper left', bbox_to_anchor=(-0.3, 1))

    plt.subplot(2,2, (2,4))
    plt.title('People age')
    plt.bar(x=df['age'].unique(), height=df['age'].value_counts())
    plt.xlabel('Age')
    plt.ylabel('Count people')

    plt.subplot(2,2, 3)
    plt.title('People live country')
    plt.barh(y=df['country_region'].unique(), width=df['country_region'].value_counts())
    plt.xlabel('People count')
    plt.ylabel('Country')
    plt.show()
    plt.cla()

def load_data():
    try:
        global df
        df = pd.read_csv('sleep_doomscrolling_habits.csv')
    except FileNotFoundError:
        print("File Not found")
    except Exception as e:
        print(f"Error: '{e}'")
    else:
        df['total_doomscroll_minutes'] = (
                df['doomscroll_sessions_per_night'] * df['avg_doomscroll_session_minutes']
        )
        print('Successful')
        print(df.head(5))
    return df

def menu():
    print('=====MENU=====')
    print('0. Exit\n'
        '1. User analytics\n'
        '2. The connection between doomscrolling and sleep'
    '')
    command=int(input('Please enter: '))
    if command==0:
        exit()
    elif command==1:
        user_analytics()
    elif command==2:
        doomscrolling_sleep()
    else:
        print('Unknown command')