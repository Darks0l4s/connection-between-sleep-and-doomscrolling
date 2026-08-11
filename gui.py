import customtkinter as ctk
import data
data.load_data()
def setup():
    app = ctk.CTk()
    ctk.set_appearance_mode('System')
    ctk.set_default_color_theme('blue')
    app.geometry('400x200')
    app.title('My app')
    btn_user_analytics = ctk.CTkButton(master=app, text='User analytics', command=data.user_analytics)
    btn_user_analytics.pack(pady=20)
    btn_doomscrolling_sleep = ctk.CTkButton(master=app, text='Dependence of sleep on parameters', command=data.doomscrolling_sleep)
    btn_doomscrolling_sleep.pack(pady=10)
    exit_btn = ctk.CTkButton(
    master=app,
    text="Выйти",
    command=app.destroy,  
    fg_color="#d9534f",
    hover_color="#c9302c" 
)
    exit_btn.pack(pady=10)
    app.mainloop()