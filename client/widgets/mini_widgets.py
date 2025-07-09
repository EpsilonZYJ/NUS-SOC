import dearpygui.dearpygui as dpg
from system.global_data import *

# 回调函数
def button_callback(sender, app_data, user_data):
    print(f"Button {user_data} was clicked!")
    dpg.set_value(
        "status_text", f"Button {user_data} was clicked at {dpg.get_total_time():.2f}s"
    )

def slider_callback(sender, app_data, user_data):
    print(f"Slider value: {app_data}")
    dpg.set_value("slider_value_text", f"Slider Value: {app_data:.2f}")

def input_callback(sender, app_data, user_data):
    print(f"Input text: {app_data}")
    dpg.set_value("input_display", f"You typed: {app_data}")