import dearpygui.dearpygui as dpg
from system import global_width, global_height

def show_about():
    with dpg.window(
        label="About",
        modal=True,
        show=True,
        tag="about_window",
        pos=[global_width//4, global_height//4],
        width=global_width//2,
        height=global_height//2,
    ):
        dpg.add_text("[Group 19] Server client")
        dpg.add_text("Version 1.0")
        dpg.add_separator()
        dpg.add_text("This is a server client application.")
        dpg.add_text("Features:")
        dpg.add_text("- Choose model files to predict")
        dpg.add_text("- Show real-time pictures and predictions")
        # dpg.add_text("• Menu bars")
        # dpg.add_text("• Themes and styling")
        dpg.add_separator()
        dpg.add_text("Architecture: Dear PyGui")
        dpg.add_text("Developed by 404 Found")
        # dpg.add_button(label="Close", callback=lambda: dpg.delete_item("about_window"))