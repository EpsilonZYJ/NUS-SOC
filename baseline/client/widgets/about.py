import dearpygui.dearpygui as dpg
from system import global_width, global_height

def show_about():
    # 检查窗口是否已经存在
    if dpg.does_item_exist("about_window"):
        # 如果存在，直接显示
        dpg.show_item("about_window")
    else:
        with dpg.window(
            label="About",
            modal=True,
            show=True,
            tag="about_window",
            pos=[global_width//4, global_height//4],
            width=global_width//2,
            height=global_height//2,
            on_close=lambda: dpg.hide_item("about_window")
        ):
            dpg.add_text("[Group 19] Server client")
            dpg.add_text("Version 1.2")
            dpg.add_separator()
            dpg.add_text("This is a server client application.")
            dpg.add_text("Features:")
            dpg.add_text("- Serve as a client for server inference")
            dpg.add_text("- Choose model files to predict")
            dpg.add_text("- Show real-time pictures")
            dpg.add_text("- Show prediction results")
            dpg.add_text("- Show animation characters simliar to the picture")
            dpg.add_text("- Data backup and restore")
            # dpg.add_text("• Menu bars")
            # dpg.add_text("• Themes and styling")
            dpg.add_separator()
            
            dpg.add_text("Developed by 404 Found")
            dpg.add_text("Group members:")
            dpg.add_text("Liao Zitao, Zhou Yujie, Cheng Shutong, Ge Yandu")
            
            dpg.add_separator()
            dpg.add_text("Architecture: Dear PyGui")
            # dpg.add_button(label="Close", callback=lambda: dpg.delete_item("about_window"))