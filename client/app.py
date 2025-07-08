import dearpygui.dearpygui as dpg
import math
import random
import os
from widgets import *
from system import client, cat_found_dict, cat_found_status_code
from utils import create_backup_on_exit


# 创建上下文
dpg.create_context()


# # 原有回调函数
# def button_callback(sender, app_data, user_data):
#     print(f"Button {user_data} was clicked!")
#     dpg.set_value(
#         "status_text", f"Button {user_data} was clicked at {dpg.get_total_time():.2f}s"
#     )


# def slider_callback(sender, app_data, user_data):
#     print(f"Slider value: {app_data}")
#     dpg.set_value("slider_value_text", f"Slider Value: {app_data:.2f}")


# def input_callback(sender, app_data, user_data):
#     print(f"Input text: {app_data}")
#     dpg.set_value("input_display", f"You typed: {app_data}")


def add_random_data():
    global plot_data_x, plot_data_y
    new_x = len(plot_data_x) / 10.0
    new_y = math.sin(new_x) + random.uniform(-0.5, 0.5)
    plot_data_x.append(new_x)
    plot_data_y.append(new_y)

    # 保持数据点在合理范围内
    if len(plot_data_x) > 200:
        plot_data_x.pop(0)
        plot_data_y.pop(0)

    dpg.set_value("plot_series", [plot_data_x, plot_data_y])


def clear_plot():
    global plot_data_x, plot_data_y
    plot_data_x.clear()
    plot_data_y.clear()
    dpg.set_value("plot_series", [plot_data_x, plot_data_y])


# def show_about():
#     with dpg.window(
#         label="About",
#         modal=True,
#         show=True,
#         tag="about_window",
#         pos=[200, 200],
#         width=300,
#         height=200,
#     ):
#         dpg.add_text("Dear PyGui Example Application")
#         dpg.add_text("Version 1.0")
#         dpg.add_separator()
#         dpg.add_text("This is a comprehensive example showcasing:")
#         dpg.add_text("• Basic widgets")
#         dpg.add_text("• Real-time plotting")
#         dpg.add_text("• Menu bars")
#         dpg.add_text("• Themes and styling")
#         dpg.add_separator()
#         dpg.add_button(label="Close", callback=lambda: dpg.delete_item("about_window"))


def file_dialog_callback(sender, app_data):
    print(f"File dialog result: {app_data}")
    if app_data["file_path_name"]:
        dpg.set_value("file_path_text", f"Selected: {app_data['file_path_name']}")


# 设置字体
with dpg.font_registry():
    import os

    # 检查字体文件是否存在
    font_file = "./fonts/lmmonolt10-regular.otf"

    if os.path.exists(font_file):
        try:
            # 加载自定义字体，调整大小为 18px（原来是 10px）
            default_font = dpg.add_font(font_file, 25)
            print(f"Custom OTF font loaded: {default_font} (size: 18px)")
        except Exception as e:
            print(f"Error loading custom font: {e}")
            # 使用系统默认字体作为备用
            default_font = dpg.add_font("", 16)
            print(f"Fallback to system font: {default_font}")
    else:
        print(f"Font file not found: {font_file}")
        # 使用系统默认字体
        default_font = dpg.add_font("", 16)
        print(f"Using system default font: {default_font}")

# 创建主题
with dpg.theme() as global_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(
            dpg.mvThemeCol_WindowBg, (15, 15, 15), category=dpg.mvThemeCat_Core
        )
        dpg.add_theme_color(
            dpg.mvThemeCol_FrameBg, (45, 45, 45), category=dpg.mvThemeCat_Core
        )
        dpg.add_theme_color(
            dpg.mvThemeCol_Button, (70, 70, 70), category=dpg.mvThemeCat_Core
        )
        dpg.add_theme_color(
            dpg.mvThemeCol_ButtonHovered, (100, 100, 100), category=dpg.mvThemeCat_Core
        )
        dpg.add_theme_color(
            dpg.mvThemeCol_ButtonActive, (120, 120, 120), category=dpg.mvThemeCat_Core
        )
        dpg.add_theme_style(
            dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core
        )
        dpg.add_theme_style(
            dpg.mvStyleVar_WindowRounding, 5, category=dpg.mvThemeCat_Core
        )
        # 防止字体压缩的关键设置
        dpg.add_theme_style(
            dpg.mvStyleVar_WindowPadding, 10, 10, category=dpg.mvThemeCat_Core
        )
        dpg.add_theme_style(
            dpg.mvStyleVar_ItemSpacing, 6, 6, category=dpg.mvThemeCat_Core
        )
        dpg.add_theme_style(
            dpg.mvStyleVar_ItemInnerSpacing, 4, 4, category=dpg.mvThemeCat_Core
        )
        dpg.add_theme_style(
            dpg.mvStyleVar_FramePadding, 8, 6, category=dpg.mvThemeCat_Core
        )

# 创建图片文件对话框
with dpg.file_dialog(
    directory_selector=False,
    show=False,
    callback=image_file_dialog_callback,
    tag="image_file_dialog",
    width=700,
    height=400,
):
    dpg.add_file_extension(".*")
    dpg.add_file_extension("", color=(255, 255, 255, 255))
    dpg.add_file_extension(".jpg", color=(255, 255, 0, 255))
    dpg.add_file_extension(".jpeg", color=(255, 255, 0, 255))
    dpg.add_file_extension(".png", color=(255, 255, 0, 255))
    dpg.add_file_extension(".bmp", color=(255, 255, 0, 255))

# 创建通用文件对话框
with dpg.file_dialog(
    directory_selector=False,
    show=False,
    callback=file_dialog_callback,
    tag="file_dialog",
    width=700,
    height=400,
):
    dpg.add_file_extension(".*")
    dpg.add_file_extension("", color=(255, 255, 255, 255))
    dpg.add_file_extension(".py", color=(255, 255, 0, 255))
    dpg.add_file_extension(".txt", color=(0, 255, 0, 255))

# 创建主窗口
with dpg.window(label="Dear PyGui Example", tag="Primary Window", width=-1, height=-1, 
                no_resize=False, no_move=False, no_scrollbar=False, no_collapse=False):

    # 菜单栏
    with dpg.menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(
                label="Open", callback=lambda: dpg.show_item("file_dialog")
            )
            dpg.add_menu_item(label="Save")
            dpg.add_separator()
            dpg.add_menu_item(label="Exit", callback=lambda: dpg.stop_dearpygui())

        with dpg.menu(label="Edit"):
            dpg.add_menu_item(label="Cut")
            dpg.add_menu_item(label="Copy")
            dpg.add_menu_item(label="Paste")

        with dpg.menu(label="Help"):
            dpg.add_menu_item(label="About", callback=show_about)

    # 主内容区域
    with dpg.group(horizontal=True):

        # 左侧控制面板
        with dpg.child_window(width=400, height=-1, tag="left_panel", 
                             horizontal_scrollbar=False,
                             border=False):
            dpg.add_text("Control Panel", color=(255, 255, 0))
            dpg.add_separator()

            # 模型选择区域
            with dpg.collapsing_header(label="Model Selection", default_open=True):
                dpg.add_text("Select Model:", color=(0, 255, 0))
                
                with dpg.group(horizontal=True):
                    # 初始化模型列表
                    models = initialize_models()
                    dpg.add_combo(
                        items=models,
                        label="",
                        callback=model_combo_callback,
                        tag="model_combo",
                        width=290,
                        default_value=models[0] if models else "",
                        indent=5
                    )
                
                    # dpg.add_button(
                    #     label="Load",
                    #     callback=refresh_models_callback,
                    #     width=90
                    # )
                
                dpg.add_text("No model selected", tag="model_status_text", color=(255, 255, 0))
                dpg.add_text("", tag="model_path_display", color=(150, 150, 150))
                
            dpg.add_separator()

            # 在app.py中修改Cat Found Status部分
            with dpg.collapsing_header(label="Cat Found Status", default_open=True):
                # 添加总体统计信息
                dpg.add_text("Progress: 0/12 cats found", tag="cat_progress_text", color=(255, 255, 0))
                dpg.add_separator()
                
                # 为每个猫咪类型创建状态显示
                with dpg.group(tag="cat_found_status_group"):
                    for cat_name in cat_found_dict.keys():
                        with dpg.group(horizontal=True, tag=f"cat_status_group_{cat_name}"):
                            # dpg.add_text("✗", tag=f"cat_icon_{cat_name}", color=(255, 100, 100))
                            dpg.add_text(f"{cat_name.replace('_', ' ').title()}", tag=f"cat_name_{cat_name}")
                            dpg.add_text("Not Found", tag=f"cat_status_{cat_name}", color=(255, 100, 100))


            # 图片文件选择区域
            with dpg.collapsing_header(label="Image Selection", default_open=False, show=False):
                dpg.add_text("Select Image for Prediction:", color=(0, 255, 0))
                
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Browse Images",
                        callback=lambda: dpg.show_item("image_file_dialog"),
                        width=190,
                        indent=5
                    )
                    dpg.add_button(
                        label="Clear Image",
                        callback=clear_image_selection,
                        width=190
                    )
                
                dpg.add_text("No image selected", tag="image_status_text", color=(255, 255, 0))
                dpg.add_text("", tag="image_path_display", color=(150, 150, 150))
                
            dpg.add_separator()
            
            # with dpg.child_window(width=400, height=-1, tag="left_panel", 
            #                  horizontal_scrollbar=False,
            #                  border=False):
            #     dpg.add_text("Control Panel", color=(255, 255, 0))
            #     dpg.add_separator() 

            # dpg.add_separator()
            # # 预测控制区域
            # with dpg.collapsing_header(label="Prediction Control", default_open=True):
            #     dpg.add_text("Prediction Status:", color=(255, 200, 200))
            #     dpg.add_text("Please select both model and image", tag="prediction_status", color=(255, 255, 0))
                
            #     dpg.add_button(
            #         label="Start Prediction",
            #         callback=start_prediction,
            #         tag="predict_button",
            #         width=200,
            #         height=40,
            #         enabled=False
            #     )
                
            # dpg.add_separator()


    
        # 右侧内容区域
        with dpg.child_window(width=-1, height=-1, tag="right_panel",
                             horizontal_scrollbar=False,
                             border=False):
            # dpg.add_text("Prediction Results", color=(0, 255, 255))
            # dpg.add_separator()

            # # 预测结果显示区域
            # with dpg.group():
            #     dpg.add_text("Prediction Output:", color=(255, 255, 0))
            #     dpg.add_input_text(
            #         tag="prediction_result",
            #         multiline=True,
            #         readonly=True,
            #         width=-1,
            #         height=200,
            #         default_value="No prediction yet. Please select a model and image, then click 'Start Prediction'."
            #     )

            # dpg.add_separator()

            # 状态显示（保留原有功能）
            with dpg.group(show=False, tag="status_group"):
                dpg.add_text("System Status:", color=(0, 255, 0))
                dpg.add_text("Status: Ready", tag="status_text", color=(0, 255, 0))
                dpg.add_text("", tag="file_path_text")

            dpg.add_separator()

            # 图表区域（保留原有功能）
            # with dpg.group():
            #     dpg.add_text("Real-time Plot", color=(255, 255, 0))

            #     with dpg.group(horizontal=True):
            #         dpg.add_button(label="Add Data Point", callback=add_random_data)
            #         dpg.add_button(label="Clear Plot", callback=clear_plot)

            #     with dpg.plot(label="Sine Wave with Noise", height=300, width=-1):
            #         dpg.add_plot_legend()
            #         dpg.add_plot_axis(dpg.mvXAxis, label="X Axis")
            #         dpg.add_plot_axis(dpg.mvYAxis, label="Y Axis", tag="y_axis")

            #         # dpg.add_line_series(
            #         #     plot_data_x,
            #         #     plot_data_y,
            #         #     label="Sine Wave",
            #         #     parent="y_axis",
            #         #     tag="plot_series",
            #         # )

            # dpg.add_separator()

            # # 数据表格（简化版）
            # with dpg.group():
            #     dpg.add_text("System Information", color=(255, 255, 0))

            #     with dpg.table(
            #         header_row=True,
            #         borders_innerH=True,
            #         borders_outerH=True,
            #         borders_innerV=True,
            #         borders_outerV=True,
            #     ):
            #         dpg.add_table_column(label="Property")
            #         dpg.add_table_column(label="Value")

            #         with dpg.table_row():
            #             dpg.add_text("Model Status")
            #             dpg.add_text("Not Selected", tag="table_model_status")
                    
            #         with dpg.table_row():
            #             dpg.add_text("Image Status")
            #             dpg.add_text("Not Selected", tag="table_image_status")
                    
            #         with dpg.table_row():
            #             dpg.add_text("Last Prediction")
            #             dpg.add_text("None", tag="table_last_prediction")

if __name__ == '__main__':
    # 设置主窗口
    dpg.set_primary_window("Primary Window", True)

    # 应用主题
    dpg.bind_theme(global_theme)

    # 应用字体（如果可用）
    if "default_font" in locals() and default_font:
        dpg.bind_font(default_font)

    # 创建目录选择对话框
    create_directory_selector()
    
    # 初始化文件浏览器
    refresh_file_explorer()

    setup_auto_refresh()

    dpg.set_exit_callback(create_backup_on_exit)  # 设置退出时的备份处理

    # 创建视口 - 设置最小尺寸以防止过度缩放
    dpg.create_viewport(title="Cat Classifier Sever Client - 404 Found", width=1400, height=900, 
                    resizable=True, min_width=1000, min_height=700, max_width=2400, max_height=1800)

    # 设置视口属性
    dpg.setup_dearpygui()

    # 设置DPI感知和字体缩放保护
    dpg.set_global_font_scale(1.0)  # 固定字体缩放比例
    if hasattr(dpg, 'set_viewport_vsync'):
        dpg.set_viewport_vsync(True)

    # 显示视口
    dpg.show_viewport()

    # 主循环
    dpg.start_dearpygui()

    # 清理
    dpg.destroy_context()
