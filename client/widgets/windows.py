import dearpygui.dearpygui as dpg

from system import global_width, global_height

def monitor_window_size():
    """监控窗口大小变化的函数"""
    global global_width, global_height
    
    # 获取当前视口大小
    viewport_width = dpg.get_viewport_width()
    viewport_height = dpg.get_viewport_height()
    
    # 检查是否发生变化
    if viewport_width != global_width or viewport_height != global_height:
        # 更新当前值
        global_width = viewport_width
        global_height = viewport_height
        
        # 更新状态文本显示当前窗口大小
        if dpg.does_item_exist("status_text"):
            dpg.set_value("status_text", f"size: {viewport_width} x {viewport_height}")

def window_resize_callback():
    """窗口大小改变时的回调函数"""
    monitor_window_size()

# 设置回调函数来监控窗口大小变化
def render_callback():
    """每帧渲染时调用，用于实时监控窗口大小"""
    monitor_window_size()
    
    # 更新显示的当前窗口大小
    if dpg.does_item_exist("current_size_text"):
        global global_width, global_height 
        
        global_width = dpg.get_viewport_width()
        global_height = dpg.get_viewport_height()
        dpg.set_value("current_size_text", f"当前大小: {global_height} x {global_height}")
