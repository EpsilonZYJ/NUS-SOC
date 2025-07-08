import os
import dearpygui.dearpygui as dpg
from system.global_data import selected_model_path, selected_image_path, available_models
from system import client, model_dict
import base64
from PIL import Image
import io
import hashlib
import time
import json
import numpy as np

# 初始化可用模型列表
def initialize_models():
    global available_models
    model_dir = "./model"
    if not os.path.exists(model_dir):
        available_models = ["Model directory not found"]
        return available_models
    available_models_copy = available_models.copy()
    for md in available_models_copy:
        if not os.path.exists(model_dict[md]):
            available_models.remove(md)
    if available_models == []:
        available_models = ["No models found"]
    return available_models


# 模型选择相关回调函数
def model_combo_callback(sender, app_data, user_data):
    global selected_model_path
    if app_data and app_data != "No models found" and app_data != "Model directory not found":
        selected_model_path = model_dict.get(app_data, "")
        print(f"Selected model path: {selected_model_path}")
        print(app_data)
        dpg.set_value("model_status_text", f"Selected Model: {app_data}")
        dpg.set_value("model_path_display", f"Path: {selected_model_path}")
        client.init_model(app_data)  # 加载模型
        # 检查是否可以开始预测
        # update_prediction_button_state()
    else:
        dpg.set_value("model_status_text", "No valid model selected")
        dpg.set_value("model_path_display", "")
        

def refresh_models_callback():
    models = initialize_models()
    dpg.configure_item("model_combo", items=models)
    if models and models[0] != "No models found":
        dpg.set_value("model_combo", models[0])
        model_combo_callback(None, models[0], None)
    dpg.set_value("model_status_text", f"Models refreshed. Found {len(models)} model(s)")

# 文件选择相关回调函数
def image_file_dialog_callback(sender, app_data):
    global selected_image_path
    print(f"Image file dialog result: {app_data}")
    if app_data["file_path_name"]:
        selected_image_path = app_data["file_path_name"]
        filename = os.path.basename(selected_image_path)
        dpg.set_value("image_status_text", f"Selected Image: {filename}")
        dpg.set_value("image_path_display", f"Path: {selected_image_path}")
        # 检查是否可以开始预测
        update_prediction_button_state()

def clear_image_selection():
    global selected_image_path
    selected_image_path = ""
    dpg.set_value("image_status_text", "No image selected")
    dpg.set_value("image_path_display", "")
    update_prediction_button_state()

# 全局变量
current_directory = "results"  # 默认图片目录
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
loaded_texture_id = 0  # 当前加载的纹理ID
loaded_texture_id_cat = 0  # 当前加载的猫图片纹理ID
last_directory_state = ""  # 上次目录状态的哈希值
refresh_interval = 1.0  # 检查间隔（秒）
auto_refresh_enabled = True  # 是否启用自动刷新

cat_image_dir_dict = {
    'cat_police': 'imgs/cat_police/',
    'chiikawa': 'imgs/chiikawa/',
    'doraemon': 'imgs/doraemon/',
    'garfield': 'imgs/garfield/',
    'hellokitty': 'imgs/hellokitty/',
    'hongmao': 'imgs/hongmao/',
    'lingjiecat': 'imgs/lingjiecat/',
    'puss_in_boots': 'imgs/puss_in_boots/',
    'luna': 'imgs/luna/',
    'sensei': 'imgs/sensei/',
    'tom': 'imgs/tom/',
    'yuumi': 'imgs/yuumi/'
}

def random_choose_cat_from_dir(dirpath):
    print("Here is the dirpath:", dirpath)
    if not dirpath:
        return None
    if not os.path.exists(dirpath):
        print(f"Directory does not exist: {dirpath}")
        return None
    files = [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f)) and is_image_file(f)]
    print(files)
    if not files:
        print(f"No valid image files found in directory: {dirpath}")
        return None
    random_file = np.random.choice(files)
    print(f"Randomly selected file: {random_file}")
    return os.path.join(dirpath, random_file)

def is_image_file(filename):
    """检查文件是否为支持的图片格式"""
    return any(filename.lower().endswith(ext) for ext in image_extensions)

def scan_directory_for_images(directory, max_deppth=3, current_depth=0):
    """扫描指定目录及其子目录下的图片文件"""
    result = {"directories": {}, "files": []}
    try:
        if os.path.exists(directory):
            for item in os.listdir(directory):
                full_path = os.path.join(directory, item)
                if os.path.isdir(full_path):
                    # 递归扫描子目录
                    result["directories"][item] = scan_directory_for_images(full_path, max_deppth, current_depth + 1)
                elif os.path.isfile(full_path) and is_image_file(item):
                    # 添加图片文件
                    result["files"].append(item)
        
        # 排序文件和目录名
        result["files"].sort()
    except Exception as e:
        print(f"Error scanning directory {directory}: {e}")
    
    return result

def build_file_tree(directory_data, parent="file_tree", directory_path=""):
    """构建文件树界面"""
    # 首先添加目录
    for dir_name, content in directory_data["directories"].items():
        dir_path = os.path.join(directory_path, dir_name)
        dir_tag = f"dir_{dir_path}".replace('/', '_').replace('\\', '_')
        
        # 添加目录节点
        with dpg.tree_node(label=dir_name, tag=dir_tag, parent=parent):
            # 递归添加子目录和文件
            build_file_tree(content, dir_tag, dir_path)
    
    # 然后添加文件
    for file_name in directory_data["files"]:
        file_path = os.path.join(directory_path, file_name)
        full_path = os.path.join(current_directory, file_path)
        
        # 添加文件节点，使用lambda捕获当前文件路径
        dpg.add_button(
            label=file_name,
            callback=lambda s, a, u: load_and_display_image(u),
            user_data=full_path,
            width=-1,
            indent=20,
            parent=parent
        )

def refresh_file_explorer():
    """刷新文件浏览器"""
    global current_directory
    
    # 清除现有树结构
    if dpg.does_item_exist("file_explorer_section"):
        dpg.delete_item("file_explorer_section")
    
    # 创建新的文件树结构
    with dpg.collapsing_header(label="File Explorer", default_open=True, tag="file_explorer_section", parent="left_panel"):
        dpg.add_text("Images Directory:", color=(0, 255, 0))
        
        # 目录输入和浏览按钮
        with dpg.group(horizontal=True):
            dpg.add_input_text(
                tag="explorer_directory_input",
                default_value=current_directory,
                width=290
            )
            dpg.add_button(
                label="Browse",
                callback=lambda: dpg.show_item("directory_selector"),
                width=90
            )
        
        # 刷新按钮
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Refresh",
                callback=refresh_file_explorer_with_path,
                width=190
            )
            dpg.add_text(f"", tag="file_count_text", color=(150, 150, 150))
        
        dpg.add_separator()
        
        # 创建文件树
        directory_data = scan_directory_for_images(current_directory)
        total_files = count_files_in_data(directory_data)
        dpg.set_value("file_count_text", f"Found {total_files} image(s)")
        
        with dpg.tree_node(label="Images", default_open=True, tag="file_tree"):
            build_file_tree(directory_data)

def count_files_in_data(directory_data):
    """计算目录数据中的文件总数"""
    count = len(directory_data["files"])
    for _, subdir_data in directory_data["directories"].items():
        count += count_files_in_data(subdir_data)
    return count

def refresh_file_explorer_with_path():
    """从输入框获取路径并刷新文件浏览器"""
    global current_directory
    new_dir = dpg.get_value("explorer_directory_input")
    if os.path.exists(new_dir) and os.path.isdir(new_dir):
        current_directory = new_dir
        refresh_file_explorer()
    else:
        dpg.set_value("status_text", f"Invalid directory: {new_dir}")

def get_right_panel_size():
    """获取右侧面板的尺寸"""
    if dpg.does_item_exist("right_panel"):
        return dpg.get_item_rect_size("right_panel")
    return (800, 600)  # 默认大小

def load_image_result(image_path, json_path='result_image.json'):
    with open(json_path, 'r') as f:
        data = json.load(f)
        for item in data:
            # if item.get("image_path") == image_path:
            if os.path.basename(item.get("image_path", "")) == os.path.basename(image_path):
                return item
    return None
        

def load_and_display_image(image_path):
    """加载并显示图片"""
    global selected_image_path, loaded_texture_id, loaded_texture_id_cat
    
    if not os.path.exists(image_path):
        dpg.set_value("status_text", f"Error: Image file not found: {image_path}")
        return
    
    print(f"Loading image: {image_path}")
    item = load_image_result(image_path)
    if item is None:
        dpg.set_value("status_text", f"Image info loaded with result: {item}")    

    try:
        score = item.get("score", -1.0)
        prediction = item.get("prediction", "No result")
        if not dpg.does_item_exist("prediction_result"):
            with dpg.group(tag="prediction_result", parent="right_panel"):
                dpg.add_text(f"Prediction: {prediction}", tag="prediction_result_text")
                dpg.add_text(f"Source: {image_path}", tag="prediction_source_text")
                dpg.add_text(f"Score: {score:.2f}" if score >= 0 else "Score: N/A", tag="prediction_score_text")
        else:
            dpg.set_value("prediction_result_text", f"Prediction: {prediction}")
            dpg.set_value("prediction_source_text", f"Source: {image_path}")
            dpg.set_value("prediction_score_text", f"Score: {score:.2f}" if score >= 0 else "Score: N/A")
    except Exception as e:
        print(f"Error displaying prediction result: {e}")
        dpg.set_value("status_text", f"Error displaying prediction result: {str(e)}")

    try:
        # 更新选定的图片路径
        selected_image_path = image_path
        filename = os.path.basename(image_path)
        dpg.set_value("image_status_text", f"Selected Image: {filename}")
        dpg.set_value("image_path_display", f"Path: {image_path}")
        dpg.set_value("status_text", f"Loaded image: {filename}")
        
        print(image_path)

        # 获取右侧面板尺寸
        panel_width, panel_height = get_right_panel_size()
        
        # 为图片预览区域留出空间（减去标题和边距）
        max_img_width = panel_width - 40  # 左右边距
        max_img_height = panel_height - 100  # 上下边距及其他控件
        
        # 加载图片并创建纹理
        img = Image.open(image_path)
        
        # 调整图片大小以适应显示区域，保持纵横比
        max_size = (max_img_width, max_img_height)  # 最大显示尺寸
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # 转换为RGB模式（处理RGBA等其他模式）
        # if img.mode != "RGB":
        img = img.convert("RGBA")
        
        img_array = np.array(img, dtype=np.float32) / 255.0

        # 展平数组（DearPyGui需要一维数组）
        img_data = img_array.flatten()
        # 获取尺寸
        width, height = img.size
        
        # 将图片转换为数据
        data = bytearray(img.tobytes())
        
        # 如果已有纹理，先删除
        if loaded_texture_id != 0 and dpg.does_item_exist(loaded_texture_id):
            dpg.delete_item(loaded_texture_id)
        
        # 创建新纹理
        with dpg.texture_registry():
            loaded_texture_id = dpg.add_static_texture(width, height, img_data)
        
        # 计算图片水平和垂直居中位置
        pos_x = (max_img_width - width) // 2
        pos_y = 40 + (max_img_height - height) // 2  # 标题下方留出空间，然后垂直居中
    
        # 确保右侧已有图片显示区域
        if not dpg.does_item_exist("image_display_area"):
            with dpg.group(tag="image_display_area", parent="right_panel"):
                dpg.add_text(f"Size: {width}x{height}", tag="image_size_text")
                dpg.add_text("Image Preview:", color=(255, 255, 0))
                # 创建一个child_window来容纳图片，使其在滚动区域内居中
                with dpg.child_window(width=max_img_width, height=max_img_height, tag="image_container", no_scroll_with_mouse=True):
                    dpg.add_image(loaded_texture_id, tag="displayed_image", pos=[pos_x, pos_y])
        else:
            # 更新现有图片和尺寸信息
            dpg.configure_item("displayed_image", texture_tag=loaded_texture_id, pos=[pos_x, pos_y])
            dpg.set_value("image_size_text", f"Size: {width}x{height}")
        
        # # 更新选中的图片（用于预测）
        # if "update_prediction_button_state" in globals():
        #     update_prediction_button_state()
        
    except Exception as e:
        dpg.set_value("status_text", f"Error loading image: {str(e)}")
        print(f"Error loading image {image_path}: {e}")
    
    try:
        img_path = random_choose_cat_from_dir(
            cat_image_dir_dict.get(item.get("cat_match_prediction", ""))
        )
        print(f"Cat image path: {img_path}")
        if img_path is None:
            img_path = "imgs/404.jpg"
        # 为图片预览区域留出空间（减去标题和边距）
        max_img_width = panel_width - 40  # 左右边距
        max_img_height = panel_height - 100  # 上下边距及其他控件
        print(img_path)
        # 加载图片并创建纹理
        img = Image.open(img_path)

        original_width, original_height = img.size
        scale_x = max_img_width / original_width
        scale_y = max_img_height / original_height
        scale = min(scale_x, scale_y)  # 保持纵横比
        
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        img =  img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 调整图片大小以适应显示区域，保持纵横比
        # max_size = (max_img_width, max_img_height)  # 最大显示尺寸
        # img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # 转换为RGB模式（处理RGBA等其他模式）
        # if img.mode != "RGB":
        img = img.convert("RGBA")
        
        img_array = np.array(img, dtype=np.float32) / 255.0

        # 展平数组（DearPyGui需要一维数组）
        img_data = img_array.flatten()
        # 获取尺寸
        width, height = img.size
        
        # 将图片转换为数据
        data = bytearray(img.tobytes())
        
        # 如果已有纹理，先删除
        if loaded_texture_id_cat != 0 and dpg.does_item_exist(loaded_texture_id_cat):
            dpg.delete_item(loaded_texture_id_cat)
        
        # 创建新纹理
        with dpg.texture_registry():
            loaded_texture_id_cat = dpg.add_static_texture(width, height, img_data)
        
        # 计算图片水平和垂直居中位置
        pos_x = max(0, (max_img_width - width) // 2)
        pos_y = max(0, (max_img_height - height) // 2)  # 标题下方留出空间，然后垂直居中
    
        # 确保右侧已有图片显示区域
        if not dpg.does_item_exist("image_display_area_cat"):
            with dpg.group(tag="image_display_area_cat", parent="right_panel"):
                # dpg.add_text(f"Size: {width}x{height}", tag="image_size_text")
                dpg.add_text("Fun:", color=(255, 255, 0))
                # 创建一个child_window来容纳图片，使其在滚动区域内居中
                with dpg.child_window(width=max_img_width, height=max_img_height, tag="image_container_cat", no_scroll_with_mouse=True):
                    dpg.add_image(loaded_texture_id_cat, tag="displayed_image_cat", pos=[pos_x, pos_y])
        else:
            # 更新现有图片和尺寸信息
            dpg.configure_item("displayed_image_cat", texture_tag=loaded_texture_id_cat, pos=[pos_x, pos_y])
            # dpg.set_value("image_size_text", f"Size: {width}x{height}")
    except Exception as e:
        print(f"Error loading cat image: {e}")
    try:
        cat_score = item.get("cat_match_score", -1.0)
        cat_prediction = item.get("cat_match_prediction", "No result")
        if not dpg.does_item_exist("prediction_result_cat"):
            with dpg.group(tag="prediction_result_cat", parent="right_panel"):
                dpg.add_text(f"Prediction: {cat_prediction}", tag="prediction_result_text_cat")
                dpg.add_text(f"Score: {cat_score:.2f}" if cat_score >= 0 else "Score: N/A", tag="prediction_score_text_cat")
        else:
            dpg.set_value("prediction_result_text_cat", f"Prediction: {cat_prediction}")
            dpg.set_value("prediction_score_text_cat", f"Score: {cat_score:.2f}" if cat_score >= 0 else "Score: N/A")
    except Exception as e:
        print(f"Error displaying cat prediction result: {e}")
        # # 更新选中的图片（用于预测）
        # if "update_prediction_button_state" in globals():
        #     update_prediction_button_state() 

# 目录选择对话框回调
def directory_selector_callback(sender, app_data):
    if app_data["file_path_name"]:
        # 对于目录选择器，我们使用目录路径
        directory = app_data["file_path_name"]
        if os.path.isdir(directory):
            dpg.set_value("explorer_directory_input", directory)
            refresh_file_explorer_with_path()

# 创建目录选择对话框
def create_directory_selector():
    with dpg.file_dialog(
        directory_selector=True,
        show=False,
        callback=directory_selector_callback,
        tag="directory_selector",
        width=700,
        height=400,
    ):
        dpg.add_file_extension("", color=(255, 255, 255, 255))

def calculate_directory_hash(directory):
    """计算目录状态的哈希值，用于检测变化"""
    if not os.path.exists(directory):
        return ""
    
    files_info = []
    try:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if is_image_file(file):
                    full_path = os.path.join(root, file)
                    # 获取文件修改时间和大小
                    stat_info = os.stat(full_path)
                    files_info.append((full_path, stat_info.st_mtime, stat_info.st_size))
    except Exception as e:
        print(f"Error scanning directory for hash: {e}")
    
    # 排序以确保一致性
    files_info.sort()
    
    # 创建表示目录状态的字符串
    dir_state = str(files_info)
    
    # 计算哈希值
    return hashlib.md5(dir_state.encode()).hexdigest()

def check_directory_changes():
    """检查目录是否有变化，如果有则刷新文件浏览器"""
    global current_directory, last_directory_state
    
    if not auto_refresh_enabled:
        return
    
    # 计算当前目录状态的哈希值
    current_hash = calculate_directory_hash(current_directory)
    
    # 如果哈希值与上次不同，则目录有变化
    if current_hash != last_directory_state and last_directory_state != "":
        print(f"Directory changes detected in {current_directory}, refreshing...")
        refresh_file_explorer()
    
    # 更新哈希值
    last_directory_state = current_hash

def setup_auto_refresh():
    """设置自动刷新定时器"""
    global last_directory_state, current_directory, refresh_interval
    
    # 计算初始目录哈希值
    last_directory_state = calculate_directory_hash(current_directory)
    
    # 使用DearPyGui的帧计数回调实现定期检查
    def frame_callback(sender, data):
        # 检查目录变化
        if auto_refresh_enabled:
            check_directory_changes()
        
        # 设置下一次检查（每60帧一次，约1秒）
        frames_until_next_check = int(refresh_interval * 60)
        dpg.set_frame_callback(dpg.get_frame_count() + frames_until_next_check, frame_callback)
    
    # 设置初始回调，延迟60帧后开始（约1秒）
    dpg.set_frame_callback(dpg.get_frame_count() + 60, frame_callback)

def toggle_auto_refresh(sender, app_data, user_data):
    """切换自动刷新功能的开关"""
    global auto_refresh_enabled
    auto_refresh_enabled = not auto_refresh_enabled
    
    # 更新按钮文本
    if auto_refresh_enabled:
        dpg.configure_item("auto_refresh_button", label="Auto-refresh: ON")
        dpg.set_value("status_text", f"Auto-refresh enabled (every {refresh_interval:.1f}s)")
    else:
        dpg.configure_item("auto_refresh_button", label="Auto-refresh: OFF")
        dpg.set_value("status_text", "Auto-refresh disabled")

# 修改refresh_file_explorer函数，加入自动刷新按钮
def refresh_file_explorer():
    """刷新文件浏览器"""
    global current_directory, last_directory_state
    
    # 清除现有树结构
    if dpg.does_item_exist("file_explorer_section"):
        dpg.delete_item("file_explorer_section")
    
    # 创建新的文件树结构
    with dpg.collapsing_header(label="Image Explorer", default_open=True, tag="file_explorer_section", parent="left_panel"):
        # dpg.add_text("Images Directory:", color=(0, 255, 0))
        
        # 目录输入和浏览按钮
        with dpg.group(horizontal=True, show=False):
            dpg.add_input_text(
                tag="explorer_directory_input",
                default_value=current_directory,
                width=290
            )
            dpg.add_button(
                label="Browse",
                callback=lambda: dpg.show_item("directory_selector"),
                width=90
            )
        
        # 刷新和自动刷新按钮
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Refresh",
                callback=refresh_file_explorer_with_path,
                width=190
            )
            dpg.add_button(
                label="Auto-refresh: ON" if auto_refresh_enabled else "Auto-refresh: OFF",
                callback=toggle_auto_refresh,
                width=190,
                tag="auto_refresh_button"
            )
        
        dpg.add_text(f"", tag="file_count_text", color=(150, 150, 150))
        dpg.add_separator()
        
        # 创建文件树
        directory_data = scan_directory_for_images(current_directory)
        total_files = count_files_in_data(directory_data)
        dpg.set_value("file_count_text", f"Found {total_files} image(s)")
        
        with dpg.tree_node(label="Images", default_open=True, tag="file_tree"):
            build_file_tree(directory_data)
    
    # 更新当前目录状态的哈希值
    last_directory_state = calculate_directory_hash(current_directory)