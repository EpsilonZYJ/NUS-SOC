import os
import dearpygui.dearpygui as dpg
from system.global_data import selected_model_path, selected_image_path, available_models
from system import client, model_dict

# def scan_files_in_directory(directory, extensions=['.jpg', '.png', '.keras', '.h5']):
#     """扫描目录中的指定类型文件"""
#     files = []
#     if os.path.exists(directory):
#         for file in os.listdir(directory):
#             if any(file.lower().endswith(ext) for ext in extensions):
#                 files.append(file)
#     return files

# def combo_file_callback(sender, app_data, user_data):
#     """下拉框选择文件的回调函数"""
#     global selected_file_path
#     selected_filename = app_data
    
#     # 构建完整路径
#     base_directory = "./data"  # 您的文件目录
#     selected_file_path = os.path.join(base_directory, selected_filename)
    
#     print(f"Selected file: {selected_file_path}")
    
#     # 更新状态显示
#     dpg.set_value("status_text", f"Selected: {selected_filename}")
#     dpg.set_value("file_path_text", f"Path: {selected_file_path}")
    
#     # 根据文件类型执行不同操作
#     if selected_filename.lower().endswith(('.jpg', '.png', '.jpeg')):
#         process_image_file(selected_file_path)
#     elif selected_filename.lower().endswith(('.keras', '.h5')):
#         process_model_file(selected_file_path)

# def process_image_file(file_path):
#     """处理图像文件"""
#     print(f"Processing image: {file_path}")
#     dpg.set_value("status_text", f"Image loaded: {os.path.basename(file_path)}")
#     # 在这里添加图像处理逻辑

# def process_model_file(file_path):
#     """处理模型文件"""
#     print(f"Processing model: {file_path}")
#     dpg.set_value("status_text", f"Model loaded: {os.path.basename(file_path)}")
#     # 在这里添加模型加载逻辑

# def refresh_file_list():
#     """刷新文件列表"""
#     global file_list
#     directory = "./data"  # 您的文件目录
#     file_list = scan_files_in_directory(directory)
    
#     # 更新下拉框选项
#     if dpg.does_item_exist("file_combo"):
#         dpg.configure_item("file_combo", items=file_list)
    
#     print(f"Found {len(file_list)} files: {file_list}")

# def refresh_file_list_from_input():
#     """从输入框获取目录并刷新文件列表"""
#     directory = dpg.get_value("directory_input")
    
#     # 获取过滤器设置
#     filter_images = dpg.get_value("filter_images")
#     filter_models = dpg.get_value("filter_models")
#     filter_all = dpg.get_value("filter_all")
    
#     # 构建扩展名列表
#     extensions = []
#     if filter_images:
#         extensions.extend(['.jpg', '.jpeg', '.png', '.bmp', '.tiff'])
#     if filter_models:
#         extensions.extend(['.keras', '.h5', '.pkl', '.joblib'])
#     if filter_all:
#         extensions = ['']  # 空字符串表示所有文件
    
#     # 扫描文件
#     global file_list
#     file_list = scan_files_in_directory(directory, extensions)
    
#     # 更新下拉框
#     if file_list:
#         dpg.configure_item("file_combo", items=file_list, default_value=file_list[0])
#         dpg.set_value("status_text", f"Found {len(file_list)} files")
#     else:
#         dpg.configure_item("file_combo", items=["No files found"], default_value="No files found")
#         dpg.set_value("status_text", "No files found in directory")

# def load_selected_file():
#     """加载选中的文件"""
#     global selected_file_path
#     if selected_file_path and os.path.exists(selected_file_path):
#         dpg.set_value("status_text", f"Loading: {os.path.basename(selected_file_path)}")
#         # 在这里添加文件加载逻辑
#         print(f"Loading file: {selected_file_path}")
#     else:
#         dpg.set_value("status_text", "No valid file selected")

# def process_selected_file():
#     """处理选中的文件"""
#     global selected_file_path
#     if selected_file_path and os.path.exists(selected_file_path):
#         file_ext = os.path.splitext(selected_file_path)[1].lower()
        
#         if file_ext in ['.jpg', '.jpeg', '.png']:
#             # 图像处理逻辑
#             dpg.set_value("status_text", f"Processing image: {os.path.basename(selected_file_path)}")
#             # 添加您的图像处理代码
            
#         elif file_ext in ['.keras', '.h5']:
#             # 模型处理逻辑
#             dpg.set_value("status_text", f"Loading model: {os.path.basename(selected_file_path)}")
#             # 添加您的模型加载代码
            
#         else:
#             dpg.set_value("status_text", f"Processing file: {os.path.basename(selected_file_path)}")
#             # 通用文件处理
#     else:
#         dpg.set_value("status_text", "No valid file selected for processing")

# # 改进的文件扫描函数
# def scan_files_in_directory(directory, extensions=['.jpg', '.png', '.keras', '.h5']):
#     """扫描目录中的指定类型文件"""
#     files = []
#     try:
#         if os.path.exists(directory):
#             for file in os.listdir(directory):
#                 file_path = os.path.join(directory, file)
#                 # 只处理文件，不处理目录
#                 if os.path.isfile(file_path):
#                     # 如果 extensions 为空或包含空字符串，则包含所有文件
#                     if not extensions or '' in extensions:
#                         files.append(file)
#                     else:
#                         # 检查文件扩展名
#                         if any(file.lower().endswith(ext.lower()) for ext in extensions):
#                             files.append(file)
#     except Exception as e:
#         print(f"Error scanning directory {directory}: {e}")
    
#     return sorted(files)  # 排序文件列表

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
        dpg.set_value("model_status_text", f"Selected Model: {app_data}")
        dpg.set_value("model_path_display", f"Path: {selected_model_path}")
        client.load_model(selected_model_path)  # 加载模型
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

# 预测功能相关回调函数
def update_prediction_button_state():
    global selected_model_path, selected_image_path
    # 只有当模型和图片都选择了才能进行预测
    can_predict = bool(selected_model_path and selected_image_path)
    dpg.configure_item("predict_button", enabled=can_predict)
    
    # 更新表格状态
    if dpg.does_item_exist("table_model_status"):
        model_status = os.path.basename(selected_model_path) if selected_model_path else "Not Selected"
        dpg.set_value("table_model_status", model_status)
    
    if dpg.does_item_exist("table_image_status"):
        image_status = os.path.basename(selected_image_path) if selected_image_path else "Not Selected"
        dpg.set_value("table_image_status", image_status)
    
    if can_predict:
        dpg.set_value("prediction_status", "Ready to predict")
        dpg.configure_item("prediction_status", color=(0, 255, 0))
    else:
        dpg.set_value("prediction_status", "Please select both model and image")
        dpg.configure_item("prediction_status", color=(255, 255, 0))

def start_prediction():
    if not (selected_model_path and selected_image_path):
        dpg.set_value("prediction_result", "Error: Please select both model and image")
        return
    
    # 这里应该调用实际的预测函数
    dpg.set_value("prediction_status", "Predicting...")
    dpg.configure_item("prediction_status", color=(255, 255, 0))
    
    try:
        # 在实际应用中，这里应该调用model_image_handler中的预测函数
        result = f"Prediction completed!\nModel: {os.path.basename(selected_model_path)}\nImage: {os.path.basename(selected_image_path)}\n\nResult: Sample prediction result"
        dpg.set_value("prediction_result", result)
        dpg.set_value("prediction_status", "Prediction completed")
        dpg.configure_item("prediction_status", color=(0, 255, 0))
        
        # 更新表格中的最后预测状态
        if dpg.does_item_exist("table_last_prediction"):
            dpg.set_value("table_last_prediction", "Success")
            
    except Exception as e:
        dpg.set_value("prediction_result", f"Prediction failed: {str(e)}")
        dpg.set_value("prediction_status", "Prediction failed")
        dpg.configure_item("prediction_status", color=(255, 0, 0))
        
        # 更新表格中的最后预测状态
        if dpg.does_item_exist("table_last_prediction"):
            dpg.set_value("table_last_prediction", "Failed")
