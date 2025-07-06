import math

# global variables to store the current viewport size
global_width = 1200
global_height = 800

plot_data_x = []
plot_data_y = []
for i in range(100):
    plot_data_x.append(i / 10.0)
    plot_data_y.append(math.sin(i / 10.0))

selected_file_path = ""
file_list = []

selected_model_path = ""
selected_image_path = ""
available_models = ["efficientnet", "xception", "insection", "test"]
